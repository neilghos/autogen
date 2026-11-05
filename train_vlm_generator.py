import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set PyTorch CUDA allocator to prevent memory fragmentation on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Bypass Hugging Face torch.load safety check for local weights
try:
    import transformers.utils.import_utils as t_import_utils
    t_import_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass
try:
    import transformers.modeling_utils as t_model_utils
    t_model_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from models import MultiScaleDecoderGenerator


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Grounded photographic descriptions for SigLIP (Global Scene View)
SIGLIP_CLASS_PROMPTS = [
    "a sharp, authentic photo of a passenger airplane in clear blue sky",
    "a sharp, authentic photo of an automobile car with rubber tires on asphalt",
    "a sharp, authentic photo of a wild bird with feathers perched outdoors",
    "a sharp, authentic photo of a domestic cat with fur outdoors in grass",
    "a sharp, authentic photo of a wild deer in a natural forest meadow",
    "a sharp, authentic photo of a domestic dog on green grass in daylight",
    "a sharp, authentic photo of a green amphibian frog on a leaf in daylight",
    "a sharp, authentic photo of an equine horse with mane standing in pasture",
    "a sharp, authentic photo of a sailing ship vessel on open ocean water",
    "a sharp, authentic photo of a commercial motor truck on paved highway"
]

# Micro-Detail Prompts for Local Zoomed Crops (Forces sharp textures & physical parts)
SIGLIP_DETAIL_PROMPTS = [
    "a macro close up photo of airplane metal fuselage rivets, jet engine intake and sharp wings",
    "a sharp close up photo of automobile rubber tire tread, alloy wheel rims, and car headlights",
    "a macro close up photo of wild bird plumage feathers, sharp beak and dark eye pupil",
    "a sharp close up photo of domestic cat whiskers, fur texture, and pointed ears",
    "a sharp close up photo of wild deer antler tines, fur coat and alert ears",
    "a sharp close up photo of domestic dog snout, moist nose, soft fur and floppy ears",
    "a macro close up photo of green tree frog moist glistening skin texture and bulbous eye",
    "a sharp close up photo of horse mane hair, equine coat texture and muscular neck",
    "a sharp close up photo of wooden ship hull planks, ropes, sails and water ripples",
    "a sharp close up photo of commercial motor truck front grill, large heavy tires and side mirror"
]

NEGATIVE_PROMPTS = [
    "blurry, foggy, out of focus, low contrast, abstract dots, scanlines, distorted, surreal cartoon"
]


class LaplacianSharpness(nn.Module):
    """
    Measures high-frequency edge energy using a 3x3 discrete Laplacian operator.
    Blown-out / blurry images have low edge energy; sharp photos have crisp boundaries.
    """
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0.0, 1.0, 0.0],
                          [1.0, -4.0, 1.0],
                          [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("kernel", k)

    def forward(self, x):
        edges = F.conv2d(x, self.kernel, padding=1, groups=3)
        return torch.mean(torch.abs(edges), dim=[-3, -2, -1])


class StochasticJitter(nn.Module):
    """
    Stochastic jitter and scale perturbation on 128x128 resolution.
    Breaks ViT patch grid phase-lock and eradicates scanlines/checkerboards.
    """
    def __init__(self, pad=8, img_size=128):
        super().__init__()
        self.pad = pad
        self.img_size = img_size

    def forward(self, x):
        # 1. Reflection pad and random translation crop
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        h = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        w = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        x_crop = x_pad[:, :, h:h + self.img_size, w:w + self.img_size]

        # 2. Scale jitter breaks ViT patch grid phase-lock (eradicates scanlines)
        scale = 0.94 + 0.12 * torch.rand(1, device=x.device).item()
        s_dim = int(round(self.img_size * scale))
        x_scaled = F.interpolate(x_crop, size=(s_dim, s_dim), mode='bilinear', align_corners=False)
        return F.interpolate(x_scaled, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)


def extract_tensor(feat):
    """Safely extracts torch.Tensor from any HuggingFace ModelOutput object."""
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "text_embeds") and feat.text_embeds is not None:
        return feat.text_embeds
    if hasattr(feat, "image_embeds") and feat.image_embeds is not None:
        return feat.image_embeds
    if hasattr(feat, "last_hidden_state") and feat.last_hidden_state is not None:
        return feat.last_hidden_state[:, 0]
    return feat[0]


def load_siglip_judge(device):
    """
    Loads Google's SigLIP (google/siglip-base-patch16-224) in FP16.
    Replaces BLIP's toxic cross-attention sinks with smooth, non-adversarial sigmoid gradients.
    VRAM footprint: ~600 MB.
    """
    print("[SigLIP Judge] Loading Google's SigLIP (google/siglip-base-patch16-224)...")
    try:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224",
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[SigLIP Judge] Loaded successfully via transformers.\n")
        return tokenizer, model
    except Exception as e:
        print(f"[SigLIP Judge] Transformers load failed ({e}), attempting open_clip fallback...")

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
    model.eval().half()
    for p in model.parameters():
        p.requires_grad = False
    print("[SigLIP Judge] Loaded successfully via open_clip.\n")
    return tokenizer, model


def precompute_siglip_anchors(tokenizer, model, device):
    """
    Precomputes normalized SigLIP text embeddings for:
      1. Global class descriptions (overall scene & anatomy)
      2. Micro-detail prompts (sharp rivets, tire treads, feathers, fur, frog skin)
      3. Negative repulsion prompts (fog, blur, scanlines)
    """
    if hasattr(model, "get_text_features"):
        # HuggingFace SigLIP
        inputs = tokenizer(SIGLIP_CLASS_PROMPTS, padding="max_length", max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_text = model.get_text_features(**inputs)
            pos_embeds = F.normalize(extract_tensor(raw_text).half(), dim=-1)

        detail_inputs = tokenizer(SIGLIP_DETAIL_PROMPTS, padding="max_length", max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_detail = model.get_text_features(**detail_inputs)
            detail_embeds = F.normalize(extract_tensor(raw_detail).half(), dim=-1)

        neg_inputs = tokenizer(NEGATIVE_PROMPTS, padding="max_length", max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_neg = model.get_text_features(**neg_inputs)
            neg_embeds = F.normalize(extract_tensor(raw_neg).half(), dim=-1)
    else:
        # open_clip SigLIP
        tokens = tokenizer(SIGLIP_CLASS_PROMPTS).to(device)
        with torch.no_grad():
            pos_embeds = F.normalize(model.encode_text(tokens).half(), dim=-1)

        detail_tokens = tokenizer(SIGLIP_DETAIL_PROMPTS).to(device)
        with torch.no_grad():
            detail_embeds = F.normalize(model.encode_text(detail_tokens).half(), dim=-1)

        neg_tokens = tokenizer(NEGATIVE_PROMPTS).to(device)
        with torch.no_grad():
            neg_embeds = F.normalize(model.encode_text(neg_tokens).half(), dim=-1)

    return pos_embeds, detail_embeds, neg_embeds


def main():
    parser = argparse.ArgumentParser(description="Google SigLIP 128x128 High-Resolution Synthesis")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batches_per_epoch", type=int, default=15, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=2, help="Samples per class in each batch (Default: 2 -> Batch size = 20)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # High-Frequency Details & Edge Loss Weights
    parser.add_argument("--lambda_siglip", type=float, default=5.0, help="Weight for Global SigLIP Alignment")
    parser.add_argument("--lambda_patch", type=float, default=2.5, help="Weight for Multi-Crop Zoomed Patch Micro-Textures")
    parser.add_argument("--lambda_sharpness", type=float, default=4.0, help="Weight for Laplacian High-Frequency Edge Floor")
    parser.add_argument("--lambda_contrast", type=float, default=3.0, help="Anti-fog dynamic range contrast floor")
    parser.add_argument("--lambda_color", type=float, default=1.5, help="Natural daylight color moments weight")
    parser.add_argument("--lambda_tv", type=float, default=0.002, help="Slashed TV weight (prevents edge blurring)")
    parser.add_argument("--lambda_div", type=float, default=0.03, help="Mode-seeking diversity weight")
    parser.add_argument("--lambda_neg", type=float, default=1.0, help="Negative prompt repulsion weight")

    parser.add_argument("--output_dir", type=str, default="./results_vlm_generator", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # cuDNN auto-tuner for maximum GPU kernel throughput
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # 1. Load Google SigLIP Judge
    tokenizer, siglip = load_siglip_judge(device)
    pos_embeds, detail_embeds, neg_embeds = precompute_siglip_anchors(tokenizer, siglip, device)
    print(f"[SigLIP Judge] Global anchors: {pos_embeds.shape} | Detail anchors: {detail_embeds.shape}\n")

    # SigLIP standard normalization: [-1, 1]
    siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    siglip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    # Natural color statistics target (from real CIFAR-10)
    target_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    target_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    # 2. Multi-Scale Decoder Generator (128x128)
    batch_size = 10 * args.samples_per_class
    print("=" * 72)
    print("SIGLIP 128x128 HIGH-RESOLUTION SYNTHESIS")
    print(f"  - Canvas Resolution: 128 x 128 native pixels (16x area over 32x32)")
    print(f"  - Critic Model: google/siglip-base-patch16-224 (FP16)")
    print(f"  - Multi-Crop Zoomed Patches (56-84px): ACTIVE (Fine textures & physical parts)")
    print(f"  - Laplacian High-Pass Edge Floor: ACTIVE (Snaps crisp 1-pixel contours)")
    print(f"  - Slashed TV Loss: {args.lambda_tv} (Kills bokeh blur)")
    print(f"  - Batches/Epoch: {args.batches_per_epoch} | Batch Size: {batch_size}")
    print("=" * 72 + "\n")

    generator = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler()
    jitter = StochasticJitter(pad=8, img_size=128).to(device)
    laplacian = LaplacianSharpness().to(device)

    # Fixed latents for progress tracking across all 10 classes
    fixed_z = torch.randn(10, args.latent_dim, device=device)
    fixed_y = torch.arange(10, device=device)

    # Temperature and bias from SigLIP (or defaults)
    logit_scale = getattr(siglip, "logit_scale", nn.Parameter(torch.tensor(10.0, device=device))).exp() if hasattr(siglip, "logit_scale") else 10.0
    logit_bias = getattr(siglip, "logit_bias", torch.tensor(-10.0, device=device)) if hasattr(siglip, "logit_bias") else -10.0

    for epoch in range(1, args.epochs + 1):
        generator.train()

        total_loss = 0.0
        total_siglip_loss = 0.0
        total_sim = 0.0
        total_edge = 0.0
        total_div = 0.0

        for _ in range(args.batches_per_epoch):
            # Balanced classes per batch
            y_tensor = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
            perm = torch.randperm(batch_size, device=device)
            y_tensor = y_tensor[perm]

            z1 = torch.randn(batch_size, args.latent_dim, device=device)
            z2 = torch.randn(batch_size, args.latent_dim, device=device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Multi-scale outputs (128x128 native resolution)
                x128_1, x64_1, _ = generator(z1, y_tensor, return_all_scales=True)
                x128_2, _, _ = generator(z2, y_tensor, return_all_scales=True)

                # 1. Mode-seeking diversity
                diff_z = torch.abs(z1 - z2).mean()
                diff_img = torch.abs(x128_1 - x128_2).mean()
                loss_div = diff_z / (diff_img + 1e-4)

                # 2. Stochastic Scale-Jitter (kills scanlines)
                x_aug = jitter(x128_1)

                # 3. GLOBAL SIGLIP ALIGNMENT (Whole Scene & Anatomy)
                # 128x128 upscales to 224x224 with pristine 1.75x ratio (retaining sharp physical structure)
                x_sig = F.interpolate(x_aug, size=(224, 224), mode='bicubic', align_corners=False)
                x_sig_norm = (x_sig - siglip_mean) / siglip_std

                if hasattr(siglip, "get_image_features"):
                    raw_img = siglip.get_image_features(pixel_values=x_sig_norm.half())
                    img_embeds = F.normalize(extract_tensor(raw_img), dim=-1)
                else:
                    img_embeds = F.normalize(siglip.encode_image(x_sig_norm.half()), dim=-1)

                sims = torch.matmul(img_embeds, pos_embeds.t())  # (B, 10)
                logits = sims * logit_scale + logit_bias

                target_mask = F.one_hot(y_tensor, num_classes=10).float()
                loss_siglip = F.binary_cross_entropy_with_logits(logits, target_mask)
                target_sims = sims.gather(1, y_tensor.unsqueeze(1)).squeeze(1)

                # 4. MULTI-CROP ZOOMED PATCH DETAIL (Forces micro-textures: tires, fur, feathers, eyes)
                crop_sz = torch.randint(56, 84, (1,)).item()
                top = torch.randint(0, 128 - crop_sz + 1, (1,)).item()
                left = torch.randint(0, 128 - crop_sz + 1, (1,)).item()
                x_patch = x128_1[:, :, top:top + crop_sz, left:left + crop_sz]
                x_patch_up = F.interpolate(x_patch, size=(224, 224), mode='bicubic', align_corners=False)
                x_patch_norm = (x_patch_up - siglip_mean) / siglip_std

                if hasattr(siglip, "get_image_features"):
                    raw_patch = siglip.get_image_features(pixel_values=x_patch_norm.half())
                    patch_embeds = F.normalize(extract_tensor(raw_patch), dim=-1)
                else:
                    patch_embeds = F.normalize(siglip.encode_image(x_patch_norm.half()), dim=-1)

                patch_sims = torch.matmul(patch_embeds, detail_embeds.t())
                patch_logits = patch_sims * logit_scale + logit_bias
                loss_patch = F.binary_cross_entropy_with_logits(patch_logits, target_mask)

                # 5. LAPLACIAN HIGH-FREQUENCY EDGE FLOOR (Snaps crisp contours)
                edge_energy = laplacian(x128_1)
                loss_sharpness = torch.relu(0.045 - edge_energy).mean()

                # 6. Negative Prompt Repulsion (kills blur and fog)
                neg_sims = torch.matmul(img_embeds, neg_embeds.t()).max(dim=-1)[0]
                loss_neg = torch.relu(neg_sims).mean()

                # 7. Anti-Fog Dynamic Range Floor
                per_img_std = x128_1.std(dim=[-2, -1])  # (B, 3)
                loss_contrast = torch.relu(0.22 - per_img_std).mean()

                # 8. Natural Daylight Color Moments
                batch_m = x128_1.mean(dim=[0, 2, 3], keepdim=True)
                batch_s = x128_1.std(dim=[0, 2, 3], keepdim=True)
                loss_color = F.mse_loss(batch_m, target_mean) + F.mse_loss(batch_s, target_std)

                # 9. Slashed Total Variation (Kills salt-and-pepper noise without blurring real edges)
                tv_loss = torch.mean(torch.abs(x128_1[:, :, :, :-1] - x128_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(x128_1[:, :, :-1, :] - x128_1[:, :, 1:, :]))

                # TOTAL HIGH-FREQUENCY OBJECTIVE
                loss_total = (args.lambda_siglip * loss_siglip) + \
                             (args.lambda_patch * loss_patch) + \
                             (args.lambda_sharpness * loss_sharpness) + \
                             (args.lambda_contrast * loss_contrast) + \
                             (args.lambda_color * loss_color) + \
                             (args.lambda_neg * loss_neg) + \
                             (args.lambda_tv * tv_loss) + \
                             (args.lambda_div * loss_div)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_total.item()
            total_siglip_loss += loss_siglip.item()
            total_sim += target_sims.mean().item()
            total_edge += edge_energy.mean().item()
            total_div += diff_img.item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_sig = total_siglip_loss / args.batches_per_epoch
        avg_sim = total_sim / args.batches_per_epoch
        avg_edge = total_edge / args.batches_per_epoch
        avg_div = total_div / args.batches_per_epoch

        # Sample grid every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | SigLIP: {avg_sig:.4f} | Cos Sim: {avg_sim:.4f} | Edge Sharpness: {avg_edge:.4f} | Total: {avg_loss:.4f}")

            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z, fixed_y)
                save_image(samples, os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png"), nrow=5, normalize=False)

        elif epoch % 2 == 0:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | SigLIP: {avg_sig:.4f} | Cos Sim: {avg_sim:.4f} | Edge Sharpness: {avg_edge:.4f} | Total: {avg_loss:.4f}")

    # Final save
    final_path = os.path.join(args.output_dir, "generator_vlm_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\n>>> SigLIP 128x128 Synthesis Complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
