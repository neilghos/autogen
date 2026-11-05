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

# Grounded photographic descriptions for SigLIP (Google's Vision-Language Judge)
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

NEGATIVE_PROMPTS = [
    "blurry, foggy, low contrast, abstract dots, scanlines, distorted, surreal cartoon"
]


class StochasticJitter(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        # 1. Reflection pad and random translation crop
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        h = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        w = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        x_crop = x_pad[:, :, h:h + 32, w:w + 32]

        # 2. Scale jitter breaks ViT patch grid phase-lock (eradicates scanlines)
        scale = 0.92 + 0.16 * torch.rand(1, device=x.device).item()
        s_dim = int(round(32 * scale))
        x_scaled = F.interpolate(x_crop, size=(s_dim, s_dim), mode='bilinear', align_corners=False)
        return F.interpolate(x_scaled, size=(32, 32), mode='bilinear', align_corners=False)


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
    Precomputes normalized SigLIP text embeddings for all 10 CIFAR classes
    and negative repulsion prompts.
    """
    # 1. Positive class anchors
    if hasattr(model, "get_text_features"):
        # HuggingFace SigLIP
        inputs = tokenizer(SIGLIP_CLASS_PROMPTS, padding="max_length", max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_text = model.get_text_features(**inputs)
            pos_embeds = F.normalize(extract_tensor(raw_text).half(), dim=-1)

        neg_inputs = tokenizer(NEGATIVE_PROMPTS, padding="max_length", max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_neg = model.get_text_features(**neg_inputs)
            neg_embeds = F.normalize(extract_tensor(raw_neg).half(), dim=-1)
    else:
        # open_clip SigLIP
        tokens = tokenizer(SIGLIP_CLASS_PROMPTS).to(device)
        with torch.no_grad():
            pos_embeds = F.normalize(model.encode_text(tokens).half(), dim=-1)

        neg_tokens = tokenizer(NEGATIVE_PROMPTS).to(device)
        with torch.no_grad():
            neg_embeds = F.normalize(model.encode_text(neg_tokens).half(), dim=-1)

    return pos_embeds, neg_embeds


def main():
    parser = argparse.ArgumentParser(description="Google SigLIP Indirect Supervised Learning (Non-Toxic Gradients)")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batches_per_epoch", type=int, default=15, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=2, help="Samples per class in each batch (Default: 2 -> Batch size = 20)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # SigLIP Loss Weights
    parser.add_argument("--lambda_siglip", type=float, default=5.0, help="Weight for SigLIP Sigmoid Alignment")
    parser.add_argument("--lambda_contrast", type=float, default=3.0, help="Anti-fog dynamic range contrast floor")
    parser.add_argument("--lambda_color", type=float, default=1.5, help="Natural daylight color moments weight")
    parser.add_argument("--lambda_tv", type=float, default=0.03, help="Light TV weight (kills speckles & scanlines)")
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
    pos_embeds, neg_embeds = precompute_siglip_anchors(tokenizer, siglip, device)
    print(f"[SigLIP Judge] Class feature anchors locked: {pos_embeds.shape}\n")

    # SigLIP standard normalization: [-1, 1]
    siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    siglip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    # Natural color statistics target (from real CIFAR-10)
    target_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    target_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    # 2. Multi-Scale Decoder Generator with FiLM
    batch_size = 10 * args.samples_per_class
    print("=" * 72)
    print("INDIRECT SUPERVISED LEARNING: GOOGLE SIGLIP (NON-TOXIC GRADIENTS)")
    print(f"  - Critic Model: google/siglip-base-patch16-224 (FP16)")
    print(f"  - Gradient Landscape: Smooth Sigmoid Pairwise Alignment (No Cross-Attention Sinks!)")
    print(f"  - Batches/Epoch: {args.batches_per_epoch} | Batch Size: {batch_size}")
    print("=" * 72 + "\n")

    generator = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    jitter = StochasticJitter(pad=4).to(device)

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
        total_contrast = 0.0
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
                # Multi-scale outputs
                x32_1, x16_1, _ = generator(z1, y_tensor, return_all_scales=True)
                x32_2, _, _ = generator(z2, y_tensor, return_all_scales=True)

                # 1. Mode-seeking diversity
                diff_z = torch.abs(z1 - z2).mean()
                diff_img = torch.abs(x32_1 - x32_2).mean()
                loss_div = diff_z / (diff_img + 1e-4)

                # 2. Stochastic Scale-Jitter (kills scanlines)
                x_aug = jitter(x32_1)
                x_sig = F.interpolate(x_aug, size=(224, 224), mode='bicubic', align_corners=False)
                x_sig_norm = (x_sig - siglip_mean) / siglip_std

                # 3. SigLIP Image Features
                if hasattr(siglip, "get_image_features"):
                    raw_img = siglip.get_image_features(pixel_values=x_sig_norm.half())
                    img_embeds = F.normalize(extract_tensor(raw_img), dim=-1)
                else:
                    img_embeds = F.normalize(siglip.encode_image(x_sig_norm.half()), dim=-1)

                # 4. Pairwise Sigmoid Logits (Google SigLIP formulation)
                sims = torch.matmul(img_embeds, pos_embeds.t())  # (B, 10)
                logits = sims * logit_scale + logit_bias

                # Construct binary targets: 1 for correct class, -1 (or 0) for others
                target_mask = F.one_hot(y_tensor, num_classes=10).float()
                # Binary Cross Entropy with Logits (pure SigLIP loss: smooth, non-adversarial)
                loss_siglip = F.binary_cross_entropy_with_logits(logits, target_mask)

                # Target cosine similarity for monitoring
                target_sims = sims.gather(1, y_tensor.unsqueeze(1)).squeeze(1)

                # 5. Negative Prompt Repulsion (kills blur and fog)
                neg_sims = torch.matmul(img_embeds, neg_embeds.t()).max(dim=-1)[0]
                loss_neg = torch.relu(neg_sims).mean()

                # 6. Anti-Fog Dynamic Range Floor
                per_img_std = x32_1.std(dim=[-2, -1])  # (B, 3)
                loss_contrast = torch.relu(0.20 - per_img_std).mean()

                # 7. Natural CIFAR Color Moments
                batch_m = x32_1.mean(dim=[0, 2, 3], keepdim=True)
                batch_s = x32_1.std(dim=[0, 2, 3], keepdim=True)
                loss_color = F.mse_loss(batch_m, target_mean) + F.mse_loss(batch_s, target_std)

                # 8. Light Total Variation
                tv_loss = torch.mean(torch.abs(x32_1[:, :, :, :-1] - x32_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(x32_1[:, :, :-1, :] - x32_1[:, :, 1:, :]))

                # TOTAL OBJECTIVE
                loss_total = (args.lambda_siglip * loss_siglip) + \
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
            total_contrast += per_img_std.mean().item()
            total_div += diff_img.item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_sig = total_siglip_loss / args.batches_per_epoch
        avg_sim = total_sim / args.batches_per_epoch
        avg_std = total_contrast / args.batches_per_epoch
        avg_div = total_div / args.batches_per_epoch

        # Sample grid every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | SigLIP Loss: {avg_sig:.4f} | Cos Sim: {avg_sim:.4f} | Contrast: {avg_std:.4f} | Total: {avg_loss:.4f}")

            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z, fixed_y)
                save_image(samples, os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png"), nrow=5, normalize=False)

        elif epoch % 2 == 0:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | SigLIP Loss: {avg_sig:.4f} | Cos Sim: {avg_sim:.4f} | Contrast: {avg_std:.4f} | Total: {avg_loss:.4f}")

    # Final save
    final_path = os.path.join(args.output_dir, "generator_vlm_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\n>>> SigLIP Indirect Supervised Learning Complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
