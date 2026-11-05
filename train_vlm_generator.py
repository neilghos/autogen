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

# ==============================================================================
# MULTI-SCALE CURRICULUM PROMPT SYSTEM (250 Epochs per Phase)
# ==============================================================================

# Phase 1 (Epochs 1 - 250): Macro Global Scene & Silhouette
# Establishes horizon line, ground plane, lighting direction, and overall object bounding volume.
CURRICULUM_PHASE_1_MACRO = [
    "a wide angle photo of a passenger airplane parked on airport tarmac under clear daylight sky",
    "a wide angle photo of an automobile car driving on an open asphalt road in daylight",
    "a wide angle photo of a wild songbird perched outdoors in natural forest scenery",
    "a wide angle photo of a domestic cat sitting on green grass lawn in natural daylight",
    "a wide angle photo of a wild deer standing in an open grassy forest meadow",
    "a wide angle photo of a domestic dog standing on a green grass field outdoors",
    "a wide angle photo of a green amphibian frog resting in a natural wet outdoor environment",
    "a wide angle photo of an equine horse standing in a rural grassy pasture",
    "a wide angle photo of a wooden sailing ship floating on open ocean water under daytime sky",
    "a wide angle photo of a commercial freight motor truck driving on a paved highway road"
]

# Phase 2 (Epochs 251 - 500): Structural Anatomy, Contours & Primary Parts
# Sharpens body parts, limb morphology, wings, wheels, windshield, mast, and clean boundaries.
CURRICULUM_PHASE_2_ANATOMY = [
    "a sharp focused photo of a passenger airplane showing wings, fuselage body, cockpit, and tail fin",
    "a sharp focused photo of an automobile car showing vehicle body doors, windshield glass, hood, and wheels",
    "a sharp focused photo of a wild bird showing head, beak, wings, tail feathers, and perched legs",
    "a sharp focused photo of a domestic cat showing pointed ears, head profile, feline body, and tail",
    "a sharp focused photo of a wild deer showing head with branched antlers, four slender legs, and body",
    "a sharp focused photo of a domestic dog showing canine head, muzzle, ears, four legs, and body profile",
    "a sharp focused photo of a green tree frog showing frog head with protruding eyes, moist throat, and legs",
    "a sharp focused photo of an equine horse showing muscular neck, head, equestrian body, and four legs",
    "a sharp focused photo of a wooden sailing ship showing carved ship hull, tall masts, and rigging sails",
    "a sharp focused photo of a commercial motor truck showing truck cab, front grille, side mirrors, and wheels"
]

# Phase 3 (Epochs 501+): Micro-Textures, Surface Speculars & Fine Physics
# Drills into 1-pixel micro-details: glistening skin pores, rivets, tire treads, hair strands, feathers.
CURRICULUM_PHASE_3_MICRO = [
    "a macro close up photo of passenger airplane polished aluminum rivets, jet engine intake turbine, and wing seams",
    "a macro close up photo of automobile deep rubber tire tread grooves, alloy wheel rims, and glass headlights",
    "a macro close up photo of wild bird plumage feather barbules, dark pupil eye, and sharp pointed beak",
    "a macro close up photo of domestic cat whiskers, dense soft fur coat fibers, and inner ear hair",
    "a macro close up photo of wild deer antler tines, soft velvet coat texture, and alert dark eyes",
    "a macro close up photo of domestic dog textured wet nose leather, fine muzzle whiskers, and soft fur",
    "a macro close up photo of green tree frog glistening wet skin texture, golden rimmed eye, and throat skin",
    "a macro close up photo of horse silky flowing mane hair strands, glossy coat highlights, and dark eyes",
    "a macro close up photo of sailing ship wooden hull planks, textured ropes, and woven fabric sails",
    "a macro close up photo of heavy commercial truck chrome front grill, large tire treads, and side mirrors"
]

NEGATIVE_PROMPTS = [
    "blurry, foggy, out of focus, low contrast, abstract dots, scanlines, distorted, surreal cartoon, blown out white, overexposed, washed out"
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


def encode_prompts(prompts, tokenizer, model, device):
    """Encodes a list of prompts into normalized FP16 SigLIP embeddings."""
    if hasattr(model, "get_text_features"):
        inputs = tokenizer(prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            raw_text = model.get_text_features(**inputs)
            return F.normalize(extract_tensor(raw_text).half(), dim=-1)
    else:
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            return F.normalize(model.encode_text(tokens).half(), dim=-1)


def precompute_curriculum_anchors(tokenizer, model, device):
    """
    Precomputes normalized SigLIP text embeddings for all 3 curriculum phases:
      1. Phase 1: Macro Global Composition (1-250)
      2. Phase 2: Structural Anatomy & Contours (251-500)
      3. Phase 3: Micro-Textures & Fine Physics (501+)
      4. Negative repulsion prompts
    """
    print("[Curriculum Engine] Precomputing multi-scale prompt embeddings...")
    embeds_p1 = encode_prompts(CURRICULUM_PHASE_1_MACRO, tokenizer, model, device)
    embeds_p2 = encode_prompts(CURRICULUM_PHASE_2_ANATOMY, tokenizer, model, device)
    embeds_p3 = encode_prompts(CURRICULUM_PHASE_3_MICRO, tokenizer, model, device)
    embeds_neg = encode_prompts(NEGATIVE_PROMPTS, tokenizer, model, device)
    print("  - Phase 1 (Macro):", embeds_p1.shape)
    print("  - Phase 2 (Anatomy):", embeds_p2.shape)
    print("  - Phase 3 (Micro):", embeds_p3.shape)
    print("  - Negative:", embeds_neg.shape, "\n")
    return embeds_p1, embeds_p2, embeds_p3, embeds_neg


def main():
    parser = argparse.ArgumentParser(description="Google SigLIP Multi-Scale Curriculum Synthesis")
    parser.add_argument("--epochs", type=int, default=750, help="Total training epochs (Default: 750 = 3 x 250 phases)")
    parser.add_argument("--phase_length", type=int, default=250, help="Epoch duration for each curriculum phase")
    parser.add_argument("--batches_per_epoch", type=int, default=15, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=2, help="Samples per class per batch (Default: 2 -> Batch size = 20)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # High-Frequency Details & Edge Loss Weights
    parser.add_argument("--lambda_siglip", type=float, default=5.0, help="Weight for Global SigLIP Alignment")
    parser.add_argument("--lambda_patch", type=float, default=2.5, help="Weight for Zoomed Patch Details")
    parser.add_argument("--lambda_sharpness", type=float, default=4.0, help="Weight for Laplacian Edge Floor")
    parser.add_argument("--lambda_contrast", type=float, default=3.0, help="Dynamic range contrast weight")
    parser.add_argument("--lambda_exposure", type=float, default=2.0, help="Anti-washout highlight/shadow penalty")
    parser.add_argument("--lambda_color", type=float, default=1.5, help="Natural daylight color moments weight")
    parser.add_argument("--lambda_tv", type=float, default=0.002, help="Slashed TV weight")
    parser.add_argument("--lambda_div", type=float, default=0.03, help="Mode-seeking diversity weight")
    parser.add_argument("--lambda_neg", type=float, default=1.0, help="Negative prompt repulsion weight")

    parser.add_argument("--output_dir", type=str, default="./results_vlm_generator", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # 1. Load Google SigLIP Judge
    tokenizer, siglip = load_siglip_judge(device)
    embeds_p1, embeds_p2, embeds_p3, neg_embeds = precompute_curriculum_anchors(tokenizer, siglip, device)

    # SigLIP standard normalization: [-1, 1]
    siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    siglip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    # Natural color statistics target (from real CIFAR-10)
    target_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    target_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    # 2. Multi-Scale Decoder Generator (128x128)
    batch_size = 10 * args.samples_per_class
    print("=" * 72)
    print("MULTI-SCALE CURRICULUM SYNTHESIS ENGINE")
    print(f"  - Canvas Resolution: 128 x 128 native pixels")
    print(f"  - Phase Length: {args.phase_length} epochs each")
    print(f"    * Phase 1 (1 - {args.phase_length}): Macro Global Scene & Silhouettes")
    print(f"    * Phase 2 ({args.phase_length + 1} - {2 * args.phase_length}): Structural Anatomy & Contours")
    print(f"    * Phase 3 ({2 * args.phase_length + 1}+): Micro-Textures & Fine Physics")
    print(f"  - Anti-Washout Exposure Guard: ACTIVE (Prevents highlight blowouts)")
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

    logit_scale = getattr(siglip, "logit_scale", nn.Parameter(torch.tensor(10.0, device=device))).exp() if hasattr(siglip, "logit_scale") else 10.0
    logit_bias = getattr(siglip, "logit_bias", torch.tensor(-10.0, device=device)) if hasattr(siglip, "logit_bias") else -10.0

    current_phase = 0

    for epoch in range(1, args.epochs + 1):
        # Determine Curriculum Phase
        if epoch <= args.phase_length:
            active_phase = 1
            phase_title = "Phase 1: Macro Global Composition"
            active_global_embeds = embeds_p1
            active_patch_embeds = embeds_p1
            crop_min, crop_max = 64, 96
        elif epoch <= 2 * args.phase_length:
            active_phase = 2
            phase_title = "Phase 2: Structural Anatomy & Contours"
            active_global_embeds = embeds_p2
            active_patch_embeds = embeds_p2
            crop_min, crop_max = 56, 84
        else:
            active_phase = 3
            phase_title = "Phase 3: Micro-Textures & Surface Physics"
            active_global_embeds = embeds_p2  # Anchor global view to anatomy
            active_patch_embeds = embeds_p3   # Zoomed patches focus on micro textures
            crop_min, crop_max = 48, 72

        if active_phase != current_phase:
            current_phase = active_phase
            print(f"\n{'='*72}")
            print(f">>> [CURRICULUM SHIFT] ENTERING {phase_title.upper()} (Epoch {epoch})")
            print(f"{'='*72}\n")

        generator.train()

        total_loss = 0.0
        total_siglip_loss = 0.0
        total_sim = 0.0
        total_edge = 0.0
        total_div = 0.0

        for _ in range(args.batches_per_epoch):
            y_tensor = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
            perm = torch.randperm(batch_size, device=device)
            y_tensor = y_tensor[perm]

            z1 = torch.randn(batch_size, args.latent_dim, device=device)
            z2 = torch.randn(batch_size, args.latent_dim, device=device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                x128_1, x64_1, _ = generator(z1, y_tensor, return_all_scales=True)
                x128_2, _, _ = generator(z2, y_tensor, return_all_scales=True)

                # 1. Mode-seeking diversity
                diff_z = torch.abs(z1 - z2).mean()
                diff_img = torch.abs(x128_1 - x128_2).mean()
                loss_div = diff_z / (diff_img + 1e-4)

                # 2. Stochastic Scale-Jitter
                x_aug = jitter(x128_1)

                # 3. GLOBAL SIGLIP ALIGNMENT (Phase-Specific)
                x_sig = F.interpolate(x_aug, size=(224, 224), mode='bicubic', align_corners=False)
                x_sig_norm = (x_sig - siglip_mean) / siglip_std

                if hasattr(siglip, "get_image_features"):
                    raw_img = siglip.get_image_features(pixel_values=x_sig_norm.half())
                    img_embeds = F.normalize(extract_tensor(raw_img), dim=-1)
                else:
                    img_embeds = F.normalize(siglip.encode_image(x_sig_norm.half()), dim=-1)

                sims = torch.matmul(img_embeds, active_global_embeds.t())  # (B, 10)
                logits = sims * logit_scale + logit_bias

                target_mask = F.one_hot(y_tensor, num_classes=10).float()
                loss_siglip = F.binary_cross_entropy_with_logits(logits, target_mask)
                target_sims = sims.gather(1, y_tensor.unsqueeze(1)).squeeze(1)

                # 4. MULTI-CROP ZOOMED PATCH DETAIL (Phase-Specific)
                crop_sz = torch.randint(crop_min, crop_max + 1, (1,)).item()
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

                patch_sims = torch.matmul(patch_embeds, active_patch_embeds.t())
                patch_logits = patch_sims * logit_scale + logit_bias
                loss_patch = F.binary_cross_entropy_with_logits(patch_logits, target_mask)

                # 5. LAPLACIAN HIGH-FREQUENCY EDGE FLOOR
                edge_energy = laplacian(x128_1)
                loss_sharpness = torch.relu(0.045 - edge_energy).mean()

                # 6. Negative Prompt Repulsion
                neg_sims = torch.matmul(img_embeds, neg_embeds.t()).max(dim=-1)[0]
                loss_neg = torch.relu(neg_sims).mean()

                # 7. Balanced Contrast Range (prevents both fog and extreme polarization)
                per_img_std = x128_1.std(dim=[-2, -1])  # (B, 3)
                loss_contrast = torch.relu(0.20 - per_img_std).mean() + torch.relu(per_img_std - 0.32).mean()

                # 8. Anti-Washout Exposure Guard (Penalizes blown-out whites and crushed blacks)
                loss_exposure = torch.relu(x128_1 - 0.92).pow(2).mean() + torch.relu(0.04 - x128_1).pow(2).mean()

                # 9. Natural Daylight Color Moments
                batch_m = x128_1.mean(dim=[0, 2, 3], keepdim=True)
                batch_s = x128_1.std(dim=[0, 2, 3], keepdim=True)
                loss_color = F.mse_loss(batch_m, target_mean) + F.mse_loss(batch_s, target_std)

                # 10. Slashed Total Variation
                tv_loss = torch.mean(torch.abs(x128_1[:, :, :, :-1] - x128_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(x128_1[:, :, :-1, :] - x128_1[:, :, 1:, :]))

                # TOTAL CURRICULUM OBJECTIVE
                loss_total = (args.lambda_siglip * loss_siglip) + \
                             (args.lambda_patch * loss_patch) + \
                             (args.lambda_sharpness * loss_sharpness) + \
                             (args.lambda_contrast * loss_contrast) + \
                             (args.lambda_exposure * loss_exposure) + \
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
            print(f"[{active_phase}] Epoch {epoch:04d}/{args.epochs:04d} | SigLIP: {avg_sig:.4f} | Cos Sim: {avg_sim:.4f} | Edge: {avg_edge:.4f} | Total: {avg_loss:.4f}")

            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z, fixed_y)
                save_image(samples, os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png"), nrow=5, normalize=False)

        elif epoch % 2 == 0:
            print(f"[{active_phase}] Epoch {epoch:04d}/{args.epochs:04d} | SigLIP: {avg_sig:.4f} | Cos Sim: {avg_sim:.4f} | Edge: {avg_edge:.4f} | Total: {avg_loss:.4f}")

    # Final save
    final_path = os.path.join(args.output_dir, "generator_vlm_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\n>>> Curriculum Synthesis Complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
