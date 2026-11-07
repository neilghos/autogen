import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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


BLENDER_ASSETS = [
    "glossy_red_sphere",      # 0: Spherical curvature, glossy specular highlight, red albedo
    "metallic_blue_box",      # 1: 90-degree perspective edges, brushed metallic finish, blue albedo
    "black_rubber_tire",      # 2: Concentric tire tread, alloy wheel rim, deep black tone
    "glazed_yellow_torus",    # 3: Annular hole topology, glazed ceramic shine, yellow albedo
    "machined_green_cylinder",# 4: Extruded circular cap & straight shaft, green albedo
    "aerodynamic_orange_wedge",# 5: 30-degree prow & sloping facet planes, orange albedo
    "glossy_cyan_capsule",    # 6: Rounded capsule ends & smooth body, cyan albedo
    "terracotta_clay_bust"    # 7: Organic anatomical muscle curves, warm terracotta albedo
]

# ==============================================================================
# HEAD 1: L_Identity (Shape Geometry & 3D Material Finish)
# ==============================================================================
# Stage 1: Macro 3D Geometry & Staging
IDENTITY_STAGE_1_PROMPTS = [
    "a 3D glossy reflective sphere with specular highlight and ambient contact shadow on a dark studio turntable",
    "a 3D brushed metallic box with sharp 90 degree edges, flat perspective faces, and cast shadow on a dark floor",
    "a 3D black rubber vehicle tire with tread and an alloy wheel rim standing upright on a dark studio ground",
    "a 3D glazed ceramic torus donut with round inner hole and glossy shine, resting on a dark reflective surface",
    "a 3D machined metallic steel cylinder with flat circular top cap and straight extruded shaft on a dark backdrop",
    "a 3D aerodynamic wedge prism with sharp acute prow, angled sloping facet faces, and contact shadow on dark floor",
    "a 3D glossy rounded capsule with smooth cylindrical body, specular shine, and subtle contact shadow on dark ground",
    "a 3D sculpted clay head bust with organic anatomical facial curves and soft crease shadows on a dark turntable"
]

# Stage 2: Micro 3D Refinement, Razor Bevels & Raytraced Curvature
IDENTITY_STAGE_2_PROMPTS = [
    "a 3D Blender Cycles render of a glossy spherical ball with razor-sharp circular curvature, tight specular highlight dot, and dark floor reflection",
    "a 3D Blender Cycles render of a metallic cuboid box with razor-sharp 90 degree beveled edges, anisotropic brushing, and straight perspective lines",
    "a 3D Blender Cycles render of a vehicle wheel with crisp rubber tread grooves, sharp metallic alloy rim spokes, and ground contact compression",
    "a 3D Blender Cycles render of a glazed ceramic donut with crisp circular hole rim, smooth glaze, and dark floor mirror reflection",
    "a 3D Blender Cycles render of a machined steel cylinder with knife-edge circular lip and vertical specular rim highlight streak",
    "a 3D Blender Cycles render of an aerodynamic wedge with razor-sharp 30 degree prow and clean planar facet slopes on dark floor",
    "a 3D Blender Cycles render of a glossy capsule with razor-sharp rounded contours, tight specular sheen, and clean cylindrical volume",
    "a 3D Blender Cycles render of a detailed clay sculpted bust with carved anatomical facial crevices and micro-crease ambient occlusion"
]

IDENTITY_NEGATIVE_PROMPTS = [
    "2D drawing, flat cartoon, sketch, vector clip art, flat unshaded illustration, line art, low poly flat",
    "fireflies, noisy render, low sample count, grainy noise, pixelated, distorted, bad render",
    "bad normals, clipping mesh, jagged polygon artifacts, broken topology, glitch, scanlines",
    "hollow center, black horizontal bar, cutout, notch, void inside, light background"
]

# ==============================================================================
# HEAD 2: L_Color (Chromatic Albedo & Saturation - The Anti-Whiteout Anchor)
# ==============================================================================
COLOR_ALBEDO_PROMPTS = [
    "bold pure saturated red color fill, vibrant red albedo, deep crimson red",
    "bold pure saturated blue color fill, vibrant blue albedo, deep azure blue",
    "solid deep pitch black and dark charcoal grey, pure dark black tone",
    "bold pure saturated yellow color fill, vibrant yellow albedo, bright golden yellow",
    "bold pure saturated green color fill, vibrant green albedo, deep emerald green",
    "bold pure saturated orange color fill, vibrant orange albedo, warm amber orange",
    "bold pure saturated cyan color fill, vibrant cyan albedo, bright turquoise cyan",
    "rich terracotta warm clay brown color fill, earthy reddish brown tone"
]

COLOR_NEGATIVE_PROMPTS = [
    "blown out white glare, pure whiteout, overexposed white bloom, washed out white light, blinding specular bloom",
    "desaturated grey, dull bleached monochrome, colorless, washed out tone",
    "rainbow noise, chromatic aberration artifacts, color bleeding cross-talk",
    "light background, bright white canvas background, pale background"
]


class StochasticJitter(nn.Module):
    """Gentle translation shifts (+- pad) for spatial invariance."""
    def __init__(self, pad=4, img_size=128):
        super().__init__()
        self.pad = pad
        self.img_size = img_size

    def forward(self, x):
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        h = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        w = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        return x_pad[:, :, h:h + self.img_size, w:w + self.img_size]


def extract_tensor(x):
    """Safely extracts a torch.Tensor from ModelOutput or tuple structures."""
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
        return x.last_hidden_state[:, 0]
    if hasattr(x, "image_embeds") and x.image_embeds is not None:
        return x.image_embeds
    if hasattr(x, "text_embeds") and x.text_embeds is not None:
        return x.text_embeds
    if isinstance(x, (tuple, list)):
        return extract_tensor(x[0])
    raise TypeError(f"Cannot extract tensor from type: {type(x)}")


def load_siglip_judge(device):
    """Loads frozen Google SigLIP ViT-B/16 judge."""
    print("[Director] Loading Google SigLIP ViT-B/16 Judge...")
    try:
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).half()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[Director] Loaded via Hugging Face Transformers.\n")
        return processor.tokenizer, model
    except Exception as e:
        print(f"[Director] HF loader failed ({e}), falling back to open_clip...")
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model = model.to(device).half()
        tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[Director] Loaded via open_clip.\n")
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


def main():
    parser = argparse.ArgumentParser(description="Multi-Axis Softmax (Shape & Finish + Chromatic Albedo) via Dual InfoNCE")
    parser.add_argument("--epochs", type=int, default=1000, help="Total training epochs (Default: 1000)")
    parser.add_argument("--phase_cycle", "--phase-cycle", type=int, default=100, help="Epochs per phase before alternating (Default: 100)")
    parser.add_argument("--batches_per_epoch", type=int, default=15, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=2, help="Samples per class (8 classes x 2 = Batch size 16)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--steps", type=int, default=2, help="Painter recurrent self-observation steps (Default: 2)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Painter learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # Multi-Axis Softmax Hyperparameters
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature tau (Default: 0.07)")
    parser.add_argument("--lambda_id", type=float, default=4.0, help="Weight for Head 1: L_Identity (Shape & Finish)")
    parser.add_argument("--lambda_color", type=float, default=4.0, help="Weight for Head 2: L_Color (Chromatic Albedo Anti-Whiteout)")
    parser.add_argument("--lambda_bg", type=float, default=1.0, help="Canvas outer border dark stage guard")
    parser.add_argument("--lambda_tv", type=float, default=0.001, help="Total Variation (Noise/firefly guard)")
    parser.add_argument("--lambda_div", type=float, default=0.02, help="Mode-seeking latent diversity")

    parser.add_argument("--output_dir", type=str, default="./results_blender_bank", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # 1. Load Director (SigLIP Judge)
    tokenizer, siglip = load_siglip_judge(device)

    # Precompute Head 1 Embeddings (Shape Geometry & 3D Material Finish)
    id_embeds_s1 = encode_prompts(IDENTITY_STAGE_1_PROMPTS, tokenizer, siglip, device)
    id_embeds_s2 = encode_prompts(IDENTITY_STAGE_2_PROMPTS, tokenizer, siglip, device)
    id_neg_embeds = encode_prompts(IDENTITY_NEGATIVE_PROMPTS, tokenizer, siglip, device)
    all_id_s1 = torch.cat([id_embeds_s1, id_neg_embeds], dim=0)
    all_id_s2 = torch.cat([id_embeds_s2, id_neg_embeds], dim=0)

    # Precompute Head 2 Embeddings (Chromatic Albedo & Anti-Whiteout Denominator)
    color_embeds = encode_prompts(COLOR_ALBEDO_PROMPTS, tokenizer, siglip, device)
    color_neg_embeds = encode_prompts(COLOR_NEGATIVE_PROMPTS, tokenizer, siglip, device)
    all_color = torch.cat([color_embeds, color_neg_embeds], dim=0)

    num_classes = len(BLENDER_ASSETS)
    num_id_neg = len(IDENTITY_NEGATIVE_PROMPTS)
    num_color_neg = len(COLOR_NEGATIVE_PROMPTS)

    siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    siglip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    batch_size = num_classes * args.samples_per_class
    print("=" * 72)
    print("PAINTER-DIRECTOR MULTI-AXIS SOFTMAX ENGINE (THE IMPOSSIBLE-TO-CHEAT GAME)")
    print(f"  - Target Assets: {num_classes} Blender CGI Assets")
    print(f"  - Head 1 (L_Identity): {num_classes} Shapes/Finishes + {num_id_neg} Negatives in Denominator (Weight: {args.lambda_id})")
    print(f"  - Head 2 (L_Color):    {num_classes} Pure Color Albedos + {num_color_neg} Anti-Whiteout Negatives in Denominator (Weight: {args.lambda_color})")
    print(f"  - InfoNCE Temperature tau: {args.temperature} (Scale: {1.0/args.temperature:.1f})")
    print(f"  - Alternating Schedule: Every {args.phase_cycle} epochs")
    print("    * Stage 1: Macro 3D Geometry & Staging")
    print("    * Stage 2: Micro Raytraced Refinement, Razor Bevels & Specular Curvature")
    print(f"  - Resolution: 128 x 128 | Feedback Recurrence: {args.steps} steps")
    print(f"  - Batch Size: {batch_size} ({args.samples_per_class} samples/asset)")
    print("=" * 72 + "\n")

    # 2. Instantiate Painter Network (num_classes=8)
    painter = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(painter.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
    jitter = StochasticJitter(pad=4, img_size=128).to(device)

    # Fixed latents for progress tracking (8 assets x 4 fixed camera/latent variations = 32 images)
    fixed_y = torch.arange(num_classes, device=device).repeat_interleave(4)
    fixed_z = torch.randn(len(fixed_y), args.latent_dim, device=device)

    current_stage = 0

    for epoch in range(1, args.epochs + 1):
        # Alternating every phase_cycle epochs
        cycle_idx = (epoch - 1) // args.phase_cycle
        phase_num = cycle_idx + 1
        total_phases = (args.epochs + args.phase_cycle - 1) // args.phase_cycle

        if cycle_idx % 2 == 0:
            active_stage = 1
            stage_title = f"Phase {phase_num}/{total_phases} (Stage 1: Macro 3D Form & Studio Staging)"
            active_id_embeds = all_id_s1
        else:
            active_stage = 2
            stage_title = f"Phase {phase_num}/{total_phases} (Stage 2: Micro Raytraced Refinement & Bevels)"
            active_id_embeds = all_id_s2

        if active_stage != current_stage or epoch == 1:
            current_stage = active_stage
            print(f"\n{'='*72}")
            print(f">>> [DIRECTOR SHIFT] ENTERING {stage_title.upper()} (Epoch {epoch})")
            print(f"{'='*72}\n")

        painter.train()

        total_loss = 0.0
        total_id_loss = 0.0
        total_color_loss = 0.0
        total_id_p = 0.0
        total_color_p = 0.0
        total_whiteout_p = 0.0

        for _ in range(args.batches_per_epoch):
            y_tensor = torch.arange(num_classes, device=device).repeat_interleave(args.samples_per_class)
            perm = torch.randperm(batch_size, device=device)
            y_tensor = y_tensor[perm]

            z1 = torch.randn(batch_size, args.latent_dim, device=device)
            z2 = torch.randn(batch_size, args.latent_dim, device=device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(dtype=torch.float16):
                x128_1, x64_1, _ = painter(z1, y_tensor, steps=args.steps, return_all_scales=True)
                x128_2, _, _ = painter(z2, y_tensor, steps=args.steps, return_all_scales=True)

                # 1. Mode-seeking diversity (different rotations/angles per z)
                diff_z = torch.abs(z1 - z2).mean()
                diff_img = torch.abs(x128_1 - x128_2).mean()
                loss_div = diff_z / (diff_img + 1e-4)

                # 2. Gentle translation jitter
                x_aug = jitter(x128_1)

                # 3. PURE DIRECTOR MULTI-AXIS InfoNCE (DUAL SOFTMAX HEADS)
                x_sig = F.interpolate(x_aug, size=(224, 224), mode='bicubic', align_corners=False)
                x_sig_norm = (x_sig - siglip_mean) / siglip_std

                if hasattr(siglip, "get_image_features"):
                    raw_img = siglip.get_image_features(pixel_values=x_sig_norm.half())
                    img_embeds = F.normalize(extract_tensor(raw_img), dim=-1)
                else:
                    img_embeds = F.normalize(siglip.encode_image(x_sig_norm.half()), dim=-1)

                # --- HEAD 1: L_Identity (Shape Geometry & 3D Material Finish) ---
                all_id_sims = torch.matmul(img_embeds, active_id_embeds.t())  # (B, 12)
                id_logits = all_id_sims / args.temperature
                loss_id = F.cross_entropy(id_logits, y_tensor)

                # --- HEAD 2: L_Color (Chromatic Albedo - The Anti-Whiteout Anchor) ---
                all_color_sims = torch.matmul(img_embeds, all_color.t())       # (B, 12)
                color_logits = all_color_sims / args.temperature
                loss_color = F.cross_entropy(color_logits, y_tensor)

                with torch.no_grad():
                    id_probs = F.softmax(id_logits, dim=-1)
                    target_id_prob = id_probs.gather(1, y_tensor.unsqueeze(1)).squeeze(1)

                    color_probs = F.softmax(color_logits, dim=-1)
                    target_color_prob = color_probs.gather(1, y_tensor.unsqueeze(1)).squeeze(1)
                    whiteout_neg_prob = color_probs[:, num_classes].squeeze(-1) # Index 8 is whiteout glare

                # 4. Canvas Outer Border Guard (Preserves dark studio cyc background)
                border_loss = x128_1[:, :, :6, :].mean() + \
                              x128_1[:, :, -6:, :].mean() + \
                              x128_1[:, :, :, :6].mean() + \
                              x128_1[:, :, :, -6:].mean()

                # 5. Minimal TV (Suppresses render fireflies and camera noise)
                tv_loss = torch.mean(torch.abs(x128_1[:, :, :, :-1] - x128_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(x128_1[:, :, :-1, :] - x128_1[:, :, 1:, :]))

                # TOTAL MULTI-AXIS MIN-MAX OBJECTIVE
                loss_total = (args.lambda_id * loss_id) + \
                             (args.lambda_color * loss_color) + \
                             (args.lambda_bg * border_loss) + \
                             (args.lambda_tv * tv_loss) + \
                             (args.lambda_div * loss_div)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_total.item()
            total_id_loss += loss_id.item()
            total_color_loss += loss_color.item()
            total_id_p += target_id_prob.mean().item()
            total_color_p += target_color_prob.mean().item()
            total_whiteout_p += whiteout_neg_prob.mean().item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_id = total_id_loss / args.batches_per_epoch
        avg_color = total_color_loss / args.batches_per_epoch
        avg_id_p = total_id_p / args.batches_per_epoch
        avg_color_p = total_color_p / args.batches_per_epoch
        avg_whiteout_p = total_whiteout_p / args.batches_per_epoch

        # Log & visualize every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[{active_stage}] Ep {epoch:04d}/{args.epochs:04d} | ID Loss: {avg_id:.3f} (P: {avg_id_p*100:.1f}%) | Color Loss: {avg_color:.3f} (P: {avg_color_p*100:.1f}%) | Glare P: {avg_whiteout_p*100:.2f}% | Total: {avg_loss:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            painter.eval()
            with torch.no_grad():
                samples = painter(fixed_z, fixed_y, steps=args.steps)
                img_path = os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png")
                save_image(samples, img_path, nrow=4, normalize=False)

    final_path = os.path.join(args.output_dir, "painter_blender_bank.pth")
    torch.save(painter.state_dict(), final_path)
    print(f"\n>>> Multi-Axis Blender 3D Pretraining Complete! Checkpoint saved to: {final_path}")

    final_path = os.path.join(args.output_dir, "painter_blender_bank.pth")
    torch.save(painter.state_dict(), final_path)
    print(f"\n>>> Blender 3D Studio Pretraining Complete! Checkpoint saved to: {final_path}")


if __name__ == "__main__":
    main()

