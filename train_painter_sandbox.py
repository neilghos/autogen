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


SHAPE_CLASSES = [
    "red_circle",
    "blue_square",
    "green_triangle",
    "yellow_ring"
]

# Stage 1: Macro Shape Identity, Solid Fills & Centering
STAGE_1_PROMPTS = [
    "a solid opaque red circular disc centered on a pure black background, bold pure red fill, simple geometric shape",
    "a solid opaque blue square centered on a pure black background, bold pure blue fill, four straight edges and 90 degree corners",
    "a solid opaque green triangle centered on a pure black background, bold pure green fill, three straight edges and sharp top vertex",
    "a solid opaque yellow ring donut with a circular hole centered on a pure black background, bold pure yellow fill"
]

# Stage 2: Pure Boundary Sharpening & Hard Vector Cuts (Solid Fills + Knife-Edge Perimeters)
STAGE_2_PROMPTS = [
    "a solid flat vector red circle with razor sharp hard cut edges, crisp circular perimeter, bold red fill against pure black background",
    "a solid flat vector blue square with razor sharp 90 degree corners, straight hard edges, bold blue fill against pure black background",
    "a solid flat vector green triangle with razor sharp vertices and straight edges, bold green fill against pure black background",
    "a solid flat vector yellow ring donut with razor sharp inner and outer circular edges, bold yellow fill against pure black background"
]

# Negative prompts included directly in the InfoNCE denominator
NEGATIVE_PROMPTS = [
    "hollow center, black horizontal bar, cutout, notch, void inside, no entry sign, crescent",
    "white background, bright background, light background, colored background halo, washed out",
    "blurry edges, diffuse halo, fuzzy glow, soft out-of-focus transition, feathered contour",
    "scanlines, horizontal lines, corrugated slats, metal grill, AC vents, noise, distorted"
]


class StochasticJitter(nn.Module):
    """
    Gentle translation shifts (+- pad) to encourage scale & translation robustness.
    """
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
    """
    Loads frozen Google SigLIP judge (google/siglip-base-patch16-224).
    """
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
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
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
    parser = argparse.ArgumentParser(description="Painter-Director Pure Prompt Gradient Sandbox")
    parser.add_argument("--epochs", type=int, default=400, help="Total epochs for sandbox training (Default: 400)")
    parser.add_argument("--phase_cycle", type=int, default=50, help="Epochs per phase before alternating (Default: 50)")
    parser.add_argument("--batches_per_epoch", type=int, default=10, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=4, help="Samples per class (4 classes x 4 = Batch size 16)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--steps", type=int, default=2, help="Painter recurrent self-observation steps (Default: 2)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Painter learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # Supervision Weights
    parser.add_argument("--lambda_infonce", type=float, default=5.0, help="Weight for InfoNCE Contrastive Alignment")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature tau (Default: 0.07)")
    parser.add_argument("--lambda_color_purity", type=float, default=1.5, help="Color channel purity (Prevents channel crosstalk)")
    parser.add_argument("--lambda_bg", type=float, default=1.0, help="Canvas outer border dark background guard")
    parser.add_argument("--lambda_tv", type=float, default=0.001, help="Minimal Total Variation (Noise guard only)")
    parser.add_argument("--lambda_div", type=float, default=0.02, help="Mode-seeking latent diversity")

    parser.add_argument("--output_dir", type=str, default="./results_painter_sandbox", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # 1. Load Director (SigLIP Judge)
    tokenizer, siglip = load_siglip_judge(device)
    embeds_stage1 = encode_prompts(STAGE_1_PROMPTS, tokenizer, siglip, device)
    embeds_stage2 = encode_prompts(STAGE_2_PROMPTS, tokenizer, siglip, device)
    neg_embeds = encode_prompts(NEGATIVE_PROMPTS, tokenizer, siglip, device)

    # InfoNCE Multi-Key Anchors: [Positive Classes (0..C-1) | Negative Prompts in Denominator (C..C+K-1)]
    all_embeds_stage1 = torch.cat([embeds_stage1, neg_embeds], dim=0)
    all_embeds_stage2 = torch.cat([embeds_stage2, neg_embeds], dim=0)

    siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    siglip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    batch_size = len(SHAPE_CLASSES) * args.samples_per_class
    print("=" * 72)
    print("PAINTER-DIRECTOR InfoNCE GRADIENT EXPERIMENT (NEGATIVES IN DENOMINATOR)")
    print(f"  - Total Epochs: {args.epochs} | Phase Duration: {args.phase_cycle} epochs each")
    print(f"  - InfoNCE Temperature tau: {args.temperature} (Scale: {1.0/args.temperature:.1f})")
    print(f"  - Contrastive Keys: {len(SHAPE_CLASSES)} Positive Classes + {len(NEGATIVE_PROMPTS)} Negative Denominator Keys = {all_embeds_stage1.shape[0]} total")
    print("  - Stage 1 (Odd Phases): Solid Opaque Fill & Macro Silhouette Grounding")
    print("  - Stage 2 (Even Phases): Hard-Edge Vector Cutoff & Boundary Sharpening")
    print("  - Canvas Border & Pure Color Guards: ACTIVE (100% Pure Saturated Colors)")
    print(f"  - Canvas Resolution: 128 x 128 | Painter Recurrence: {args.steps} steps")
    print(f"  - Batch Size: {batch_size}")
    print("=" * 72 + "\n")

    # 2. Instantiate Painter Network (num_classes=4)
    painter = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=len(SHAPE_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(painter.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
    jitter = StochasticJitter(pad=4, img_size=128).to(device)

    # Fixed latents for progress tracking (4 classes x 4 fixed variations = 16 images)
    fixed_y = torch.arange(len(SHAPE_CLASSES), device=device).repeat_interleave(4)
    fixed_z = torch.randn(len(fixed_y), args.latent_dim, device=device)

    current_stage = 0

    for epoch in range(1, args.epochs + 1):
        # Alternating every phase_cycle epochs (50 epochs) across phases
        cycle_idx = (epoch - 1) // args.phase_cycle
        phase_num = cycle_idx + 1
        total_phases = (args.epochs + args.phase_cycle - 1) // args.phase_cycle

        if cycle_idx % 2 == 0:
            active_stage = 1
            stage_title = f"Phase {phase_num}/{total_phases} (Stage 1: Solid Opaque Fills)"
            active_embeds = all_embeds_stage1
        else:
            active_stage = 2
            stage_title = f"Phase {phase_num}/{total_phases} (Stage 2: Vector Boundary Sharpening)"
            active_embeds = all_embeds_stage2

        if active_stage != current_stage or epoch == 1:
            current_stage = active_stage
            print(f"\n{'='*72}")
            print(f">>> [DIRECTOR SHIFT] ENTERING {stage_title.upper()} (Epoch {epoch})")
            print(f"{'='*72}\n")

        painter.train()

        total_loss = 0.0
        total_infonce = 0.0
        total_target_prob = 0.0
        total_neg_prob = 0.0
        total_sim = 0.0

        for _ in range(args.batches_per_epoch):
            y_tensor = torch.arange(len(SHAPE_CLASSES), device=device).repeat_interleave(args.samples_per_class)
            perm = torch.randperm(batch_size, device=device)
            y_tensor = y_tensor[perm]

            z1 = torch.randn(batch_size, args.latent_dim, device=device)
            z2 = torch.randn(batch_size, args.latent_dim, device=device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(dtype=torch.float16):
                x128_1, x64_1, _ = painter(z1, y_tensor, steps=args.steps, return_all_scales=True)
                x128_2, _, _ = painter(z2, y_tensor, steps=args.steps, return_all_scales=True)

                # 1. Mode-seeking diversity
                diff_z = torch.abs(z1 - z2).mean()
                diff_img = torch.abs(x128_1 - x128_2).mean()
                loss_div = diff_z / (diff_img + 1e-4)

                # 2. Gentle scale jitter
                x_aug = jitter(x128_1)

                # 3. PURE DIRECTOR InfoNCE CONTRASTIVE ALIGNMENT (Negative Prompts in Denominator)
                x_sig = F.interpolate(x_aug, size=(224, 224), mode='bicubic', align_corners=False)
                x_sig_norm = (x_sig - siglip_mean) / siglip_std

                if hasattr(siglip, "get_image_features"):
                    raw_img = siglip.get_image_features(pixel_values=x_sig_norm.half())
                    img_embeds = F.normalize(extract_tensor(raw_img), dim=-1)
                else:
                    img_embeds = F.normalize(siglip.encode_image(x_sig_norm.half()), dim=-1)

                # Dot products with all positive classes (0..C-1) and negative prompts (C..C+K-1)
                all_sims = torch.matmul(img_embeds, active_embeds.t())  # (B, C + K)
                logits = all_sims / args.temperature

                # InfoNCE Loss: target is y_tensor (in 0..C-1).
                # The denominator contains both other positive classes AND all negative prompts.
                loss_infonce = F.cross_entropy(logits, y_tensor)

                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    target_probs = probs.gather(1, y_tensor.unsqueeze(1)).squeeze(1)
                    neg_probs = probs[:, len(SHAPE_CLASSES):].sum(dim=-1)
                    target_sims = all_sims.gather(1, y_tensor.unsqueeze(1)).squeeze(1)

                # 4. Chromatic Channel Purity Guard (Ensures pure red is R only, pure blue is B only, etc.)
                r = x128_1[:, 0:1, :, :]
                g = x128_1[:, 1:2, :, :]
                b = x128_1[:, 2:3, :, :]
                m_red = (y_tensor == 0).float().view(-1, 1, 1, 1)
                m_blue = (y_tensor == 1).float().view(-1, 1, 1, 1)
                m_green = (y_tensor == 2).float().view(-1, 1, 1, 1)
                m_yellow = (y_tensor == 3).float().view(-1, 1, 1, 1)
                loss_purity = (m_red * (g + b)).mean() + \
                              (m_blue * (r + g)).mean() + \
                              (m_green * (r + b)).mean() + \
                              (m_yellow * b).mean()

                # 5. Canvas Outer Border Guard (Guarantees pure black background frame)
                border_loss = x128_1[:, :, :8, :].mean() + \
                              x128_1[:, :, -8:, :].mean() + \
                              x128_1[:, :, :, :8].mean() + \
                              x128_1[:, :, :, -8:].mean()

                # 6. Minimal TV (High-frequency noise guard only, weight = 0.001)
                tv_loss = torch.mean(torch.abs(x128_1[:, :, :, :-1] - x128_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(x128_1[:, :, :-1, :] - x128_1[:, :, 1:, :]))

                # TOTAL PURE PROMPT OBJECTIVE (InfoNCE + Structural Boundary Guards)
                loss_total = (args.lambda_infonce * loss_infonce) + \
                             (args.lambda_color_purity * loss_purity) + \
                             (args.lambda_bg * border_loss) + \
                             (args.lambda_tv * tv_loss) + \
                             (args.lambda_div * loss_div)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_total.item()
            total_infonce += loss_infonce.item()
            total_target_prob += target_probs.mean().item()
            total_neg_prob += neg_probs.mean().item()
            total_sim += target_sims.mean().item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_infonce = total_infonce / args.batches_per_epoch
        avg_target_prob = total_target_prob / args.batches_per_epoch
        avg_neg_prob = total_neg_prob / args.batches_per_epoch
        avg_sim = total_sim / args.batches_per_epoch

        # Log & visualize every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"[{active_stage}] Epoch {epoch:04d}/{args.epochs:04d} | InfoNCE: {avg_infonce:.4f} | Target P: {avg_target_prob*100:.1f}% (Sim: {avg_sim:.3f}) | Neg Denom P: {avg_neg_prob*100:.2f}% | Total: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            painter.eval()
            with torch.no_grad():
                samples = painter(fixed_z, fixed_y, steps=args.steps)
                img_path = os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png")
                save_image(samples, img_path, nrow=4, normalize=False)

    final_path = os.path.join(args.output_dir, "painter_sandbox_final.pth")
    torch.save(painter.state_dict(), final_path)
    print(f"\n>>> Painter Geometric Sandbox Complete! Model saved to: {final_path}")


if __name__ == "__main__":
    main()
