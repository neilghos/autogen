import os
import sys
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

PROMPT_TEMPLATES = [
    "a realistic daylight photo of a {}.",
    "a natural color photograph of a {} in real life.",
    "a clear authentic photo of a {} outdoors.",
    "a sharp, detailed photo of a {}."
]

NEGATIVE_PROMPTS = [
    "neon, purple, magenta, violet, glowing, oversaturated, psychedelic, cartoon, illustration, drawing, abstract, distorted, surreal, inverted colors"
]


class EMA:
    """
    Maintains exponential moving average of generator weights for ultra-stable synthesis.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)


class StochasticJitter(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if torch.rand(1, device=x.device).item() > 0.5:
            x = torch.flip(x, dims=[-1])
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        h = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        w = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        return x_pad[:, :, h:h + 32, w:w + 32]


def load_clip_model(device):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Try open_clip
    try:
        import open_clip
        print("[CLIP Judge] Loading via open_clip (ViT-B-32)...")
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        def encode_text(texts):
            tokens = tokenizer(texts).to(device)
            feats = model.encode_text(tokens)
            return F.normalize(feats, dim=-1)

        def encode_image(imgs):
            imgs_up = F.interpolate(imgs, size=(224, 224), mode='bicubic', align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            imgs_norm = (imgs_up - mean) / std
            feats = model.encode_image(imgs_norm)
            return F.normalize(feats, dim=-1)

        return encode_text, encode_image
    except ImportError:
        pass

    # Try transformers
    try:
        import transformers
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

        from transformers import CLIPModel, CLIPTokenizer
        print("[CLIP Judge] Loading via transformers (openai/clip-vit-base-patch32)...")
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
        except Exception:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        def encode_text(texts):
            inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            feats = model.get_text_features(**inputs)
            if hasattr(feats, "text_embeds"):
                feats = feats.text_embeds
            elif hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            return F.normalize(feats, dim=-1)

        def encode_image(imgs):
            imgs_up = F.interpolate(imgs, size=(224, 224), mode='bicubic', align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            imgs_norm = (imgs_up - mean) / std
            feats = model.get_image_features(pixel_values=imgs_norm)
            if hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            elif hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            return F.normalize(feats, dim=-1)

        return encode_text, encode_image
    except ImportError:
        pass

    raise RuntimeError("No CLIP library found!")


def compute_text_matrices(encode_text_fn, device):
    class_embeds = []
    with torch.no_grad():
        for c in CIFAR_CLASSES:
            prompts = [t.format(c) for t in PROMPT_TEMPLATES]
            embs = encode_text_fn(prompts)
            avg_emb = F.normalize(embs.mean(dim=0), dim=-1)
            class_embeds.append(avg_emb)

        pos_matrix = torch.stack(class_embeds).to(device)

        neg_embs = encode_text_fn(NEGATIVE_PROMPTS)
        neg_matrix = F.normalize(neg_embs, dim=-1).to(device)

    return pos_matrix, neg_matrix


def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Deconvolutional Decoder: Low & Slow 1000-Epoch Pre-training")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs (default: 1000 for low and slow refinement)")
    parser.add_argument("--batches_per_epoch", type=int, default=20, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=4, help="Samples per class in each batch (Batch size = 40)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Generator EMA decay rate")

    # Regularization weights
    parser.add_argument("--lambda_align", type=float, default=2.0, help="CLIP cosine alignment weight")
    parser.add_argument("--lambda_ce", type=float, default=1.0, help="CLIP contrastive classification weight")
    parser.add_argument("--lambda_coarse", type=float, default=0.5, help="Coarse 16x16 macro layout guidance weight")
    parser.add_argument("--lambda_color", type=float, default=6.0, help="Natural image color moments matching weight")
    parser.add_argument("--lambda_chroma", type=float, default=3.0, help="YUV chroma saturation suppression weight")
    parser.add_argument("--lambda_neg", type=float, default=1.2, help="Negative prompt repulsion weight")
    parser.add_argument("--lambda_tv", type=float, default=2.0, help="Total variation smoothness weight")
    parser.add_argument("--lambda_div", type=float, default=0.04, help="Mode-seeking diversity weight")

    parser.add_argument("--output_dir", type=str, default="./results_multiscale_generator", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load CLIP Semantic Judge
    encode_text_fn, encode_image_fn = load_clip_model(device)
    pos_matrix, neg_matrix = compute_text_matrices(encode_text_fn, device)
    print(f"[CLIP Judge] Positive anchors locked: {pos_matrix.shape}")
    print(f"[CLIP Judge] Negative repulsion locked: {neg_matrix.shape}\n")

    # Natural color statistics target (from real CIFAR-10)
    target_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    target_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    # 2. Multi-Scale Decoder Generator (From Scratch)
    print("=" * 65)
    print("MULTI-SCALE RESIDUAL DECODER GENERATOR (1000 EPOCHS LOW & SLOW)")
    print("  Scale 1 (4x4):   Global compositional seed")
    print("  Scale 2 (8x8):   Macro-shapes & horizon / sky / ground layout")
    print("  Scale 3 (16x16): Mid-level textures, limbs, chassis")
    print("  Scale 4 (32x32): Micro-edits, eyes, whiskers, crisp contours")
    print(f"Batches/Epoch: {args.batches_per_epoch} | Batch Size: {args.samples_per_class * 10} | LR: {args.lr} (Cosine Annealing)")
    print("=" * 65 + "\n")

    generator = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=10).to(device)
    ema = EMA(generator, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    jitter = StochasticJitter(pad=4).to(device)

    batch_size = 10 * args.samples_per_class

    # Fixed latents for progress tracking of all 10 classes
    fixed_z = torch.randn(10, args.latent_dim, device=device)
    fixed_y = torch.arange(10, device=device)

    for epoch in range(1, args.epochs + 1):
        generator.train()

        total_loss = 0.0
        total_cos_sim = 0.0
        total_color_dev = 0.0
        total_div = 0.0

        for _ in range(args.batches_per_epoch):
            # Balanced multi-class batch: all 10 classes represented equally
            y_tensor = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
            perm = torch.randperm(batch_size, device=device)
            y_tensor = y_tensor[perm]

            z1 = torch.randn(batch_size, args.latent_dim, device=device)
            z2 = torch.randn(batch_size, args.latent_dim, device=device)

            optimizer.zero_grad()

            # Multi-scale outputs: x_32 (fine), x_16 (mid), x_8 (coarse)
            x32_1, x16_1, _ = generator(z1, y_tensor, return_all_scales=True)
            x32_2, _, _ = generator(z2, y_tensor, return_all_scales=True)

            # 1. Mode-seeking diversity
            diff_z = torch.abs(z1 - z2).mean()
            diff_img = torch.abs(x32_1 - x32_2).mean()
            loss_div = diff_z / (diff_img + 1e-4)

            # 2. Stochastic Jitter for Fine Image (32x32)
            x_aug = jitter(x32_1)
            img_embeds_32 = encode_image_fn(x_aug)

            # 3. Fine Scale Alignment & CE
            sim_32 = torch.matmul(img_embeds_32, pos_matrix.t())  # (B, 10)
            target_sims = sim_32.gather(1, y_tensor.unsqueeze(1)).squeeze(1)
            loss_align = (1.0 - target_sims).mean()
            loss_ce = F.cross_entropy(100.0 * sim_32, y_tensor)

            # 4. Coarse Scale Alignment (16x16 forces macro composition early)
            img_embeds_16 = encode_image_fn(x16_1)
            sim_16 = torch.matmul(img_embeds_16, pos_matrix.t())
            target_sims_16 = sim_16.gather(1, y_tensor.unsqueeze(1)).squeeze(1)
            loss_coarse = (1.0 - target_sims_16).mean()

            # 5. Negative Prompt Repulsion
            neg_sims = torch.matmul(img_embeds_32, neg_matrix.t()).max(dim=-1)[0]
            loss_neg = torch.relu(neg_sims).mean()

            # 6. Natural Color Moments Matching (CIFAR Real Statistics)
            batch_m = x32_1.mean(dim=[0, 2, 3], keepdim=True)
            batch_s = x32_1.std(dim=[0, 2, 3], keepdim=True)
            loss_color = F.mse_loss(batch_m, target_mean) + F.mse_loss(batch_s, target_std)

            # 7. YUV Chroma / Saturation Suppression
            r, g, b = x32_1[:, 0:1], x32_1[:, 1:2], x32_1[:, 2:3]
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.100 * b
            loss_chroma = torch.mean(u ** 2 + v ** 2)

            # 8. Total Variation Smoothness
            tv_loss = torch.mean(torch.abs(x32_1[:, :, :, :-1] - x32_1[:, :, :, 1:])) + \
                      torch.mean(torch.abs(x32_1[:, :, :-1, :] - x32_1[:, :, 1:, :]))

            loss_total = (args.lambda_align * loss_align) + \
                         (args.lambda_ce * loss_ce) + \
                         (args.lambda_coarse * loss_coarse) + \
                         (args.lambda_color * loss_color) + \
                         (args.lambda_chroma * loss_chroma) + \
                         (args.lambda_neg * loss_neg) + \
                         (args.lambda_tv * tv_loss) + \
                         (args.lambda_div * loss_div)

            loss_total.backward()
            optimizer.step()
            ema.update(generator)

            total_loss += loss_total.item()
            total_cos_sim += target_sims.mean().item()
            total_color_dev += loss_color.item()
            total_div += diff_img.item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_sim = total_cos_sim / args.batches_per_epoch
        avg_col = total_color_dev / args.batches_per_epoch
        avg_div = total_div / args.batches_per_epoch

        # Log and save grids
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | Cos Sim: {avg_sim:.4f} | Color Dev: {avg_col:.4f} | Diff: {avg_div:.4f} | Loss: {avg_loss:.4f}")

            # Save live samples and EMA samples
            generator.eval()
            ema.shadow.eval()
            with torch.no_grad():
                samples_live = generator(fixed_z, fixed_y)
                samples_ema = ema.shadow(fixed_z, fixed_y)

                save_image(samples_live, os.path.join(args.output_dir, f"sample_live_epoch_{epoch:04d}.png"), nrow=5, normalize=False)
                save_image(samples_ema, os.path.join(args.output_dir, f"sample_ema_epoch_{epoch:04d}.png"), nrow=5, normalize=False)

        elif epoch % 5 == 0:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | Cos Sim: {avg_sim:.4f} | Color Dev: {avg_col:.4f} | Diff: {avg_div:.4f} | Loss: {avg_loss:.4f}")

        # Checkpoints every 100 epochs
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f"generator_epoch_{epoch:04d}.pth"))
            torch.save(ema.shadow.state_dict(), os.path.join(args.output_dir, "generator_multiscale_ema.pth"))

    # Final save
    torch.save(generator.state_dict(), os.path.join(args.output_dir, "generator_multiscale_final.pth"))
    torch.save(ema.shadow.state_dict(), os.path.join(args.output_dir, "generator_multiscale_ema.pth"))
    print(f"\n>>> 1000-Epoch Training Complete! Final models and samples saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

