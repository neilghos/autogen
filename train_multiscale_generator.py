import os
import sys

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

CLASS_DESCRIPTIONS = {
    "airplane": [
        "a commercial passenger airliner flying through clear blue sky, sharp fuselage and wings",
        "a sleek modern jet airplane on airport runway tarmac in bright daylight",
        "a sharp, detailed photo of a jet aircraft with wings and tail fin"
    ],
    "automobile": [
        "a sleek sports car coupe with shiny paint, headlights and rubber wheels on road",
        "a detailed photograph of a vintage automobile parked on street pavement",
        "a clear photo of an automobile driving on tarmac highway in daylight"
    ],
    "bird": [
        "a detailed macro photo of a songbird with sharp beak, dark eye and colorful plumage",
        "a small colorful wild bird perched alert on a tree branch with blurred foliage",
        "a vivid close up portrait of an authentic wild bird in nature"
    ],
    "cat": [
        "a close up portrait of a furry domestic cat with pointed ears, whiskers and bright eyes",
        "a cute tabby cat sitting upright in natural daylight, sharp focus on fur",
        "an authentic photograph of a domestic cat outdoors in green garden"
    ],
    "deer": [
        "a majestic wild deer stag with antlers and white throat standing in tall green grass",
        "a clear wildlife photo of a golden brown deer standing alert in forest meadow",
        "a close up portrait of a wild fawn deer in sunlight outdoors"
    ],
    "dog": [
        "a close up portrait of a happy domestic dog with floppy ears, dark nose and soft fur",
        "a golden retriever puppy dog looking at camera in sunny green park",
        "an authentic photograph of a friendly dog outdoors on grass"
    ],
    "frog": [
        "a vivid green tree frog with glistening skin and bulbous eyes perched on moss",
        "a macro photograph of a colorful amphibian bullfrog in outdoor natural wetland",
        "a clear close up of a small green frog sitting alert on green leaf"
    ],
    "horse": [
        "a strong chestnut brown horse stallion standing in a lush green pasture",
        "a full body photo of a noble equine horse with flowing mane and hooves on grass",
        "a majestic brown mare horse in sunny meadow outdoors"
    ],
    "ship": [
        "a tall wooden galleon sailing ship with white sails on deep blue sea water",
        "a large maritime passenger ocean vessel floating on ocean waves in daylight",
        "an authentic photograph of a majestic ship sailing on open water"
    ],
    "truck": [
        "a heavy duty commercial freight cargo truck with front grill and big tires on asphalt",
        "a large transport semi-truck parked outdoors on concrete road in sunlight",
        "a clear photo of a motor truck vehicle on highway pavement"
    ]
}

NEGATIVE_PROMPTS = [
    "neon, purple, magenta, violet, glowing, oversaturated, psychedelic, cartoon, illustration, drawing, abstract, distorted, surreal, inverted colors"
]


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
            prompts = CLASS_DESCRIPTIONS[c]
            embs = encode_text_fn(prompts)
            avg_emb = F.normalize(embs.mean(dim=0), dim=-1)
            class_embeds.append(avg_emb)

        pos_matrix = torch.stack(class_embeds).to(device)

        neg_embs = encode_text_fn(NEGATIVE_PROMPTS)
        neg_matrix = F.normalize(neg_embs, dim=-1).to(device)

    return pos_matrix, neg_matrix


def main():
    parser = argparse.ArgumentParser(description="Multi-Scale Deconvolutional Decoder: High-Contrast Pure Generator")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--batches_per_epoch", type=int, default=25, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=4, help="Samples per class in each batch (Batch size = 40)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # Regularization & Goal-Seeking Target Margin weights
    parser.add_argument("--target_sim", type=float, default=0.75, help="Target cosine similarity margin (default: 0.75)")
    parser.add_argument("--lambda_align", type=float, default=12.0, help="Quadratic margin pull weight (default: 12.0)")
    parser.add_argument("--lambda_ce", type=float, default=1.0, help="CLIP contrastive classification weight")
    parser.add_argument("--lambda_coarse", type=float, default=1.0, help="Coarse 16x16 macro layout guidance weight")
    parser.add_argument("--lambda_contrast", type=float, default=4.0, help="Per-image contrast floor (prevents fog / wash-out)")
    parser.add_argument("--lambda_color", type=float, default=2.0, help="Natural image color moments matching weight")
    parser.add_argument("--lambda_chroma", type=float, default=1.0, help="Hinge chroma penalty (only kills neon glow)")
    parser.add_argument("--lambda_neg", type=float, default=1.0, help="Negative prompt repulsion weight")
    parser.add_argument("--lambda_tv", type=float, default=0.01, help="Light TV weight (no blur)")
    parser.add_argument("--lambda_div", type=float, default=0.04, help="Mode-seeking diversity weight")

    parser.add_argument("--output_dir", type=str, default="./results_multiscale_generator", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load CLIP Semantic Judge
    encode_text_fn, encode_image_fn = load_clip_model(device)
    pos_matrix, neg_matrix = compute_text_matrices(encode_text_fn, device)
    print(f"[CLIP Judge] High-Information Visual Anchors locked: {pos_matrix.shape}")
    print(f"[CLIP Judge] Negative repulsion locked: {neg_matrix.shape}\n")

    # Natural color statistics target (from real CIFAR-10)
    target_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    target_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    # 2. Multi-Scale Decoder Generator (32x32)
    print("=" * 65)
    print("MULTI-SCALE DECODER: SUPERCHARGED QUADRATIC MARGIN PULL")
    print(f"  - Target Cosine Goal: {args.target_sim:.2f}")
    print(f"  - Quadratic Pull Weight (lambda_align): {args.lambda_align:.1f}")
    print(f"  - Visually Rich Class Prompt Ensembles: ACTIVE")
    print(f"Batches/Epoch: {args.batches_per_epoch} | Batch Size: {args.samples_per_class * 10}")
    print("=" * 65 + "\n")

    generator = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=10).to(device)
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
        total_contrast = 0.0
        total_div = 0.0

        for _ in range(args.batches_per_epoch):
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
            sim_32 = torch.matmul(img_embeds_32, pos_matrix.t())
            target_sims = sim_32.gather(1, y_tensor.unsqueeze(1)).squeeze(1)

            # Quadratic Goal-Seeking Margin Pull towards target_sim (0.75)
            loss_align = torch.mean(torch.relu(args.target_sim - target_sims) ** 2)
            loss_ce = F.cross_entropy(100.0 * sim_32, y_tensor)

            # 4. Coarse Scale Alignment (16x16 macro composition)
            img_embeds_16 = encode_image_fn(x16_1)
            sim_16 = torch.matmul(img_embeds_16, pos_matrix.t())
            target_sims_16 = sim_16.gather(1, y_tensor.unsqueeze(1)).squeeze(1)
            loss_coarse = torch.mean(torch.relu(0.60 - target_sims_16) ** 2)

            # 5. Negative Prompt Repulsion
            neg_sims = torch.matmul(img_embeds_32, neg_matrix.t()).max(dim=-1)[0]
            loss_neg = torch.relu(neg_sims).mean()

            # 6. Per-Image Contrast Floor (Prevents foggy / washed out collapse)
            per_img_std = x32_1.std(dim=[-2, -1])  # (B, 3)
            loss_contrast = torch.relu(0.22 - per_img_std).mean()

            # 7. Natural Color Moments Matching
            batch_m = x32_1.mean(dim=[0, 2, 3], keepdim=True)
            batch_s = x32_1.std(dim=[0, 2, 3], keepdim=True)
            loss_color = F.mse_loss(batch_m, target_mean) + F.mse_loss(batch_s, target_std)

            # 8. Hinge Chroma Penalty (ONLY penalizes neon / psychedelic glow, preserves rich colors!)
            r, g, b = x32_1[:, 0:1], x32_1[:, 1:2], x32_1[:, 2:3]
            u = -0.147 * r - 0.289 * g + 0.436 * b
            v = 0.615 * r - 0.515 * g - 0.100 * b
            chroma = u ** 2 + v ** 2
            loss_chroma = torch.relu(chroma - 0.08).mean()

            # 9. Light Total Variation (only kills single-pixel speckle)
            tv_loss = torch.mean(torch.abs(x32_1[:, :, :, :-1] - x32_1[:, :, :, 1:])) + \
                      torch.mean(torch.abs(x32_1[:, :, :-1, :] - x32_1[:, :, 1:, :]))

            loss_total = (args.lambda_align * loss_align) + \
                         (args.lambda_ce * loss_ce) + \
                         (args.lambda_coarse * loss_coarse) + \
                         (args.lambda_contrast * loss_contrast) + \
                         (args.lambda_color * loss_color) + \
                         (args.lambda_chroma * loss_chroma) + \
                         (args.lambda_neg * loss_neg) + \
                         (args.lambda_tv * tv_loss) + \
                         (args.lambda_div * loss_div)

            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()
            total_cos_sim += target_sims.mean().item()
            total_contrast += per_img_std.mean().item()
            total_div += diff_img.item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_sim = total_cos_sim / args.batches_per_epoch
        avg_std = total_contrast / args.batches_per_epoch
        avg_div = total_div / args.batches_per_epoch

        # Save sample grid every 25 epochs
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | Cos Sim: {avg_sim:.4f} | Contrast(std): {avg_std:.4f} | Diff: {avg_div:.4f} | Loss: {avg_loss:.4f}")

            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z, fixed_y)
                save_image(samples, os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png"), nrow=5, normalize=False)

        elif epoch % 5 == 0:
            print(f"Epoch {epoch:04d}/{args.epochs:04d} | Cos Sim: {avg_sim:.4f} | Contrast(std): {avg_std:.4f} | Diff: {avg_div:.4f} | Loss: {avg_loss:.4f}")

        # Checkpoints every 100 epochs
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f"generator_epoch_{epoch:04d}.pth"))

    # Final save
    final_path = os.path.join(args.output_dir, "generator_multiscale_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\n>>> Training Complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
