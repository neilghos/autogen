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
import timm

from models import MultiScaleDecoderGenerator


CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Mapping from CIFAR-10 classes to ImageNet-1k synsets for ConvNeXt spatial supervision
CIFAR_TO_IMAGENET = {
    0: [404, 895],                                                         # airplane: airliner, warplane
    1: [436, 468, 511, 609, 627, 656, 751, 817],                           # automobile: wagon, cab, convertible, jeep, limo, minivan, racer, sports car
    2: list(range(8, 25)) + list(range(80, 101)) + list(range(127, 147)),  # bird: songbirds, raptors, waterfowl, poultry
    3: [281, 282, 283, 284, 285],                                          # cat: tabby, tiger cat, Persian, Siamese, Egyptian
    4: [351, 352, 353, 350, 349],                                          # deer: hartebeest, impala, gazelle, ibex, bighorn
    5: list(range(151, 269)),                                              # dog: all 118 domestic dog breeds
    6: [25, 30, 31, 32],                                                   # frog: toad, bullfrog, tree frog, tailed frog
    7: [339, 340],                                                         # horse: sorrel horse, zebra
    8: [403, 510, 625, 724, 814, 871],                                    # ship: aircraft carrier, container ship, lifeboat, pirate ship, speedboat, trimaran
    9: [555, 569, 717, 734, 864, 867],                                    # truck: fire engine, garbage truck, pickup, police van, tow truck, trailer truck
}

# ==============================================================================
# ALTERNATING CURRICULUM PROMPT SYSTEM (Alternates Every 100 Epochs)
# ==============================================================================

# Phase 1: Full Scene & Complete Bounding Box (Macro wide-angle, grounds terrain & horizon)
CURRICULUM_PHASE_1_MACRO = [
    "a wide angle photo of a complete passenger airplane parked on airport tarmac under clear daylight sky",
    "a wide angle photo of a complete automobile car centered driving on an open asphalt road in daylight",
    "a wide angle photo of a complete wild songbird perched outdoors in natural forest scenery",
    "a wide angle photo of a complete domestic cat sitting on green grass lawn in natural daylight",
    "a wide angle photo of a complete wild deer standing in an open grassy forest meadow",
    "a wide angle photo of a complete domestic dog standing on a green grass field outdoors",
    "a wide angle photo of a complete green amphibian frog resting in a natural wet outdoor environment",
    "a wide angle photo of a complete equine horse standing in a rural grassy pasture",
    "a wide angle photo of a complete wooden sailing ship floating on open ocean water under daylight sky",
    "a wide angle photo of a complete commercial freight motor truck driving on a paved highway road"
]

# Phase 2 (Former Phase 3): Micro-Textures, Surface Speculars & Fine Physics (Applied to patch crops)
CURRICULUM_PHASE_2_MICRO = [
    "detailed texture of passenger airplane polished aluminum rivets, jet engine intake turbine, and wing seams",
    "detailed texture of automobile deep rubber tire tread grooves, alloy wheel rims, and glass headlights",
    "detailed texture of wild bird plumage feather barbules, dark pupil eye, and sharp pointed beak",
    "detailed texture of domestic cat whiskers, dense soft fur coat fibers, and inner ear hair",
    "detailed texture of wild deer antler tines, soft velvet coat texture, and alert dark eyes",
    "detailed texture of domestic dog textured wet nose leather, fine muzzle whiskers, and soft fur",
    "detailed texture of green tree frog glistening wet skin texture, golden rimmed eye, and throat skin",
    "detailed texture of horse silky flowing mane hair strands, glossy coat highlights, and dark eyes",
    "detailed texture of sailing ship wooden hull planks, textured ropes, and woven fabric sails",
    "detailed texture of heavy commercial truck chrome front grill, large tire treads, and side mirrors"
]

NEGATIVE_PROMPTS = [
    "cropped, extreme close up, zoomed in, macro, severed head, headless, dismembered, detached parts, tire only, wheel only, floating pieces, blurry, foggy, out of focus, low contrast, abstract dots, scanlines, distorted, surreal cartoon, blown out white, overexposed, washed out"
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
    Differentiable stochastic spatial jitter:
    - Translation shifts (+- pad)
    - Random horizontal flip
    """
    def __init__(self, pad=8, img_size=128):
        super().__init__()
        self.pad = pad
        self.img_size = img_size

    def forward(self, x):
        # 1. Random horizontal flip
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) > 0.5
        x = torch.where(mask, torch.flip(x, dims=[-1]), x)

        # 2. Random translation pad & crop
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
    print("[SigLIP Judge] Loading google/siglip-base-patch16-224...")
    try:
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device).half()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[SigLIP Judge] Loaded via Hugging Face Transformers.\n")
        return processor.tokenizer, model
    except Exception as e:
        print(f"[SigLIP Judge] HF loader failed ({e}), falling back to open_clip...")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model = model.to(device).half()
        tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print("[SigLIP Judge] Loaded via open_clip.\n")
        return tokenizer, model


def load_spatial_critic(device, model_name="convnext_femto.d1_in1k"):
    """
    Loads pretrained ConvNeXt Femto classifier on ImageNet-1k as the Spatial Critic.
    """
    print(f"[Spatial Critic] Loading pretrained {model_name}...")
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[Spatial Critic] Loaded successfully and frozen.\n")
    return model


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
    Precomputes normalized SigLIP text embeddings for both curriculum phases.
    """
    print("[Curriculum Engine] Precomputing multi-scale prompt embeddings...")
    embeds_p1 = encode_prompts(CURRICULUM_PHASE_1_MACRO, tokenizer, model, device)
    embeds_p2 = encode_prompts(CURRICULUM_PHASE_2_MICRO, tokenizer, model, device)
    embeds_neg = encode_prompts(NEGATIVE_PROMPTS, tokenizer, model, device)
    print("  - Phase 1 (Macro):", embeds_p1.shape)
    print("  - Phase 2 (Micro):", embeds_p2.shape)
    print("  - Negative:", embeds_neg.shape, "\n")
    return embeds_p1, embeds_p2, embeds_neg


def main():
    parser = argparse.ArgumentParser(description="Recurrent Feedback Generator with Alternating Macro/Micro Curriculum")
    parser.add_argument("--epochs", type=int, default=500, help="Total training epochs (Default: 500)")
    parser.add_argument("--phase_cycle", type=int, default=100, help="Epochs per phase before alternating (Default: 100)")
    parser.add_argument("--batches_per_epoch", type=int, default=15, help="Batches per epoch")
    parser.add_argument("--samples_per_class", type=int, default=2, help="Samples per class per batch (Default: 2 -> Batch size = 20)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent noise vector dimension")
    parser.add_argument("--steps", type=int, default=2, help="Recurrent feedback steps per image (Default: 2)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")

    # Supervision & Loss Weights
    parser.add_argument("--lambda_siglip", type=float, default=5.0, help="Weight for Global SigLIP Alignment")
    parser.add_argument("--lambda_patch", type=float, default=2.5, help="Weight for Zoomed Patch Details")
    parser.add_argument("--lambda_spatial", type=float, default=2.0, help="Weight for ConvNeXt Spatial Critic")
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
    embeds_p1, embeds_p2, neg_embeds = precompute_curriculum_anchors(tokenizer, siglip, device)

    # 2. Load Pretrained ConvNeXt Spatial Critic
    spatial_critic = load_spatial_critic(device)

    # Construct vectorized CIFAR -> ImageNet 1k probability projection matrix
    cifar_imagenet_mask = torch.zeros(10, 1000, device=device)
    for c, idxs in CIFAR_TO_IMAGENET.items():
        cifar_imagenet_mask[c, idxs] = 1.0

    # Normalization statistics
    siglip_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    siglip_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Natural color statistics target (from real CIFAR-10)
    target_mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    target_std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    # 3. Recurrent Feedback Decoder Generator (128x128)
    batch_size = 10 * args.samples_per_class
    print("=" * 72)
    print("RECURRENT FEEDBACK AUTOREGRESSIVE GENERATOR (ALTERNATING CURRICULUM)")
    print(f"  - Canvas Resolution: 128 x 128 native pixels")
    print(f"  - Feedback Recurrence Steps: {args.steps} iterations / image")
    print(f"  - Alternation Cycle: Every {args.phase_cycle} epochs")
    print(f"    * Phase 1 (Macro): Full Scene & Complete Bounding Box")
    print(f"    * Phase 2 (Micro): Micro-Textures, Speculars & Fine Physics on Patches")
    print(f"  - Spatial Topology Critic: ConvNeXt Femto (ImageNet-1k) Active (Weight: {args.lambda_spatial})")
    print(f"  - Batches/Epoch: {args.batches_per_epoch} | Batch Size: {batch_size}")
    print("=" * 72 + "\n")

    generator = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()
    jitter = StochasticJitter(pad=8, img_size=128).to(device)
    laplacian = LaplacianSharpness().to(device)

    # Fixed latents for progress tracking across all 10 classes
    fixed_z = torch.randn(10, args.latent_dim, device=device)
    fixed_y = torch.arange(10, device=device)

    logit_scale = getattr(siglip, "logit_scale", nn.Parameter(torch.tensor(10.0, device=device))).exp() if hasattr(siglip, "logit_scale") else 10.0
    logit_bias = getattr(siglip, "logit_bias", torch.tensor(-10.0, device=device)) if hasattr(siglip, "logit_bias") else -10.0

    current_phase = 0

    for epoch in range(1, args.epochs + 1):
        # Determine Curriculum Phase (Alternating every phase_cycle epochs: 1-100, 101-200, 201-300...)
        cycle_idx = (epoch - 1) // args.phase_cycle
        if cycle_idx % 2 == 0:
            active_phase = 1
            phase_title = "Phase 1: Full Scene & Complete Bounding Box (Macro)"
            active_global_embeds = embeds_p1
            active_patch_embeds = embeds_p1
            crop_min, crop_max = 64, 96
        else:
            active_phase = 2
            phase_title = "Phase 2: Micro-Textures & Fine Surface Physics (Micro)"
            active_global_embeds = embeds_p1
            active_patch_embeds = embeds_p2
            crop_min, crop_max = 48, 72

        if active_phase != current_phase:
            current_phase = active_phase
            print(f"\n{'='*72}")
            print(f">>> [CURRICULUM SHIFT] ENTERING {phase_title.upper()} (Epoch {epoch})")
            print(f"{'='*72}\n")

        generator.train()

        total_loss = 0.0
        total_siglip_loss = 0.0
        total_spatial_loss = 0.0
        total_spatial_conf = 0.0
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

            with torch.amp.autocast('cuda', dtype=torch.float16) if hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(dtype=torch.float16):
                x128_1, x64_1, _ = generator(z1, y_tensor, steps=args.steps, return_all_scales=True)
                x128_2, _, _ = generator(z2, y_tensor, steps=args.steps, return_all_scales=True)

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

                # 4. SPATIAL CONVNEXT CRITIC (Enforces Complete 3D Object Anatomy)
                x_crit = F.interpolate(x_aug, size=(224, 224), mode='bicubic', align_corners=False)
                x_crit_norm = (x_crit - imagenet_mean) / imagenet_std
                crit_logits = spatial_critic(x_crit_norm)
                crit_probs = F.softmax(crit_logits, dim=-1)
                batch_crit_mask = cifar_imagenet_mask[y_tensor]  # (B, 1000)
                spatial_conf = (crit_probs * batch_crit_mask).sum(dim=-1)
                loss_spatial = -torch.log(spatial_conf + 1e-6).mean()

                # 5. MULTI-CROP ZOOMED PATCH DETAIL (Phase-Specific)
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

                # 6. LAPLACIAN HIGH-FREQUENCY EDGE FLOOR
                edge_energy = laplacian(x128_1)
                loss_sharpness = torch.relu(0.045 - edge_energy).mean()

                # 7. Negative Prompt Repulsion
                neg_sims = torch.matmul(img_embeds, neg_embeds.t()).max(dim=-1)[0]
                loss_neg = torch.relu(neg_sims).mean()

                # 8. Balanced Contrast Range
                per_img_std = x128_1.std(dim=[-2, -1])  # (B, 3)
                loss_contrast = torch.relu(0.20 - per_img_std).mean() + torch.relu(per_img_std - 0.32).mean()

                # 9. Anti-Washout Exposure Guard
                loss_exposure = torch.relu(x128_1 - 0.92).pow(2).mean() + torch.relu(0.04 - x128_1).pow(2).mean()

                # 10. Natural Daylight Color Moments
                batch_m = x128_1.mean(dim=[0, 2, 3], keepdim=True)
                batch_s = x128_1.std(dim=[0, 2, 3], keepdim=True)
                loss_color = F.mse_loss(batch_m, target_mean) + F.mse_loss(batch_s, target_std)

                # 11. Slashed Total Variation
                tv_loss = torch.mean(torch.abs(x128_1[:, :, :, :-1] - x128_1[:, :, :, 1:])) + \
                          torch.mean(torch.abs(x128_1[:, :, :-1, :] - x128_1[:, :, 1:, :]))

                # TOTAL CURRICULUM OBJECTIVE
                loss_total = (args.lambda_siglip * loss_siglip) + \
                             (args.lambda_patch * loss_patch) + \
                             (args.lambda_spatial * loss_spatial) + \
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
            total_spatial_loss += loss_spatial.item()
            total_spatial_conf += spatial_conf.mean().item()
            total_sim += target_sims.mean().item()
            total_edge += edge_energy.mean().item()
            total_div += diff_img.item()

        scheduler.step()

        avg_loss = total_loss / args.batches_per_epoch
        avg_sig = total_siglip_loss / args.batches_per_epoch
        avg_spatial = total_spatial_loss / args.batches_per_epoch
        avg_spconf = total_spatial_conf / args.batches_per_epoch
        avg_sim = total_sim / args.batches_per_epoch
        avg_edge = total_edge / args.batches_per_epoch
        avg_div = total_div / args.batches_per_epoch

        # Sample grid every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{active_phase}] Epoch {epoch:04d}/{args.epochs:04d} | SigLIP: {avg_sig:.4f} | Spatial: {avg_spatial:.4f} (Conf: {avg_spconf:.1%}) | Edge: {avg_edge:.4f} | Total: {avg_loss:.4f}")

            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z, fixed_y, steps=args.steps)
                save_image(samples, os.path.join(args.output_dir, f"sample_epoch_{epoch:04d}.png"), nrow=5, normalize=False)

        elif epoch % 2 == 0:
            print(f"[{active_phase}] Epoch {epoch:04d}/{args.epochs:04d} | SigLIP: {avg_sig:.4f} | Spatial: {avg_spatial:.4f} (Conf: {avg_spconf:.1%}) | Edge: {avg_edge:.4f} | Total: {avg_loss:.4f}")

    # Final save
    final_path = os.path.join(args.output_dir, "generator_vlm_final.pth")
    torch.save(generator.state_dict(), final_path)
    print(f"\n>>> Recurrent Feedback Synthesis Complete! Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
