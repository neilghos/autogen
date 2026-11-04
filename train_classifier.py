import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import MultiScaleDecoderGenerator, get_classifier


class RobustBatchAugmentation(nn.Module):
    """
    GPU-accelerated anti-overfitting augmentations for synthetic streams:
    - Random horizontal flip
    - Random reflection crop
    - Random erasing / cutout
    """
    def __init__(self, pad=4, erase_prob=0.3):
        super().__init__()
        self.pad = pad
        self.erase_prob = erase_prob

    def forward(self, x):
        # 1. Random horizontal flip
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) > 0.5
        x = torch.where(mask, torch.flip(x, dims=[-1]), x)

        # 2. Random reflection crop
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        h = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        w = torch.randint(0, 2 * self.pad + 1, (1,), device=x.device).item()
        x = x_pad[:, :, h:h + 32, w:w + 32]

        # 3. Random erasing / cutout (prevents template memorization)
        if self.erase_prob > 0 and torch.rand(1, device=x.device).item() < self.erase_prob:
            e_h = torch.randint(4, 12, (1,), device=x.device).item()
            e_w = torch.randint(4, 12, (1,), device=x.device).item()
            y1 = torch.randint(0, 32 - e_h, (1,), device=x.device).item()
            x1 = torch.randint(0, 32 - e_w, (1,), device=x.device).item()
            noise = torch.rand(x.size(0), 3, e_h, e_w, device=x.device)
            x[:, :, y1:y1 + e_h, x1:x1 + e_w] = noise

        return x


def get_test_loader(data_dir=r"D:\auto-generative Neural network\data\data_cifar10", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[Dataset] Loaded {len(test_set)} real CIFAR-10 test photographs from: {data_dir}")
    return test_loader


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="High-Throughput Dataless Distillation: ConvNeXt Femto from Scratch")
    parser.add_argument("--epochs", type=int, default=100, help="Classifier training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Safe GPU batch size (default: 128, never OOMs)")
    parser.add_argument("--batches_per_epoch", type=int, default=500, help="Batches per epoch (e.g. 500 * 128 = 64,000 unique images/epoch)")
    parser.add_argument("--latent_dim", type=int, default=128, help="Generator latent dimension")
    parser.add_argument("--latent_scale", type=float, default=1.0, help="Latent noise scaling factor")
    parser.add_argument("--lr", type=float, default=5e-4, help="Classifier learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Cross entropy label smoothing")
    parser.add_argument("--use_mixup", action="store_true", default=True, help="Use Mixup regularization to break template overfitting")
    parser.add_argument("--generator_ckpt", type=str, default=r"D:\auto-generative Neural network\results_multiscale_generator\generator_multiscale_final.pth", help="Generator checkpoint")
    parser.add_argument("--data_dir", type=str, default=r"D:\auto-generative Neural network\data\data_cifar10", help="CIFAR-10 data dir")
    parser.add_argument("--output_dir", type=str, default="./results_distilled_classifier", help="Where to save logs and model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # 1. Load Real 10,000 CIFAR-10 Test Set
    test_loader = get_test_loader(args.data_dir, batch_size=128)

    # 2. Load and Freeze Trained Multi-Scale Generator
    print(f"\n[Generator] Loading trained checkpoint: {args.generator_ckpt}")
    if not os.path.exists(args.generator_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at: {args.generator_ckpt}")
        
    generator = MultiScaleDecoderGenerator(latent_dim=args.latent_dim, num_classes=10).to(device)
    generator.load_state_dict(torch.load(args.generator_ckpt, map_location=device, weights_only=True))
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False
    print("[Generator] MultiScaleDecoderGenerator frozen in inference mode.")

    # 3. Instantiate Student ConvNeXt Femto from scratch
    print("[Student] Instantiating ConvNeXt Femto from scratch (pretrained=False, 0% prior knowledge)...")
    classifier = get_classifier(model_name="convnext_femto", pretrained=False, num_classes=10).to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    augment = RobustBatchAugmentation(pad=4, erase_prob=0.3).to(device)

    init_acc = evaluate(classifier, test_loader, device)
    print(f"Initial Scratch Test Accuracy: {init_acc:.2f}%\n")

    total_images_per_epoch = args.batch_size * args.batches_per_epoch
    images_per_class = total_images_per_epoch // 10

    print("=" * 68)
    print("HIGH-THROUGHPUT DATALESS DISTILLATION: CONVNEXT FEMTO FROM SCRATCH")
    print(f"GPU Batch Size: {args.batch_size} (Safe Memory) | Batches/Epoch: {args.batches_per_epoch}")
    print(f"Throughput: {total_images_per_epoch:,} unique images per epoch (~{images_per_class:,}/class)!")
    print(f"Mixup & Cutout: ACTIVE (Breaks template overfitting)")
    print("=" * 68 + "\n")

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for _ in range(args.batches_per_epoch):
            # Uniformly sample classes across batch
            y_synth = torch.randint(0, 10, (args.batch_size,), device=device)

            # Sample fresh unique random latent noise
            z = torch.randn(args.batch_size, args.latent_dim, device=device) * args.latent_scale

            with torch.no_grad():
                x_synth = generator(z, y_synth)
                x_synth = augment(x_synth)

            # Mixup augmentation: blend images and targets
            if args.use_mixup and np.random.rand() > 0.3:
                lam = np.random.beta(0.4, 0.4)
                perm = torch.randperm(args.batch_size, device=device)
                x_train = lam * x_synth + (1 - lam) * x_synth[perm]
                y_a = y_synth
                y_b = y_synth[perm]

                optimizer.zero_grad()
                logits = classifier(x_train)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=-1)
                total_train_correct += ((preds == y_a) | (preds == y_b)).sum().item()
            else:
                optimizer.zero_grad()
                logits = classifier(x_synth)
                loss = criterion(logits, y_synth)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=-1)
                total_train_correct += (preds == y_synth).sum().item()

            total_loss += loss.item()
            total_train_samples += args.batch_size

        scheduler.step()
        avg_loss = total_loss / args.batches_per_epoch
        train_acc = 100.0 * total_train_correct / total_train_samples

        # Evaluate against the REAL CIFAR-10 test set
        if epoch % 5 == 0 or epoch == 1:
            test_acc = evaluate(classifier, test_loader, device)
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                torch.save(classifier.state_dict(), os.path.join(args.output_dir, "best_convnext_femto.pth"))
            
            star = " ★ (NEW BEST)" if is_best else ""
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | REAL TEST ACC: {test_acc:.2f}%{star}")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

    print(f"\n==================================================")
    print(f">>> Final Peak Real Test Accuracy: {best_acc:.2f}%")
    print(f">>> Best model saved to: {os.path.join(args.output_dir, 'best_convnext_femto.pth')}")
    print(f"==================================================")


if __name__ == "__main__":
    main()
