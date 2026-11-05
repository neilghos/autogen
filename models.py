import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def get_classifier(model_name="convnext_femto", pretrained=False, num_classes=10, in_chans=3):
    """
    Instantiates ConvNeXt Femto classifier.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans
    )
    return model


class DecoderResBlock(nn.Module):
    """
    Residual block with nearest-neighbor upsampling and reflection padding.
    Completely eliminates deconvolution checkerboard artifacts.
    """
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        if upsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            
        residual = self.shortcut(x)
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act2(h + residual)


class MultiScaleDecoderGenerator(nn.Module):
    """
    Recurrent Feedback Multi-Scale Decoder Generator (3x128x128).
    
    Iterative Autoregressive Refinement:
      - Step 1: Seeds initial canvas (coarse silhouette, layout) from (z, c)
      - Step 2+: Feeds previous canvas back into 4x4 stem via feedback residual connection,
                 enabling self-observation, error correction, and multi-stage refinement.
    """
    def __init__(self, latent_dim=128, num_classes=10, embed_dim=64, base_channels=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        
        # 1. Conditioning Mapping MLP
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. Stage 1: 4x4 initial tensor
        self.init_fc = nn.Sequential(
            nn.Linear(256, base_channels * 4 * 4),
            nn.BatchNorm1d(base_channels * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. Recurrent Feedback Stem (projects previous 128x128 canvas back into 4x4 residual)
        self.feedback_stem = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Initialize feedback projection gently
        nn.init.zeros_(self.feedback_stem[1].weight)
        
        # 4. Stage 2: 4x4 -> 8x8 (Macro shapes)
        self.block_8 = DecoderResBlock(base_channels, 192, upsample=True)
        self.to_rgb_8 = nn.Conv2d(192, 3, kernel_size=1)
        
        # 5. Stage 3: 8x8 -> 16x16 (Mid-level parts)
        self.block_16 = DecoderResBlock(192, 128, upsample=True)
        self.to_rgb_16 = nn.Conv2d(128, 3, kernel_size=1)
        
        # 6. Stage 4: 16x16 -> 32x32 (Anatomy & shapes)
        self.block_32 = DecoderResBlock(128, 96, upsample=True)
        self.to_rgb_32 = nn.Conv2d(96, 3, kernel_size=1)

        # 7. Stage 5: 32x32 -> 64x64 (Surface details)
        self.block_64 = DecoderResBlock(96, 64, upsample=True)
        self.to_rgb_64 = nn.Conv2d(64, 3, kernel_size=1)

        # 8. Stage 6: 64x64 -> 128x128 (Micro-edits, crisp contours)
        self.block_128 = DecoderResBlock(64, 32, upsample=True)
        self.to_rgb_128 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def _decode_from_h4(self, h4):
        # Stage 2: 8x8
        h8 = self.block_8(h4)
        rgb_8 = self.to_rgb_8(h8)
        
        # Stage 3: 16x16 (Residual addition from 8x8)
        h16 = self.block_16(h8)
        rgb_16 = self.to_rgb_16(h16) + F.interpolate(rgb_8, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Stage 4: 32x32 (Residual addition from 16x16)
        h32 = self.block_32(h16)
        rgb_32 = self.to_rgb_32(h32) + F.interpolate(rgb_16, scale_factor=2, mode='bilinear', align_corners=False)

        # Stage 5: 64x64 (Residual addition from 32x32)
        h64 = self.block_64(h32)
        rgb_64 = self.to_rgb_64(h64) + F.interpolate(rgb_32, scale_factor=2, mode='bilinear', align_corners=False)

        # Stage 6: 128x128 (Residual addition from 64x64)
        h128 = self.block_128(h64)
        rgb_128 = self.to_rgb_128(h128) + F.interpolate(rgb_64, scale_factor=2, mode='bilinear', align_corners=False)
        
        return rgb_128, rgb_64, rgb_32

    def forward(self, z, labels, steps=2, return_all_scales=False):
        c = self.label_emb(labels)
        cond = torch.cat([z, c], dim=-1)
        w = self.mapping(cond)
        
        # Stage 1: 4x4 initial spatial seed
        h4_base = self.init_fc(w).view(z.size(0), self.base_channels, 4, 4)
        
        canvas = None
        out_128, out_64, out_32 = None, None, None
        
        for step in range(steps):
            if step == 0:
                h4 = h4_base
            else:
                # Recurrent feedback: add previous canvas back as a residual connection into the 4x4 stem
                h4 = h4_base + self.feedback_stem(canvas)
                
            rgb_128, rgb_64, rgb_32 = self._decode_from_h4(h4)
            out_128 = torch.sigmoid(rgb_128)
            canvas = out_128
            
        if return_all_scales:
            out_64 = torch.sigmoid(rgb_64)
            out_32 = torch.sigmoid(rgb_32)
            return out_128, out_64, out_32
            
        return out_128
