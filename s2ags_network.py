import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SS2D(nn.Module):
    """Selective state propagation over 2D feature maps.

    state_mode controls the actual scanning directions:
        - "none": no state propagation; return the input feature map.
        - "row": row-wise forward/backward propagation.
        - "row_col": row-wise and column-wise bidirectional propagation.
    """

    def __init__(self, d_model, state_mode="row_col"):
        super().__init__()
        if state_mode not in {"none", "row", "row_col"}:
            raise ValueError(f"Unsupported state_mode: {state_mode}. Expected one of: none, row, row_col.")

        self.d_model = d_model
        self.state_mode = state_mode

        if self.state_mode != "none":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError(
                    "mamba_ssm is required when state_mode is 'row' or 'row_col'. "
                    "Install it before enabling the spatial-state branch."
                ) from exc

            self.core_mamba = Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, C, H, W]
        if self.state_mode == "none":
            return x

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.norm(x_flat)

        outputs = []

        # Row-wise forward scan.
        out_row_fwd = self.core_mamba(x_norm)
        outputs.append(out_row_fwd)

        # Row-wise backward scan.
        seq_row_bwd = torch.flip(x_norm, dims=[1])
        out_row_bwd = torch.flip(self.core_mamba(seq_row_bwd), dims=[1])
        outputs.append(out_row_bwd)

        if self.state_mode == "row_col":
            # Column-wise forward scan.
            seq_col_fwd = x_norm.view(B, H, W, C).transpose(1, 2).reshape(B, H * W, C)
            out_col_fwd = self.core_mamba(seq_col_fwd).view(B, W, H, C).transpose(1, 2).reshape(B, H * W, C)
            outputs.append(out_col_fwd)

            # Column-wise backward scan.
            seq_col_bwd = torch.flip(seq_col_fwd, dims=[1])
            out_col_bwd = torch.flip(self.core_mamba(seq_col_bwd), dims=[1]).view(B, W, H, C).transpose(1, 2).reshape(B, H * W, C)
            outputs.append(out_col_bwd)

        out_merged = torch.stack(outputs, dim=0).mean(dim=0)
        out_2d = out_merged.transpose(1, 2).reshape(B, C, H, W)
        return out_2d + x


class S2AGSFeatureExtractor(nn.Module):
    """Feature extractor for the S²A-GS spatial-state guidance branch.

    The shared contextual feature is decoded by two task-specific heads:

      - structure_head predicts a full-foreground structural importance map;
      - highlight_head predicts photometric-reliability logits for
        high-luminance candidates.

    This gives the spatial-state branch an explicit optimization path outside
    reflective regions while retaining candidate-gated regulation of
    view-dependent residuals.  The branch does not explicitly decompose
    reflectance or illumination; both heads condition gradients received by
    the same Gaussian representation during training.
    """

    def __init__(self, use_vmamba=True, state_mode="row_col"):
        super().__init__()
        if state_mode not in {"none", "row", "row_col"}:
            raise ValueError(f"Unsupported state_mode: {state_mode}. Expected one of: none, row, row_col.")

        
        self.state_mode = state_mode if use_vmamba else "none"

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1)
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.state_block = SS2D(d_model=64, state_mode=self.state_mode)

        self.structure_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.highlight_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # ImageNet-pretrained ResNet-34 expects normalized RGB input.
        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, img):
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(
                f"Expected an RGB BCHW tensor, got shape={tuple(img.shape)}"
            )

        img = (img - self.input_mean.to(img)) / self.input_std.to(img)

        def custom_forward(x):
            c1 = self.layer1(x)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)

            p3 = self.latlayer1(c3)
            p2 = self.latlayer2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            p2 = self.smooth1(p2)
            p1 = c1 + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            p1 = self.smooth2(p1)
            return p1

        if self.training and img.requires_grad:
            p1 = checkpoint(custom_forward, img, use_reentrant=False)
        else:
            p1 = custom_forward(img)

        features = self.state_block(p1)
        structure_logits = self.structure_head(features)
        highlight_logits = self.highlight_head(features)
        return structure_logits, highlight_logits



S2GS_FeatureExtractor = S2AGSFeatureExtractor
SpatialStateBlock = SS2D
