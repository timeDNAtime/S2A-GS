"""Loss utilities for the S²A-GS training objective.

The spatial-state branch has two explicit optimization paths:

1. A full-foreground structural path.  A learned structure score is trained
   from non-parametric weak-texture, detail, and silhouette-boundary targets.
   The score emphasizes a Charbonnier-plus-gradient residual over the complete
   foreground, so the state branch remains active outside highlight regions.
2. A reflective-region path.  A learned reliability value regulates the
   photometric contribution of high-luminance foreground candidates.  This
   does not estimate a separate reflection layer; it changes the residual
   gradients received by the shared Gaussian appearance, opacity, spatial
   parameters, and densification process.

All functions operate on tensors in the [0, 1] range.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_bchw(tensor: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    """Return a BCHW view and validate the optional channel count."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape={tuple(tensor.shape)}")
    if channels is not None and tensor.shape[1] != channels:
        raise ValueError(
            f"Expected {channels} channels, got shape={tuple(tensor.shape)}"
        )
    return tensor


def _resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    mask = _as_bchw(mask)
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask.float(), size=size, mode="nearest")
    return (mask > 0.5).to(dtype=torch.float32)


def rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB CHW/BCHW input to one-channel BCHW luminance."""
    image = _as_bchw(image, channels=3)
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return torch.sum(image * weights, dim=1, keepdim=True)


def sobel_components(luminance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized Sobel x/y responses for a one-channel image."""
    luminance = _as_bchw(luminance, channels=1)
    kernel_x = luminance.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)


def _masked_quantile(
    value_map: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    values = value_map[valid_mask > 0.5]
    if values.numel() == 0:
        return value_map.new_tensor(0.0)
    return torch.quantile(values.detach(), quantile)


@torch.no_grad()
def build_structure_target(
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    local_window: int = 7,
    low_texture_quantile: float = 0.30,
    detail_quantile: float = 0.70,
    boundary_radius: int = 2,
) -> torch.Tensor:
    """Build a reproducible full-foreground structural supervision target.

    The target is the union of:
      - low-texture pixels selected from local luminance standard deviation;
      - image-detail pixels selected from Sobel magnitude;
      - the inner silhouette boundary obtained from the foreground mask.

    Quantiles are computed independently for each image over valid foreground
    pixels.  The returned tensor has shape [B, 1, H, W] and values in {0, 1}.
    """
    if local_window < 3 or local_window % 2 == 0:
        raise ValueError("local_window must be an odd integer >= 3")
    if not 0.0 <= low_texture_quantile <= 1.0:
        raise ValueError("low_texture_quantile must be in [0, 1]")
    if not 0.0 <= detail_quantile <= 1.0:
        raise ValueError("detail_quantile must be in [0, 1]")
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be >= 0")

    gt_image = _as_bchw(gt_image, channels=3)
    valid_mask = _resize_mask(valid_mask, gt_image.shape[-2:]).to(
        device=gt_image.device, dtype=gt_image.dtype
    )
    luminance = rgb_to_luminance(gt_image)

    padding = local_window // 2
    local_mean = F.avg_pool2d(
        luminance, kernel_size=local_window, stride=1, padding=padding
    )
    local_second_moment = F.avg_pool2d(
        luminance.square(), kernel_size=local_window, stride=1, padding=padding
    )
    local_std = torch.sqrt(
        torch.clamp(local_second_moment - local_mean.square(), min=0.0) + 1e-8
    )

    grad_x, grad_y = sobel_components(luminance)
    gradient_magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)

    targets = []
    for batch_index in range(gt_image.shape[0]):
        mask_b = valid_mask[batch_index : batch_index + 1]
        std_b = local_std[batch_index : batch_index + 1]
        grad_b = gradient_magnitude[batch_index : batch_index + 1]

        low_threshold = _masked_quantile(
            std_b, mask_b, low_texture_quantile
        )
        detail_threshold = _masked_quantile(
            grad_b, mask_b, detail_quantile
        )
        low_texture = (std_b <= low_threshold).to(gt_image.dtype)
        local_detail = (grad_b >= detail_threshold).to(gt_image.dtype)

        if boundary_radius > 0:
            kernel_size = 2 * boundary_radius + 1
            eroded = -F.max_pool2d(
                -mask_b,
                kernel_size=kernel_size,
                stride=1,
                padding=boundary_radius,
            )
            inner_boundary = (mask_b - eroded).clamp(0.0, 1.0)
        else:
            inner_boundary = torch.zeros_like(mask_b)

        target_b = torch.maximum(
            torch.maximum(low_texture, local_detail), inner_boundary
        )
        targets.append(target_b * mask_b)

    return torch.cat(targets, dim=0)


@torch.no_grad()
def build_highlight_candidate(
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    quantile: float = 0.97,
) -> torch.Tensor:
    """Select high-luminance foreground candidates per image."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    gt_image = _as_bchw(gt_image, channels=3)
    valid_mask = _resize_mask(valid_mask, gt_image.shape[-2:]).to(
        device=gt_image.device, dtype=gt_image.dtype
    )
    luminance = rgb_to_luminance(gt_image)

    candidates = []
    for batch_index in range(gt_image.shape[0]):
        mask_b = valid_mask[batch_index : batch_index + 1]
        luminance_b = luminance[batch_index : batch_index + 1]
        threshold = _masked_quantile(luminance_b, mask_b, quantile)
        candidate_b = (luminance_b >= threshold).to(gt_image.dtype) * mask_b
        candidates.append(candidate_b)
    return torch.cat(candidates, dim=0)


def build_highlight_photometric_weight(
    valid_mask: torch.Tensor,
    highlight_candidate: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Apply the paper's candidate-gated photometric weighting rule.

    The returned weight is zero outside the valid foreground, exactly one in
    foreground pixels outside the high-luminance candidate set, and equal to
    the state-conditioned confidence inside the candidate set:

        w = M * (1 - H * (1 - C)).

    Keeping this rule in one named function makes the implementation of
    Eq. (15) explicit and ensures consistent candidate gating.
    """
    valid_mask = _as_bchw(valid_mask, channels=1)
    highlight_candidate = _as_bchw(highlight_candidate, channels=1)
    confidence = _as_bchw(confidence, channels=1)
    if not (
        valid_mask.shape == highlight_candidate.shape == confidence.shape
    ):
        raise ValueError(
            "valid_mask, highlight_candidate, and confidence must have "
            f"identical BCHW shapes, got {tuple(valid_mask.shape)}, "
            f"{tuple(highlight_candidate.shape)}, and {tuple(confidence.shape)}"
        )
    return valid_mask * (
        1.0 - highlight_candidate * (1.0 - confidence)
    )


def _resize_guidance_map(
    guidance: torch.Tensor,
    size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resize a one-channel guidance map to the Gaussian rendering resolution."""
    guidance = _as_bchw(guidance)
    if guidance.shape[1] != 1:
        guidance = torch.mean(guidance, dim=1, keepdim=True)
    guidance = guidance.to(device=device, dtype=dtype)
    if guidance.shape[-2:] != size:
        guidance = F.interpolate(
            guidance, size=size, mode="bilinear", align_corners=False
        )
    return guidance


def state_guided_structure_loss_from_score(
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    structure_score: torch.Tensor,
    epsilon: float = 1e-3,
    gradient_weight: float = 0.5,
    emphasis: float = 1.0,
) -> torch.Tensor:
    """Eq. (18) using a supplied structural-importance map.

    This helper is used both at guidance-refresh steps and at intermediate
    cached-guidance steps.  The supplied score is treated as a contextual
    weight; callers control whether it is live or detached.
    """
    rendered_image = _as_bchw(rendered_image, channels=3)
    gt_image = _as_bchw(gt_image, channels=3)
    size = rendered_image.shape[-2:]
    if gt_image.shape[-2:] != size:
        raise ValueError("rendered_image and gt_image must have the same size")

    valid_mask = _resize_mask(valid_mask, size).to(
        device=rendered_image.device, dtype=rendered_image.dtype
    )
    structure_score = _resize_guidance_map(
        structure_score,
        size,
        device=rendered_image.device,
        dtype=rendered_image.dtype,
    )

    pixel_residual = torch.mean(
        torch.sqrt((rendered_image - gt_image).square() + epsilon**2),
        dim=1,
        keepdim=True,
    )
    rendered_luminance = rgb_to_luminance(rendered_image)
    gt_luminance = rgb_to_luminance(gt_image)
    render_dx, render_dy = sobel_components(rendered_luminance)
    gt_dx, gt_dy = sobel_components(gt_luminance)
    gradient_residual = 0.5 * (
        torch.abs(render_dx - gt_dx) + torch.abs(render_dy - gt_dy)
    )

    structural_residual = pixel_residual + gradient_weight * gradient_residual
    structural_weight = valid_mask * (1.0 + emphasis * structure_score)
    return torch.sum(structural_weight * structural_residual) / (
        torch.sum(structural_weight) + 1e-6
    )


def structure_target_bce_loss(
    structure_logits: torch.Tensor,
    structure_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Eq. (14): foreground-normalized BCE at the reference-image resolution."""
    structure_logits = _as_bchw(structure_logits)
    if structure_logits.shape[1] != 1:
        raise ValueError("structure_logits must have one channel")
    structure_target = _as_bchw(structure_target, channels=1)
    size = structure_target.shape[-2:]
    valid_mask = _resize_mask(valid_mask, size).to(
        device=structure_logits.device, dtype=structure_logits.dtype
    )
    structure_target = _resize_mask(structure_target, size).to(
        device=structure_logits.device, dtype=structure_logits.dtype
    )
    logits_full = structure_logits
    if logits_full.shape[-2:] != size:
        logits_full = F.interpolate(
            logits_full, size=size, mode="bilinear", align_corners=False
        )
    loss_map = F.binary_cross_entropy_with_logits(
        logits_full, structure_target, reduction="none"
    )
    return torch.sum(loss_map * valid_mask) / (torch.sum(valid_mask) + 1e-6)


def state_guided_structure_losses(
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    structure_logits: torch.Tensor,
    structure_target: torch.Tensor,
    epsilon: float = 1e-3,
    gradient_weight: float = 0.5,
    emphasis: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Eqs. (14) and (18) at a guidance-refresh step.

    Eq. (18) applies stop-gradient to the structural score.
    The structure head itself is optimized through Eq. (14).
    The returned score is kept at the branch's native resolution so it can be
    detached and stored in the per-view guidance cache.
    """
    structure_logits = _as_bchw(structure_logits)
    if structure_logits.shape[1] != 1:
        raise ValueError("structure_logits must have one channel")

    structure_score = torch.sigmoid(structure_logits)
    target_map_loss = structure_target_bce_loss(
        structure_logits,
        structure_target,
        valid_mask,
    )
    guidance_loss = state_guided_structure_loss_from_score(
        rendered_image,
        gt_image,
        valid_mask,
        structure_score.detach(),
        epsilon=epsilon,
        gradient_weight=gradient_weight,
        emphasis=emphasis,
    )
    return guidance_loss, target_map_loss, structure_score


def highlight_weighted_charbonnier_from_confidence(
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    highlight_candidate: torch.Tensor,
    confidence: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Eq. (19) using a supplied contextual reliability map."""
    rendered_image = _as_bchw(rendered_image, channels=3)
    gt_image = _as_bchw(gt_image, channels=3)
    size = rendered_image.shape[-2:]
    valid_mask = _resize_mask(valid_mask, size).to(
        device=rendered_image.device, dtype=rendered_image.dtype
    )
    highlight_candidate = _resize_mask(highlight_candidate, size).to(
        device=rendered_image.device, dtype=rendered_image.dtype
    )
    confidence = _resize_guidance_map(
        confidence,
        size,
        device=rendered_image.device,
        dtype=rendered_image.dtype,
    )

    photometric_weight = build_highlight_photometric_weight(
        valid_mask,
        highlight_candidate,
        confidence,
    )
    pixel_residual = torch.mean(
        torch.sqrt((rendered_image - gt_image).square() + epsilon**2),
        dim=1,
        keepdim=True,
    )
    return torch.sum(photometric_weight * pixel_residual) / (
        torch.sum(photometric_weight) + 1e-6
    )


def highlight_confidence_regularizer(
    highlight_candidate: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """Eq. (20): discourage indiscriminate attenuation in candidates."""
    highlight_candidate = _as_bchw(highlight_candidate, channels=1)
    confidence = _resize_guidance_map(
        confidence,
        highlight_candidate.shape[-2:],
        device=highlight_candidate.device,
        dtype=highlight_candidate.dtype,
    )
    return torch.sum(highlight_candidate * (1.0 - confidence).square()) / (
        torch.sum(highlight_candidate) + 1e-6
    )


def highlight_weighted_charbonnier_loss(
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    valid_mask: torch.Tensor,
    highlight_candidate: torch.Tensor,
    highlight_logits: torch.Tensor | None,
    confidence_floor: float = 0.5,
    fixed_confidence: float = 0.75,
    epsilon: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Eqs. (15), (19), and (20) at a refresh/rule-only step.

    For the full model, the returned confidence remains at the branch's native
    resolution and can be detached into the per-view guidance cache.  For the
    w/o S²A-State ablation, ``highlight_logits`` is ``None`` and a fixed rule-
    only reliability is used inside the high-luminance candidate set.
    """
    if not 0.0 < confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in (0, 1]")
    if not 0.0 < fixed_confidence <= 1.0:
        raise ValueError("fixed_confidence must be in (0, 1]")

    rendered_image = _as_bchw(rendered_image, channels=3)
    gt_image = _as_bchw(gt_image, channels=3)
    size = rendered_image.shape[-2:]
    valid_mask_full = _resize_mask(valid_mask, size).to(
        device=rendered_image.device, dtype=rendered_image.dtype
    )
    candidate_full = _resize_mask(highlight_candidate, size).to(
        device=rendered_image.device, dtype=rendered_image.dtype
    )

    if highlight_logits is not None:
        highlight_logits = _as_bchw(highlight_logits)
        if highlight_logits.shape[1] != 1:
            highlight_logits = torch.mean(highlight_logits, dim=1, keepdim=True)
        confidence = confidence_floor + (1.0 - confidence_floor) * torch.sigmoid(
            highlight_logits
        )
        confidence_full = _resize_guidance_map(
            confidence,
            size,
            device=rendered_image.device,
            dtype=rendered_image.dtype,
        )
        confidence_regularizer = highlight_confidence_regularizer(
            candidate_full,
            confidence_full,
        )
    else:
        confidence = torch.full_like(candidate_full, fixed_confidence)
        confidence_regularizer = rendered_image.new_zeros(())

    weighted_loss = highlight_weighted_charbonnier_from_confidence(
        rendered_image,
        gt_image,
        valid_mask_full,
        candidate_full,
        confidence,
        epsilon=epsilon,
    )
    return weighted_loss, confidence_regularizer, confidence
