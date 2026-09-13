"""Evaluate object-level foreground metrics for S2A-GS.

This version computes Projected Opacity Leakage (POL) from accumulated
opacity/alpha maps, matching the paper definition:

    POL = sum_{x outside M} A(x) / (sum_x A(x) + eps)

where A(x) is the accumulated opacity rendered by the Gaussian rasterizer.
The script intentionally requires alpha/opacity maps by default so that POL is
not accidentally computed from RGB luminance.
"""

from __future__ import annotations

import os
import re
import glob
import json
import argparse

import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_func


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
ALPHA_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.npy")


def list_files(directory: str, patterns: tuple[str, ...]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(files)


def get_iter_num(path: str) -> int:
    """Extract the iteration number from a path such as ours_30000."""
    match = re.search(r"ours_(\d+)", path)
    return int(match.group(1)) if match else -1


def find_latest_render_dir(model_path: str) -> str | None:
    """Find the latest renders directory under the model output directory."""
    candidates = glob.glob(os.path.join(model_path, "test", "ours_*", "renders"))
    if len(candidates) == 0:
        candidates = glob.glob(os.path.join(model_path, "test", "renders"))
    if len(candidates) == 0:
        return None
    return max(candidates, key=get_iter_num)


def find_gt_dir(render_dir: str) -> str | None:
    gt_dir = render_dir.replace("renders", "gt")
    return gt_dir if os.path.exists(gt_dir) else None


def find_alpha_dir(render_dir: str, explicit_alpha_dir: str | None = None) -> str | None:
    """Find the opacity/alpha directory corresponding to a renders directory.

    The preferred layout is:
        model_path/test/ours_30000/opacity
    but several common directory names are also supported.
    """
    if explicit_alpha_dir:
        return explicit_alpha_dir if os.path.exists(explicit_alpha_dir) else None

    parent = os.path.dirname(render_dir)
    candidate_names = [
        "opacity",
        "opacities",
        "alpha",
        "alphas",
        "alpha_maps",
        "opacity_maps",
        "accum_alpha",
        "accumulated_alpha",
        "accumulated_opacity",
    ]
    for name in candidate_names:
        candidate = os.path.join(parent, name)
        if os.path.exists(candidate):
            return candidate

    return None


def read_bgr_3ch(path: str) -> np.ndarray:
    """Read BGR, RGBA, or grayscale images and return a uint8 3-channel BGR image."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        bgr = img[:, :, :3].copy()
        alpha = img[:, :, 3]
        bgr[alpha == 0] = 0
        return bgr

    if img.shape[2] == 3:
        return img

    raise RuntimeError(f"Unsupported image format: {path}, shape={img.shape}")


def read_alpha_float(path: str) -> np.ndarray:
    """Read accumulated opacity/alpha as a float32 2D array in [0, 1].

    Supported formats:
      - .npy files containing HxW or 1xHxW arrays;
      - grayscale PNG/TIFF/JPEG/BMP images;
      - RGBA images, where the alpha channel is used.
    """
    if path.lower().endswith(".npy"):
        alpha = np.load(path)
        alpha = np.asarray(alpha)
        alpha = np.squeeze(alpha)
        if alpha.ndim != 2:
            raise RuntimeError(f"Alpha npy must be 2D after squeeze: {path}, shape={alpha.shape}")
        alpha = alpha.astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read alpha/opacity map: {path}")
        if img.ndim == 2:
            alpha = img.astype(np.float32)
        elif img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32)
        elif img.ndim == 3 and img.shape[2] >= 3:
            # Some renderers save alpha replicated to RGB channels.
            alpha = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported alpha map format: {path}, shape={img.shape}")

    max_val = float(np.max(alpha)) if alpha.size else 0.0
    if max_val > 1.5:
        if max_val > 4096:
            alpha = alpha / 65535.0
        else:
            alpha = alpha / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha.astype(np.float32)


def get_binary_mask(mask_img: np.ndarray) -> np.ndarray:
    return (mask_img > 127).astype(np.uint8)


def create_aligned_test_masks(all_mask_dir: str, render_dir: str, out_mask_dir: str, hold: int = 8) -> str:
    """Create masks aligned with the test-set renders."""
    os.makedirs(out_mask_dir, exist_ok=True)

    all_masks = list_files(all_mask_dir, IMAGE_EXTS)
    render_imgs = sorted(glob.glob(os.path.join(render_dir, "*.png")))

    if len(all_masks) == 0:
        raise FileNotFoundError(f"No masks found: {all_mask_dir}")
    if len(render_imgs) == 0:
        raise FileNotFoundError(f"No renders found: {render_dir}")

    if len(all_masks) == len(render_imgs):
        selected_masks = all_masks
        print("[INFO] Mask count matches render count; using masks as already aligned.")
    else:
        selected_masks = all_masks[::hold]
        print("[INFO] Detected full mask set; selecting test masks using the hold interval.")
        print(f"   all masks: {len(all_masks)}")
        print(f"   hold={hold} after hold selection: {len(selected_masks)}")
        print(f"   renders: {len(render_imgs)}")

    if len(selected_masks) != len(render_imgs):
        print("[WARN] Mask count differs from render count; aligning with the smaller count.")
        print(f"   selected masks: {len(selected_masks)}")
        print(f"   renders:        {len(render_imgs)}")

    n = min(len(selected_masks), len(render_imgs))
    for idx in range(n):
        mask = cv2.imread(selected_masks[idx], cv2.IMREAD_GRAYSCALE)
        render = cv2.imread(render_imgs[idx], cv2.IMREAD_COLOR)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {selected_masks[idx]}")
        if render is None:
            raise RuntimeError(f"Failed to read render: {render_imgs[idx]}")
        h, w = render.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_mask_dir, f"{idx:05d}.png"), mask)

    return out_mask_dir


def get_mask_bbox(mask: np.ndarray, padding: int = 8):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, w)
    y2 = min(int(ys.max()) + padding + 1, h)
    return x1, y1, x2, y2


def apply_mask_black(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[mask == 0] = 0
    return out


def crop_by_bbox(img: np.ndarray, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


def calculate_fg_psnr(render_img: np.ndarray, gt_img: np.ndarray, gt_mask: np.ndarray) -> float:
    fg = gt_mask.astype(bool)
    if fg.sum() == 0:
        return 0.0
    diff = render_img.astype(np.float32) - gt_img.astype(np.float32)
    diff_fg = diff[fg]
    mse = np.mean(diff_fg ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def calculate_fg_ssim(render_img: np.ndarray, gt_img: np.ndarray, gt_mask: np.ndarray, padding: int = 8) -> float:
    bbox = get_mask_bbox(gt_mask, padding=padding)
    if bbox is None:
        return 0.0
    render_crop = crop_by_bbox(apply_mask_black(render_img, gt_mask), bbox)
    gt_crop = crop_by_bbox(apply_mask_black(gt_img, gt_mask), bbox)
    min_hw = min(render_crop.shape[:2])
    if min_hw < 3:
        return 0.0
    win_size = min(7, min_hw)
    if win_size % 2 == 0:
        win_size -= 1
    return float(ssim_func(render_crop, gt_crop, channel_axis=2, data_range=255, win_size=win_size))


def resize_max_side(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    if max_side is None or max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_lpips_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2.0 - 1.0
    return img.to(device)


def calculate_fg_lpips(
    render_img: np.ndarray,
    gt_img: np.ndarray,
    gt_mask: np.ndarray,
    loss_fn,
    device: torch.device,
    padding: int = 8,
    lpips_max_side: int = 1024,
) -> float:
    bbox = get_mask_bbox(gt_mask, padding=padding)
    if bbox is None:
        return 0.0
    render_crop = crop_by_bbox(apply_mask_black(render_img, gt_mask), bbox)
    gt_crop = crop_by_bbox(apply_mask_black(gt_img, gt_mask), bbox)
    render_crop = resize_max_side(render_crop, max_side=lpips_max_side)
    gt_crop = resize_max_side(gt_crop, max_side=lpips_max_side)
    with torch.no_grad():
        value = loss_fn(image_to_lpips_tensor(render_crop, device), image_to_lpips_tensor(gt_crop, device)).item()
    return float(value)


def get_alpha_silhouette(alpha_map: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Extract the predicted foreground silhouette from accumulated opacity."""
    return (alpha_map > threshold).astype(np.uint8)


def get_render_silhouette(render_img: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Legacy RGB-based silhouette extraction, kept only as an optional fallback."""
    gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
    return (gray > threshold).astype(np.uint8)


def calculate_silhouette_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_projected_opacity_leakage(alpha_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Projected Opacity Leakage from accumulated opacity.

    POL = sum_{x outside M} alpha(x) / (sum_x alpha(x) + eps)
    """
    if alpha_map.ndim != 2:
        raise ValueError(f"alpha_map must be 2D, got shape={alpha_map.shape}")
    gt = gt_mask.astype(bool)
    outside = np.logical_not(gt)
    total_opacity = float(alpha_map.sum()) + 1e-8
    outside_opacity = float(alpha_map[outside].sum())
    return outside_opacity / total_opacity


def match_alpha_paths(alpha_dir: str, render_paths: list[str]) -> list[str]:
    """Match alpha maps to render file names whenever possible."""
    alpha_files = list_files(alpha_dir, ALPHA_EXTS)
    if len(alpha_files) == 0:
        raise FileNotFoundError(f"No alpha/opacity maps found in: {alpha_dir}")

    # Match by stem first, e.g., 00000.png -> 00000.png / 00000.npy.
    stem_to_alpha: dict[str, str] = {}
    for path in alpha_files:
        stem_to_alpha[os.path.splitext(os.path.basename(path))[0]] = path

    matched: list[str] = []
    missing: list[str] = []
    for render_path in render_paths:
        stem = os.path.splitext(os.path.basename(render_path))[0]
        alpha_path = stem_to_alpha.get(stem)
        if alpha_path is None:
            missing.append(os.path.basename(render_path))
        else:
            matched.append(alpha_path)

    if len(missing) == 0:
        return matched

    if len(alpha_files) == len(render_paths):
        print("[WARN] Alpha file names do not match renders; using sorted alpha files by index.")
        return alpha_files

    preview = ", ".join(missing[:10])
    raise FileNotFoundError(
        f"Cannot match alpha/opacity maps to renders. Missing examples: {preview}. "
        f"alpha_dir={alpha_dir}"
    )


def evaluate_fg5_metrics(
    model_path: str,
    original_mask_dir: str | None = None,
    alpha_dir: str | None = None,
    hold: int = 8,
    bbox_padding: int = 8,
    lpips_max_side: int = 1024,
    silhouette_threshold: int = 5,
    alpha_threshold: float = 0.01,
    use_alpha_silhouette: bool = True,
    allow_luminance_fallback: bool = False,
):
    """Compute the five object-level evaluation metrics.

    Projected_Opacity_Leakage is computed from accumulated alpha/opacity maps.
    By default, missing alpha maps raise an error to avoid reporting RGB-based
    leakage under an opacity-based name.
    """
    render_dir = find_latest_render_dir(model_path)
    if render_dir is None:
        raise FileNotFoundError(f"No renders directory found: {model_path}")

    gt_dir = find_gt_dir(render_dir)
    if gt_dir is None:
        raise FileNotFoundError(f"No gt directory found: {render_dir.replace('renders', 'gt')}")

    mask_dir = render_dir.replace("renders", "masks")
    if not os.path.exists(mask_dir):
        if original_mask_dir is None:
            raise FileNotFoundError(
                f"No aligned masks found: {mask_dir}\n"
                f"--mask_dir was not provided; object-level metrics cannot be computed."
            )
        print("[INFO] No aligned masks found; generating test masks from --mask_dir.")
        mask_dir = create_aligned_test_masks(original_mask_dir, render_dir, mask_dir, hold=hold)

    found_alpha_dir = find_alpha_dir(render_dir, explicit_alpha_dir=alpha_dir)
    if found_alpha_dir is None and not allow_luminance_fallback:
        raise FileNotFoundError(
            "Projected Opacity Leakage requires accumulated alpha/opacity maps, but none were found.\n"
            "Expected a sibling directory such as test/ours_30000/opacity, or pass --alpha_dir.\n"
            "If you intentionally use the RGB-luminance proxy, pass "
            "--allow_luminance_fallback. Note that this is not equivalent to "
            "Projected Opacity Leakage computed from accumulated opacity."
        )

    render_paths = sorted(glob.glob(os.path.join(render_dir, "*.png")))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    mask_paths = list_files(mask_dir, IMAGE_EXTS)
    alpha_paths = match_alpha_paths(found_alpha_dir, render_paths) if found_alpha_dir is not None else []

    n = min(len(render_paths), len(gt_paths), len(mask_paths), len(alpha_paths) if alpha_paths else len(render_paths))
    if n == 0:
        raise RuntimeError("renders / gt / masks / alpha count is 0; evaluation cannot proceed.")
    if found_alpha_dir is not None and not (len(render_paths) == len(gt_paths) == len(mask_paths) == len(alpha_paths)):
        print("[WARN] renders / gt / masks / alpha counts do not match; using the minimum count:")
        print(f"   renders: {len(render_paths)}")
        print(f"   gt:      {len(gt_paths)}")
        print(f"   masks:   {len(mask_paths)}")
        print(f"   alpha:   {len(alpha_paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
    loss_fn_vgg.eval()

    metrics = {
        "FG_PSNR": [],
        "FG_SSIM": [],
        "FG_LPIPS": [],
        "Silhouette_IoU": [],
        "Projected_Opacity_Leakage": [],
    }

    for idx in range(n):
        render = read_bgr_3ch(render_paths[idx])
        gt = read_bgr_3ch(gt_paths[idx])
        mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_paths[idx]}")

        h, w = render.shape[:2]
        if gt.shape[:2] != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        gt_mask = get_binary_mask(mask)

        if found_alpha_dir is not None:
            alpha_map = read_alpha_float(alpha_paths[idx])
            if alpha_map.shape[:2] != (h, w):
                alpha_map = cv2.resize(alpha_map, (w, h), interpolation=cv2.INTER_AREA)
            alpha_map = np.clip(alpha_map, 0.0, 1.0)
        else:
            # Optional RGB-luminance fallback. This is not equivalent to opacity-based POL.
            gray = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            alpha_map = gray

        fg_psnr = calculate_fg_psnr(render, gt, gt_mask)
        fg_ssim = calculate_fg_ssim(render, gt, gt_mask, padding=bbox_padding)
        fg_lpips = calculate_fg_lpips(render, gt, gt_mask, loss_fn_vgg, device, padding=bbox_padding, lpips_max_side=lpips_max_side)

        if use_alpha_silhouette:
            pred_mask = get_alpha_silhouette(alpha_map, threshold=alpha_threshold)
        else:
            pred_mask = get_render_silhouette(render, threshold=silhouette_threshold)
        sil_iou = calculate_silhouette_iou(pred_mask, gt_mask)
        leakage = calculate_projected_opacity_leakage(alpha_map, gt_mask)

        metrics["FG_PSNR"].append(fg_psnr)
        metrics["FG_SSIM"].append(fg_ssim)
        metrics["FG_LPIPS"].append(fg_lpips)
        metrics["Silhouette_IoU"].append(sil_iou)
        metrics["Projected_Opacity_Leakage"].append(leakage)

    return {
        "model_path": model_path,
        "render_dir": render_dir,
        "gt_dir": gt_dir,
        "mask_dir": mask_dir,
        "alpha_dir": found_alpha_dir,
        "num_images": n,
        "FG_PSNR": float(np.mean(metrics["FG_PSNR"])),
        "FG_SSIM": float(np.mean(metrics["FG_SSIM"])),
        "FG_LPIPS": float(np.mean(metrics["FG_LPIPS"])),
        "Silhouette_IoU": float(np.mean(metrics["Silhouette_IoU"])),
        "Projected_Opacity_Leakage": float(np.mean(metrics["Projected_Opacity_Leakage"])),
        "bbox_padding": bbox_padding,
        "lpips_max_side": lpips_max_side,
        "silhouette_source": "alpha" if use_alpha_silhouette else "render_luminance",
        "alpha_threshold": alpha_threshold,
        "silhouette_threshold": silhouette_threshold,
        "allow_luminance_fallback": bool(allow_luminance_fallback),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Model output directory.")
    parser.add_argument("--mask_dir", default=None, help="Directory containing all masks or aligned test masks.")
    parser.add_argument("--alpha_dir", default=None, help="Directory containing accumulated opacity/alpha maps.")
    parser.add_argument("--hold", type=int, default=8, help="LLFF hold interval for the test split.")
    parser.add_argument("--bbox_padding", type=int, default=8)
    parser.add_argument("--lpips_max_side", type=int, default=1024)
    parser.add_argument("--silhouette_threshold", type=int, default=5, help="Legacy RGB silhouette threshold.")
    parser.add_argument("--alpha_threshold", type=float, default=0.01, help="Alpha threshold for Silhouette IoU.")
    parser.add_argument("--use_rgb_silhouette", action="store_true", help="Use legacy RGB-threshold silhouette instead of alpha silhouette.")
    parser.add_argument("--allow_luminance_fallback", action="store_true", help="Legacy fallback only. Do not use for POL results in the paper.")
    parser.add_argument("--save_json", default=None, help="Path for saving the result JSON.")
    args = parser.parse_args()

    result = evaluate_fg5_metrics(
        model_path=args.model_path,
        original_mask_dir=args.mask_dir,
        alpha_dir=args.alpha_dir,
        hold=args.hold,
        bbox_padding=args.bbox_padding,
        lpips_max_side=args.lpips_max_side,
        silhouette_threshold=args.silhouette_threshold,
        alpha_threshold=args.alpha_threshold,
        use_alpha_silhouette=not args.use_rgb_silhouette,
        allow_luminance_fallback=args.allow_luminance_fallback,
    )

    print("\n================ FG-5 Metrics Result ================")
    print(f"Model: {result['model_path']}")
    print(f"Images: {result['num_images']}")
    print(f"Render dir: {result['render_dir']}")
    print(f"GT dir:     {result['gt_dir']}")
    print(f"Mask dir:   {result['mask_dir']}")
    print(f"Alpha dir:  {result['alpha_dir']}")
    print(f"Silhouette source: {result['silhouette_source']}")
    print(f"FG-PSNR(↑): {result['FG_PSNR']:.4f}")
    print(f"FG-SSIM(↑): {result['FG_SSIM']:.4f}")
    print(f"FG-LPIPS(↓): {result['FG_LPIPS']:.4f}")
    print(f"Silhouette IoU(↑): {result['Silhouette_IoU']:.4f}")
    print(f"Projected Opacity Leakage(↓): {result['Projected_Opacity_Leakage']:.4f}")
    print("=====================================================\n")

    if args.save_json is not None:
        save_dir = os.path.dirname(args.save_json)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
