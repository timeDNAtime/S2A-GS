#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import time
import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim
from s2ags_losses import (
    build_highlight_candidate,
    build_structure_target,
    highlight_weighted_charbonnier_loss,
    highlight_weighted_charbonnier_from_confidence,
    state_guided_structure_losses,
    state_guided_structure_loss_from_score,
)
try:
    from pytorch_msssim import ms_ssim
    MS_SSIM_AVAILABLE = True
except ImportError:
    MS_SSIM_AVAILABLE = False
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


_APPROXIMATE_MASK_WARNING_EMITTED = False


def _load_mask_from_directory(mask_dir, image_name, target_size, cache):
    """Load an exact foreground mask by camera image name.

    Masks are cached as CPU uint8 tensors to avoid repeated disk decoding over
    long training runs.  The function accepts camera names with or without an
    extension and searches PNG/JPEG variants.
    """
    image_basename = Path(str(image_name)).name
    image_stem = Path(image_basename).stem
    cache_key = (os.path.abspath(mask_dir), image_stem)

    if cache_key not in cache:
        candidates = [
            os.path.join(mask_dir, image_basename),
            os.path.join(mask_dir, f"{image_stem}.png"),
            os.path.join(mask_dir, f"{image_stem}.jpg"),
            os.path.join(mask_dir, f"{image_stem}.jpeg"),
        ]
        mask_path = next((path for path in candidates if os.path.isfile(path)), None)
        if mask_path is None:
            raise FileNotFoundError(
                f"No foreground mask matches image '{image_name}' in: {mask_dir}"
            )

        with Image.open(mask_path) as mask_image:
            mask_array = np.asarray(mask_image.convert("L"), dtype=np.uint8).copy()
        cache[cache_key] = torch.from_numpy(mask_array).unsqueeze(0)

    mask = cache[cache_key].to(device="cuda", dtype=torch.float32) / 255.0
    if mask.shape[-2:] != target_size:
        mask = F.interpolate(
            mask.unsqueeze(0), size=target_size, mode="nearest"
        ).squeeze(0)
    return (mask > 0.5).float()


def get_valid_foreground_mask(viewpoint_cam, gt_image, mask_dir, cache):
    """Resolve an exact mask, preferring an explicit directory or camera alpha."""
    global _APPROXIMATE_MASK_WARNING_EMITTED

    if mask_dir:
        return _load_mask_from_directory(
            mask_dir,
            viewpoint_cam.image_name,
            gt_image.shape[-2:],
            cache,
        )

    if getattr(viewpoint_cam, "alpha_mask", None) is not None:
        mask = viewpoint_cam.alpha_mask.cuda().float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != gt_image.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0), size=gt_image.shape[-2:], mode="nearest"
            ).squeeze(0)
        return (mask > 0.5).float()

    if not _APPROXIMATE_MASK_WARNING_EMITTED:
        warnings.warn(
            "No --foreground_mask_dir or camera alpha mask was provided. "
            "Approximating the foreground from non-black RGB pixels; exact "
            "masks are strongly recommended for reproducible S²A-GS training.",
            RuntimeWarning,
        )
        _APPROXIMATE_MASK_WARNING_EMITTED = True

    mask = (torch.mean(gt_image, dim=0, keepdim=True) > 1e-3).float()
    if torch.sum(mask) < 100:
        mask = torch.ones_like(mask)
    return mask



def _view_cache_key(viewpoint_cam) -> str:
    """Return a stable per-view key for deterministic and guidance caches."""
    return str(Path(str(viewpoint_cam.image_name)).name)


def _pack_binary_prior(tensor: torch.Tensor) -> torch.Tensor:
    """Store a binary prior compactly on CPU."""
    tensor = tensor.detach()
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return (tensor > 0.5).to(device="cpu", dtype=torch.uint8).contiguous()


def _unpack_binary_prior(tensor: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Restore a cached binary prior as BCHW float32."""
    restored = tensor.to(device=device, dtype=torch.float32)
    if restored.dim() == 3:
        restored = restored.unsqueeze(0)
    return restored


def _precompute_deterministic_priors(
    cameras,
    mask_dir,
    mask_cache,
    *,
    need_structure: bool,
    need_highlight: bool,
    structure_local_window: int,
    structure_low_texture_quantile: float,
    structure_detail_quantile: float,
    structure_boundary_radius: int,
    highlight_quantile: float,
):
    """Build the deterministic per-view priors once before optimization.

    The manuscript defines these image-space quantities as deterministic.  The
    implementation therefore evaluates their final structural-target and
    high-luminance candidate maps once per training view and reuses them for all
    subsequent visits.  The cached binary maps are kept on CPU to avoid
    unnecessary GPU memory growth.
    """
    if not (need_structure or need_highlight):
        return {}

    cache = {}
    print("[S²A-GS] Precomputing deterministic image-space priors per training view...")
    with torch.no_grad():
        for cam in tqdm(cameras, desc="Precompute priors", leave=False):
            gt_image = cam.original_image.cuda()
            valid_mask = get_valid_foreground_mask(
                cam, gt_image, mask_dir, mask_cache
            )
            gt_tensor = gt_image.unsqueeze(0)
            mask_tensor = valid_mask.unsqueeze(0)
            item = {}
            if need_structure:
                structure_target = build_structure_target(
                    gt_tensor,
                    mask_tensor,
                    local_window=structure_local_window,
                    low_texture_quantile=structure_low_texture_quantile,
                    detail_quantile=structure_detail_quantile,
                    boundary_radius=structure_boundary_radius,
                )
                item["structure_target"] = _pack_binary_prior(structure_target)
            if need_highlight:
                highlight_candidate = build_highlight_candidate(
                    gt_tensor,
                    mask_tensor,
                    quantile=highlight_quantile,
                )
                item["highlight_candidate"] = _pack_binary_prior(highlight_candidate)
            cache[_view_cache_key(cam)] = item
    torch.cuda.empty_cache()
    print(f"[S²A-GS] Cached deterministic priors for {len(cache)} training views.")
    return cache


def _write_training_runtime(model_path: str, payload: dict) -> None:
    """Persist end-to-end optimization timing used by the paper efficiency report."""
    try:
        with open(os.path.join(model_path, "training_runtime.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        warnings.warn(f"Failed to write training_runtime.json: {exc}", RuntimeWarning)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # Measure end-to-end optimization time from Gaussian initialization
    # through the final model save.
    training_wall_start = time.perf_counter()
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # ==========================================
    # S²A-GS spatial-state guidance branch
    # ==========================================
    if args.use_state_branch:
        from s2ags_network import S2AGSFeatureExtractor
        import torch.optim as optim

        print(f"[S²A-GS] Enable spatial-state branch: state_mode={args.state_mode}")
        s2ags_net = S2AGSFeatureExtractor(
            use_vmamba=(args.state_mode != "none"),
            state_mode=args.state_mode,
        ).cuda()
        s2ags_optimizer = optim.Adam(s2ags_net.parameters(), lr=args.state_lr)
    elif args.use_rule_highlight:
        # Main ablation: remove the complete spatial-state branch while keeping
        # fixed reliability inside the rule-based high-luminance candidates.
        print("[Ablation] Disable spatial-state branch; use fixed reflective-region residual regulation only.")
        s2ags_net = None
        s2ags_optimizer = None
    else:
        print("[Baseline] Disable spatial-state branch.")
        s2ags_net = None
        s2ags_optimizer = None

    gaussians.training_setup(opt)
    if checkpoint:
        checkpoint_payload = torch.load(checkpoint)
        if isinstance(checkpoint_payload, dict):
            gaussians.restore(checkpoint_payload["gaussians"], opt)
            first_iter = int(checkpoint_payload["iteration"])
            if s2ags_net is not None and checkpoint_payload.get("state_branch") is not None:
                s2ags_net.load_state_dict(checkpoint_payload["state_branch"])
            if (
                s2ags_optimizer is not None
                and checkpoint_payload.get("state_optimizer") is not None
            ):
                s2ags_optimizer.load_state_dict(checkpoint_payload["state_optimizer"])
        else:
            # Backward compatibility with original 3DGS tuple checkpoints.
            model_params, first_iter = checkpoint_payload
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    all_train_cameras = scene.getTrainCameras().copy()
    viewpoint_stack = all_train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    foreground_mask_cache = {}

    # Deterministic image-space priors are evaluated once per training view and
    # Reuse the deterministic priors throughout optimization.
    deterministic_prior_cache = _precompute_deterministic_priors(
        all_train_cameras,
        args.foreground_mask_dir,
        foreground_mask_cache,
        need_structure=(s2ags_net is not None),
        need_highlight=(args.lambda_highlight > 0.0),
        structure_local_window=args.structure_local_window,
        structure_low_texture_quantile=args.structure_low_texture_quantile,
        structure_detail_quantile=args.structure_detail_quantile,
        structure_boundary_radius=args.structure_boundary_radius,
        highlight_quantile=args.highlight_quantile,
    )

    # Per-view cached asynchronous guidance (Sec. 3.4).  Guidance maps remain on
    # GPU at the state branch's native spatial resolution; deterministic binary
    # priors remain compact CPU tensors and are transferred on demand.
    guidance_cache = {}
    guidance_visits_since_refresh = {}
    state_refresh_count = 0
    state_reuse_count = 0

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    run_start_iter = first_iter
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # ================================================================
        # S²A-GS objective and ablation control.
        #
        # The spatial-state branch contains a full-foreground structural path
        # and candidate-gated reflective residual regulation.
        # The latter changes gradients received by the common Gaussian
        # representation; it is not a separate reflection renderer.
        # Consequently:
        #   1) w/o Highlight keeps the spatial-state structural path active;
        #   2) w/o S²A-State keeps only rule-based candidate regulation;
        #   3) the full model enables both paths.
        # ================================================================
        state_branch_refresh = False
        if s2ags_net is not None or args.use_rule_highlight:
            valid_mask = get_valid_foreground_mask(
                viewpoint_cam,
                gt_image,
                args.foreground_mask_dir,
                foreground_mask_cache,
            )
            gt_tensor = gt_image.unsqueeze(0)
            image_tensor = image.unsqueeze(0)
            view_key = _view_cache_key(viewpoint_cam)
            prior_item = deterministic_prior_cache.get(view_key, {})

            epsilon = 1e-3
            # Eq. (16): channel-averaged Charbonnier residual over the complete
            # foreground-constrained reference image domain.
            charbonnier_loss = torch.mean(
                torch.sqrt((image - gt_image).square() + epsilon**2)
            )

            zero_loss = image.new_zeros(())
            state_structure_loss = zero_loss
            structure_target_loss = zero_loss
            highlight_charbonnier_loss = charbonnier_loss
            highlight_confidence_loss = zero_loss

            # ------------------------------------------------------------
            # Per-view cached asynchronous state guidance (Sec. 3.4).
            # The first visit refreshes the branch.  The next seven visits to
            # the same view reuse fixed detached guidance; the ninth visit
            # refreshes again for K=8.
            # ------------------------------------------------------------
            structure_logits = None
            highlight_logits = None
            cached_guidance = guidance_cache.get(view_key)
            if s2ags_net is not None:
                visits = guidance_visits_since_refresh.get(view_key, 0)
                state_branch_refresh = (
                    cached_guidance is None
                    or visits >= args.guidance_refresh_interval
                )
                if state_branch_refresh:
                    if s2ags_optimizer is not None:
                        s2ags_optimizer.zero_grad(set_to_none=True)
                    structure_logits, highlight_logits = s2ags_net(gt_tensor)
                    state_refresh_count += 1
                else:
                    state_reuse_count += 1

            # ------------------------------------------------------------
            # Eq. (18): full-foreground spatial-state structural guidance.
            # Eq. (14) is evaluated only when the contextual branch refreshes;
            # intermediate visits reuse a detached structural-importance map.
            # ------------------------------------------------------------
            structure_score_for_cache = None
            if s2ags_net is not None:
                structure_target_cpu = prior_item.get("structure_target")
                if structure_target_cpu is None:
                    raise KeyError(
                        f"Missing cached structure target for training view: {view_key}"
                    )
                structure_target = _unpack_binary_prior(structure_target_cpu)

                if state_branch_refresh:
                    (
                        state_structure_loss,
                        structure_target_loss,
                        structure_score_for_cache,
                    ) = state_guided_structure_losses(
                        image_tensor,
                        gt_tensor,
                        valid_mask.unsqueeze(0),
                        structure_logits,
                        structure_target,
                        epsilon=epsilon,
                        gradient_weight=args.state_gradient_weight,
                        emphasis=args.state_structure_emphasis,
                    )
                else:
                    cached_structure = cached_guidance.get("structure_score")
                    if cached_structure is None:
                        raise KeyError(
                            f"Missing cached structural guidance for view: {view_key}"
                        )
                    state_structure_loss = state_guided_structure_loss_from_score(
                        image_tensor,
                        gt_tensor,
                        valid_mask.unsqueeze(0),
                        cached_structure,
                        epsilon=epsilon,
                        gradient_weight=args.state_gradient_weight,
                        emphasis=args.state_structure_emphasis,
                    )

            # ------------------------------------------------------------
            # Eqs. (15), (19), and (20): candidate-gated reflective residual
            # regulation.  At refresh steps the live reliability map receives
            # gradients from the current objective; intermediate visits reuse
            # the detached cached reliability map as a fixed contextual weight.
            # ------------------------------------------------------------
            confidence_for_cache = None
            if args.lambda_highlight > 0.0:
                highlight_candidate_cpu = prior_item.get("highlight_candidate")
                if highlight_candidate_cpu is None:
                    raise KeyError(
                        f"Missing cached high-luminance candidate for training view: {view_key}"
                    )
                highlight_candidate = _unpack_binary_prior(highlight_candidate_cpu)

                if s2ags_net is None:
                    # w/o S²A-State: fixed rule-only candidate reliability.
                    (
                        highlight_charbonnier_loss,
                        highlight_confidence_loss,
                        _,
                    ) = highlight_weighted_charbonnier_loss(
                        image_tensor,
                        gt_tensor,
                        valid_mask.unsqueeze(0),
                        highlight_candidate,
                        highlight_logits=None,
                        confidence_floor=args.highlight_confidence_floor,
                        fixed_confidence=args.rule_highlight_confidence,
                        epsilon=epsilon,
                    )
                elif state_branch_refresh:
                    (
                        highlight_charbonnier_loss,
                        highlight_confidence_loss,
                        confidence_for_cache,
                    ) = highlight_weighted_charbonnier_loss(
                        image_tensor,
                        gt_tensor,
                        valid_mask.unsqueeze(0),
                        highlight_candidate,
                        highlight_logits,
                        confidence_floor=args.highlight_confidence_floor,
                        fixed_confidence=args.rule_highlight_confidence,
                        epsilon=epsilon,
                    )
                else:
                    cached_confidence = cached_guidance.get("confidence")
                    if cached_confidence is None:
                        raise KeyError(
                            f"Missing cached photometric reliability for view: {view_key}"
                        )
                    highlight_charbonnier_loss = (
                        highlight_weighted_charbonnier_from_confidence(
                            image_tensor,
                            gt_tensor,
                            valid_mask.unsqueeze(0),
                            highlight_candidate,
                            cached_confidence,
                            epsilon=epsilon,
                        )
                    )
                    # Eq. (20) trains the reliability branch only at refresh
                    # steps.  It is constant with respect to Gaussian parameters
                    # during cached intermediate visits, so it is omitted here.
                    highlight_confidence_loss = zero_loss

            # Store detached guidance after the current refresh forward pass.
            if s2ags_net is not None:
                if state_branch_refresh:
                    cache_item = {
                        "structure_score": structure_score_for_cache.detach().contiguous(),
                    }
                    if args.lambda_highlight > 0.0:
                        cache_item["confidence"] = confidence_for_cache.detach().contiguous()
                    guidance_cache[view_key] = cache_item
                    guidance_visits_since_refresh[view_key] = 1
                else:
                    guidance_visits_since_refresh[view_key] = (
                        guidance_visits_since_refresh.get(view_key, 0) + 1
                    )

            photometric_loss = (
                (1.0 - args.lambda_highlight) * charbonnier_loss
                + args.lambda_highlight * highlight_charbonnier_loss
            )

            # Eq. (21): MS-SSIM term in the full S²A-GS objective.
            if args.use_ms_ssim:
                if not MS_SSIM_AVAILABLE:
                    raise ImportError(
                        "Please install pytorch-msssim first: pip install pytorch-msssim"
                    )
                image_ms = torch.clamp(image, 0.0, 1.0).unsqueeze(0)
                gt_ms = torch.clamp(gt_image, 0.0, 1.0).unsqueeze(0)
                min_hw = min(image_ms.shape[-2], image_ms.shape[-1])
                if min_hw < 160:
                    raise ValueError(
                        "MS-SSIM in the paper objective requires a sufficiently "
                        f"large image; got min spatial size {min_hw}."
                    )
                ms_ssim_value = ms_ssim(
                    image_ms,
                    gt_ms,
                    data_range=1.0,
                    size_average=True,
                )
                structure_loss = 1.0 - ms_ssim_value
            else:
                structure_loss = 1.0 - ssim_value

            loss = (
                (1.0 - opt.lambda_dssim) * photometric_loss
                + opt.lambda_dssim * structure_loss
                + args.lambda_state_structure * state_structure_loss
                + args.lambda_structure_target * structure_target_loss
                + args.lambda_highlight_confidence * highlight_confidence_loss
            )

        else:
            # ----- Vanilla 3DGS baseline computation path -----
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # ==========================================
                # S²A-GS: update the state-branch network weights.
                # Keep this outside the Gaussian optimizer branch so that
                # the state branch is also updated when SparseAdam is used.
                # ==========================================
                if s2ags_optimizer is not None and state_branch_refresh:
                    s2ags_optimizer.step()
                    s2ags_optimizer.zero_grad(set_to_none=True)
                    

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_payload = {
                    "gaussians": gaussians.capture(),
                    "iteration": iteration,
                    "state_branch": (
                        s2ags_net.state_dict() if s2ags_net is not None else None
                    ),
                    "state_optimizer": (
                        s2ags_optimizer.state_dict()
                        if s2ags_optimizer is not None
                        else None
                    ),
                }
                torch.save(
                    checkpoint_payload,
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    training_elapsed_seconds = time.perf_counter() - training_wall_start
    completed_iterations = max(0, int(opt.iterations) - int(run_start_iter))
    runtime_payload = {
        "completed_iterations": completed_iterations,
        "elapsed_seconds": training_elapsed_seconds,
        "elapsed_minutes": training_elapsed_seconds / 60.0,
        "throughput_it_per_s": (
            completed_iterations / training_elapsed_seconds
            if training_elapsed_seconds > 0 else None
        ),
        "guidance_refresh_interval_visits": (
            int(args.guidance_refresh_interval) if s2ags_net is not None else None
        ),
        "state_refresh_steps": int(state_refresh_count),
        "state_cached_reuse_steps": int(state_reuse_count),
        "resumed_from_checkpoint": bool(checkpoint),
    }
    _write_training_runtime(scene.model_path, runtime_payload)
    print(
        "[Timing] End-to-end optimization: "
        f"{runtime_payload['elapsed_minutes']:.3f} min, "
        f"{runtime_payload['throughput_it_per_s']:.3f} it/s"
    )
    if s2ags_net is not None:
        print(
            "[S²A-GS] Guidance schedule: "
            f"refresh={state_refresh_count}, cached_reuse={state_reuse_count}, "
            f"K={args.guidance_refresh_interval} visits"
        )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/reference".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # ==========================================
    # S²A-GS method switches
    # ==========================================
    parser.add_argument(
        "--use_state_branch",
        action="store_true",
        help="Enable the S²A-GS spatial-state guidance branch."
    )
    parser.add_argument(
        "--use_rule_highlight",
        action="store_true",
        help=(
            "Use fixed rule-based reflective residual regulation without the CNN/state branch. "
            "This is intended for the w/o Spatial-State Branch ablation."
        )
    )
    parser.add_argument(
        "--state_mode",
        choices=["none", "row", "row_col"],
        default="row_col",
        help="Spatial-state propagation mode: none, row, or row_col."
    )
    parser.add_argument(
        "--state_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the spatial-state guidance branch."
    )
    parser.add_argument(
        "--guidance_refresh_interval",
        type=int,
        default=8,
        help=(
            "Per-view guidance refresh interval K measured in visits to the same "
            "training view. The manuscript uses K=8."
        ),
    )
    parser.add_argument(
        "--foreground_mask_dir",
        type=str,
        default=None,
        help=(
            "Directory containing exact foreground masks matched by camera image name. "
            "Strongly recommended for S²A-GS and its ablations."
        ),
    )
    parser.add_argument(
        "--lambda_highlight",
        type=float,
        default=0.0,
        help=(
            "Mixing weight between the base and candidate-regulated Charbonnier "
            "terms. Must be in [0, 1]; use 0 for w/o Highlight."
        )
    )
    parser.add_argument(
        "--highlight_quantile",
        type=float,
        default=0.97,
        help="Foreground luminance quantile used to select candidate highlight pixels. Default 0.97 selects the brightest 3% foreground pixels."
    )
    parser.add_argument(
        "--use_ms_ssim",
        action="store_true",
        help="Use MS-SSIM as the structural loss in the S²A-GS branch."
    )
    parser.add_argument(
        "--lambda_state_structure",
        type=float,
        default=0.02,
        help=(
            "Weight of the full-foreground state-guided structural residual. "
            "Set to 0 when the complete spatial-state branch is removed."
        ),
    )
    parser.add_argument(
        "--lambda_structure_target",
        type=float,
        default=0.05,
        help="Weight of structure-map supervision for the spatial-state branch.",
    )
    parser.add_argument(
        "--lambda_highlight_confidence",
        type=float,
        default=0.01,
        help="Regularization weight that discourages indiscriminate candidate attenuation.",
    )
    parser.add_argument(
        "--state_gradient_weight",
        type=float,
        default=0.5,
        help="Gradient-consistency weight inside the state structural residual.",
    )
    parser.add_argument(
        "--state_structure_emphasis",
        type=float,
        default=1.0,
        help="Maximum additional emphasis assigned by the structure score.",
    )
    parser.add_argument(
        "--structure_local_window",
        type=int,
        default=7,
        help="Odd local window used to estimate weak-texture regions.",
    )
    parser.add_argument(
        "--structure_low_texture_quantile",
        type=float,
        default=0.30,
        help="Foreground local-variance quantile defining weak-texture candidates.",
    )
    parser.add_argument(
        "--structure_detail_quantile",
        type=float,
        default=0.70,
        help="Foreground gradient quantile defining local-detail candidates.",
    )
    parser.add_argument(
        "--structure_boundary_radius",
        type=int,
        default=2,
        help="Inner silhouette-boundary radius in pixels.",
    )
    parser.add_argument(
        "--highlight_confidence_floor",
        type=float,
        default=0.5,
        help="Lower bound of learned highlight reliability in the full model.",
    )
    parser.add_argument(
        "--rule_highlight_confidence",
        type=float,
        default=0.75,
        help=(
            "Fixed candidate reliability for the w/o S²A-State ablation. "
            "The default matches a zero-logit learned confidence when beta=0.5."
        ),
    )

    
    parser.add_argument("--use_vmamba", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--circular_scan", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args(sys.argv[1:])

    
    if args.use_vmamba:
        args.use_state_branch = True
    if args.circular_scan:
        args.use_state_branch = True
        args.state_mode = "row_col"

    if not 0.0 <= args.lambda_highlight <= 1.0:
        parser.error("--lambda_highlight must be in [0, 1].")
    if args.lambda_state_structure < 0.0 or args.lambda_structure_target < 0.0:
        parser.error("State-structure loss weights must be non-negative.")
    if args.lambda_highlight_confidence < 0.0:
        parser.error("--lambda_highlight_confidence must be non-negative.")
    if args.guidance_refresh_interval < 1:
        parser.error("--guidance_refresh_interval must be >= 1.")
    if not 0.0 < args.highlight_confidence_floor <= 1.0:
        parser.error("--highlight_confidence_floor must be in (0, 1].")
    if not 0.0 < args.rule_highlight_confidence <= 1.0:
        parser.error("--rule_highlight_confidence must be in (0, 1].")
    for name in (
        "highlight_quantile",
        "structure_low_texture_quantile",
        "structure_detail_quantile",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0, 1].")
    if args.use_rule_highlight and args.use_state_branch:
        parser.error("--use_rule_highlight and --use_state_branch are mutually exclusive.")
    if args.use_rule_highlight and args.lambda_highlight <= 0.0:
        parser.error("--use_rule_highlight requires --lambda_highlight > 0.")
    if (
        args.lambda_highlight > 0.0
        and not args.use_state_branch
        and not args.use_rule_highlight
    ):
        parser.error(
            "--lambda_highlight > 0 requires --use_state_branch or "
            "--use_rule_highlight."
        )
    if not args.use_state_branch and args.lambda_state_structure != 0.0:
        # Baseline and rule-only modes must not silently claim state guidance.
        args.lambda_state_structure = 0.0
        args.lambda_structure_target = 0.0
    if args.structure_local_window < 3 or args.structure_local_window % 2 == 0:
        parser.error("--structure_local_window must be an odd integer >= 3.")

    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    if args.use_state_branch:
        print(
            f"S²A-GS state branch enabled: state_mode={args.state_mode}, "
            f"lambda_state_structure={args.lambda_state_structure}, "
            f"lambda_highlight={args.lambda_highlight}, "
            f"highlight_quantile={args.highlight_quantile}, "
            f"guidance_refresh_interval={args.guidance_refresh_interval}"
        )
    elif args.use_rule_highlight:
        print(
            "S²A-GS spatial-state branch disabled; "
            f"fixed candidate residual regulation enabled with lambda_highlight={args.lambda_highlight}, "
            f"highlight_quantile={args.highlight_quantile}"
        )
    else:
        print("S²A-GS state branch disabled; vanilla 3DGS loss will be used.")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
