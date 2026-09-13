# S²A-GS: Spatial-State-Aware Gaussian Splatting for Ceramic Object Reconstruction

This repository contains the official implementation of **S²A-GS**, a training-time robust optimization framework for object-level ceramic reconstruction under uncontrolled illumination.

S²A-GS combines:

- foreground-constrained RGB inputs,
- a spatial-state branch with hierarchical feature encoding and selective state propagation,
- full-foreground state-guided structural optimization for weak-texture,
  boundary, and local-detail regions,
- state-conditioned regulation of unreliable view-dependent residuals over
  candidate high-luminance regions,
- accumulated-opacity evaluation for object-level leakage analysis,
- per-view cached asynchronous guidance with a refresh interval of K=8 visits.

The spatial-state branch is used only during training. After optimization, novel-view rendering uses the optimized Gaussian representation and the standard Gaussian renderer.

The reflective-region path is broader than a highlight-shape enhancer. By
changing the photometric gradients that pass through the differentiable
rasterizer, it can reduce highlight baking into Gaussian appearance and
opacity, and limit reflection-driven errors in primitive placement,
densification, contours, and nearby local structure. It retains specular cues
supported by the captured views, but it does **not** estimate BRDF,
illumination, or a physically separated reflection layer. Accordingly, paper
claims should use observation consistency and reconstruction stability rather
than physical reflectance accuracy.

For the corresponding manuscript equations and section-by-section revisions,
see [`docs/current_manuscript_alignment.md`](docs/current_manuscript_alignment.md).
The implementation changes and validation status are summarized in
[`CHANGELOG_STATE_STRUCTURE.md`](CHANGELOG_STATE_STRUCTURE.md).

## Repository structure

```text
S-A-GS/
├── train.py                         # 3DGS training with S²A-GS switches
├── render.py                        # test/train view rendering
├── s2ags_network.py                 # spatial-state branch
├── s2ags_losses.py                  # structural/highlight losses and priors
├── tools/
│   ├── frames.py                    # sharp-frame extraction from videos
│   ├── generate_masks.py            # YOLO + SAM2 foreground mask generation
│   ├── run_colmap.py                # COLMAP + GLOMAP pose estimation
│   ├── render_opacity.py            # accumulated opacity/alpha export
│   ├── evaluate_fg_metrics.py       # FG-5 metrics and POL evaluation
│   ├── run_experiments.py           # batch training/rendering/evaluation
│   └── summarize_results.py         # aggregate CSV files into mean/std summaries
├── scripts/                         # paper-result reproduction commands
├── docs/                            # dataset, metric, and reproduction notes
├── configs/                         # example experiment configs
├── data/                            # dataset instructions or example sample
└── outputs/                         # generated results, ignored by git
```

## Installation

This repository is based on GraphDECO/Inria 3D Gaussian Splatting. Install the original 3DGS dependencies first, including the Gaussian rasterizer used by your 3DGS fork. Then install the S²A-GS-specific dependencies:

```bash
conda env create -f environment.yml
conda activate s2ags
```

or:

```bash
pip install -r requirements.txt
```

Additional external dependencies:

- COLMAP
- GLOMAP
- SAM2
- YOLO11 / Ultralytics
- a COLMAP vocabulary tree file, for example `vocab_tree_b100k_k256.bin`
- `mamba-ssm` for `--state_mode row` and `--state_mode row_col`

## Dataset format

A processed sample should follow this structure:

```text
data/processed/01/
├── images/                          # extracted RGB frames
├── masks/                           # foreground masks
├── images_masked/                   # foreground-constrained RGB images
└── colmap_db/
    ├── database.db
    └── sparse/
```

See [`docs/dataset_format.md`](docs/dataset_format.md) for the full specification.

## Data preparation

### 1. Extract sharp frames from videos

```bash
python tools/frames.py \
  --video_dir data/raw_videos \
  --output_dir data/processed \
  --extract_fps 3 \
  --overwrite
```

### 2. Generate foreground masks

```bash
python tools/generate_masks.py \
  --base_out_dir data/processed \
  --yolo_model yolo11n.pt \
  --sam2_config_dir /path/to/sam2/configs \
  --sam2_config sam2.1_hiera_l.yaml \
  --sam2_checkpoint /path/to/sam2.1_hiera_large.pt \
  --device cuda
```

### 3. Estimate camera poses

```bash
python tools/run_colmap.py \
  --base_dir data/processed \
  --vocab_tree /path/to/vocab_tree_b100k_k256.bin \
  --matcher vocab_tree \
  --camera_model PINHOLE \
  --overwrite
```

## Training examples

### 3DGS-Mask baseline

```bash
python train.py \
  -s data/processed/01/colmap_db \
  -i images_masked \
  -m outputs/01_3DGS_Mask \
  -r 2 \
  --iterations 30000 \
  --eval
```

### Full S²A-GS

```bash
python train.py \
  -s data/processed/01/colmap_db \
  -i images_masked \
  -m outputs/01_S2A_GS_Full \
  -r 2 \
  --iterations 30000 \
  --eval \
  --use_state_branch \
  --state_mode row_col \
  --foreground_mask_dir data/processed/01/masks \
  --lambda_dssim 0.2 \
  --lambda_state_structure 0.02 \
  --lambda_structure_target 0.05 \
  --lambda_highlight 0.02 \
  --lambda_highlight_confidence 0.01 \
  --highlight_quantile 0.97 \
  --guidance_refresh_interval 8 \
  --use_ms_ssim
```

In the revised implementation, the spatial-state branch has two heads.  The
structure head predicts a full-foreground importance map supervised by
non-parametric weak-texture, local-detail, and silhouette-boundary targets.
The resulting map emphasizes a Charbonnier-plus-gradient residual over the
whole foreground.  The highlight head predicts candidate reliability for
regulating unreliable photometric residuals only inside high-luminance
regions. These weighted residuals still update the same Gaussian appearance,
opacity, spatial parameters, and densification process through
backpropagation.

### Main ablation: w/o Highlight

This setting retains CNN-based hierarchical encoding and row-column state
propagation, including the full-foreground structural loss, but disables the
highlight-aware path.

```bash
python train.py \
  -s data/processed/01/colmap_db \
  -i images_masked \
  -m outputs/01_S2A_GS_NoHighlight \
  -r 2 \
  --iterations 30000 \
  --eval \
  --use_state_branch \
  --state_mode row_col \
  --foreground_mask_dir data/processed/01/masks \
  --lambda_dssim 0.2 \
  --lambda_state_structure 0.02 \
  --lambda_structure_target 0.05 \
  --lambda_highlight 0.0 \
  --lambda_highlight_confidence 0.01 \
  --guidance_refresh_interval 8 \
  --use_ms_ssim
```

### Main ablation: w/o S²A-State

This removes the complete spatial-state branch and keeps fixed rule-based
residual regulation inside the luminance-based highlight candidates.

```bash
python train.py \
  -s data/processed/01/colmap_db \
  -i images_masked \
  -m outputs/01_S2A_GS_NoSpatialState \
  -r 2 \
  --iterations 30000 \
  --eval \
  --use_rule_highlight \
  --foreground_mask_dir data/processed/01/masks \
  --lambda_dssim 0.2 \
  --lambda_highlight 0.02 \
  --lambda_highlight_confidence 0.01 \
  --highlight_quantile 0.97 \
  --use_ms_ssim
```

## Cached asynchronous guidance

The current manuscript uses a **per-view** refresh interval of `K=8` visits.
Each training view owns an independent guidance cache and visit counter. On a
refresh visit the spatial-state branch is evaluated and updated; detached
structural-importance and photometric-reliability maps are stored. During the
next seven visits to that same view the cached maps are reused as fixed
contextual weights while Gaussian parameters continue to update every
iteration. Deterministic structure/high-luminance priors are precomputed once
per training view.

Training writes `training_runtime.json` with end-to-end elapsed time, throughput,
and refresh/reuse counts. The spatial-state branch and all caches are discarded
after optimization.

## External comparison baselines

The paper also reports **2DGS** and **GaussianShader**. Their third-party
implementations are not vendored into this repository. Run those methods from
their official repositories using the same registered cameras, masked-RGB
training images, LLFFhold=8 split, and test viewpoints, then evaluate their
renders/opacity maps with `tools/evaluate_fg_metrics.py`. This keeps licensing
and upstream implementations separate while preserving the paper's common
evaluation protocol.

## Evaluation

S²A-GS reports FG-PSNR, FG-SSIM, FG-LPIPS, Silhouette IoU, and Projected Opacity Leakage. POL is computed from accumulated opacity/alpha maps, not RGB luminance.

```bash
python render.py \
  -m outputs/01_S2A_GS_Full \
  --skip_train

python tools/render_opacity.py \
  -m outputs/01_S2A_GS_Full \
  --skip_train

python tools/evaluate_fg_metrics.py \
  --model_path outputs/01_S2A_GS_Full \
  --mask_dir data/processed/01/masks \
  --alpha_dir outputs/01_S2A_GS_Full/test/ours_30000/opacity \
  --hold 8 \
  --save_json outputs/01_S2A_GS_Full/fg5_metrics.json
```

See [`docs/metrics.md`](docs/metrics.md) for metric definitions.

## Reproducing paper tables

Use `tools/run_experiments.py` or the shell scripts under `scripts/`.

```bash
bash scripts/run_main_ablation.sh
bash scripts/run_state_direction_ablation.sh
bash scripts/run_mask_sensitivity.sh
bash scripts/run_highlight_quantile.sh
```

The batch runner writes command logs and metric JSON files for each experiment.

The batch runner also records `training_runtime.json`. To reproduce the paper's
paired object-level confidence intervals and grouped challenge-category table:

```bash
python tools/bootstrap_paired_ci.py \
  --csv outputs/main_ablation_fg5_metrics.csv \
  --baseline 3DGS_Masked --method S2A_GS_Full \
  --resamples 10000 --seed 42 \
  --output outputs/main_bootstrap_ci.csv

python tools/summarize_categories.py \
  --csv outputs/main_ablation_fg5_metrics.csv \
  --output outputs/grouped_table3.csv
```

For mask sensitivity, `tools/perturb_masks.py` can generate deterministic
eroded/dilated training masks before running `scripts/run_mask_sensitivity.sh`.


## Method names

| Code name | Paper name |
|---|---|
| `3DGS_Raw` | 3DGS-Raw |
| `3DGS_Masked` | 3DGS-Mask |
| `S2A_GS_NoHighlight` | w/o Highlight |
| `S2A_GS_NoSpatialState` | w/o S²A-State |
| `S2A_GS_NoStatePropagation` | w/o State Propagation |
| `S2A_GS_RowState` | Row-wise State |
| `S2A_GS_Full` | Ours |

## Data availability

If the full Ceramic3D-General Core Set is not hosted in this repository, place the download link here. The processed release should include extracted RGB frames, foreground masks, camera parameters, and split information.

```text
Dataset link: TBD
```

## Citation

```bibtex
@article{yang2026s2ags,
  title={Spatial-State-Aware Gaussian Splatting for Ceramic Object Reconstruction under Uncontrolled Illumination},
  author={Yang, Jie and Hu, Kaihua and Chen, Guangyang},
  journal={TBD},
  year={2026}
}
```

## Acknowledgements

This repository builds upon the original 3D Gaussian Splatting implementation by GraphDECO/Inria. Please follow the original license terms for GraphDECO-derived files.

## License

See [`LICENSE.md`](LICENSE.md). Files derived from GraphDECO/Inria 3DGS retain the original non-commercial research/evaluation license requirements.
