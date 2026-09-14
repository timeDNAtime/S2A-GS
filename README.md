# S²A-GS: Spatial-State-Aware Gaussian Splatting for Ceramic Object Reconstruction

Core implementation and public dataset release for:

**Spatial-State-Aware Gaussian Splatting for Ceramic Object Reconstruction under Uncontrolled Illumination**

S²A-GS is a training-time optimization framework for object-level ceramic reconstruction with 3D Gaussian Splatting (3DGS). It uses shared spatial-state context to coordinate structural guidance and reflective-region residual regulation under challenging conditions such as view-dependent specular reflection, weak texture, thin boundaries, hollow structures, and local surface details.

The auxiliary spatial-state branch is used only during training. After optimization, rendering uses the optimized Gaussian representation and the standard Gaussian rasterizer.

---

## Repository Contents

```text
S2A-GS/
├── Ceramic3D-General Core Set/   # Public 12-object ceramic image dataset
├── train.py                      # Main S²A-GS training entry
├── render.py                     # Novel-view rendering
├── s2ags_network.py              # Spatial-state contextual branch
├── s2ags_losses.py               # Structural / reflective-region losses and priors
├── evaluate_fg_metrics.py        # Object-level evaluation metrics
├── requirements.txt              # Python dependencies
└── README.md
```

This is a **lightweight core release**. It provides the main S²A-GS implementation, rendering code, evaluation code, and the Ceramic3D-General Core Set.

Preprocessing utilities, third-party baseline implementations, and experiment-management scripts are not included in this minimal repository.

---

## Method Overview

S²A-GS formulates ceramic reconstruction as **spatial-state-conditioned residual optimization**.

The core training pipeline contains:

- foreground-constrained Gaussian optimization;
- hierarchical visual feature encoding;
- bidirectional row- and column-wise selective state propagation;
- full-foreground structural guidance for weak-texture, local-detail, and inner-boundary regions;
- candidate-gated reflective residual regulation in high-luminance regions;
- view-specific cached asynchronous guidance;
- standard Gaussian rendering at inference.

For the final configuration, the spatial-state guidance is refreshed every **8 visits to the same training view**. During intermediate visits, detached guidance maps are reused as fixed contextual weights while Gaussian parameters continue to be optimized.

The deterministic image-space priors are precomputed for each training view and reused throughout optimization.

---

## Base Implementation

This code is built on the official **GraphDECO/Inria 3D Gaussian Splatting (3DGS)** implementation.

The files in this repository contain the S²A-GS-specific modifications and are intended to be used with a compatible 3DGS codebase that provides the standard modules required by `train.py` and `render.py`, including components such as:

```text
arguments/
gaussian_renderer/
scene/
utils/
```

Please install the original 3DGS implementation and its CUDA rasterization dependencies before using the S²A-GS files.

Original copyright and license notices in modified upstream files should be retained.

---

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` includes the main Python packages used by the S²A-GS-specific code.

A CUDA-enabled PyTorch environment compatible with the underlying 3DGS implementation is required.

Main additional dependencies include:

- PyTorch / TorchVision
- NumPy / SciPy
- OpenCV
- scikit-image
- LPIPS
- pytorch-msssim
- mamba-ssm

Please follow the original 3DGS installation instructions for the Gaussian rasterizer and other CUDA extensions.

---

## Ceramic3D-General Core Set

The repository includes the **Ceramic3D-General Core Set**, a 12-object ceramic image dataset captured with a consumer-grade mobile device under uncontrolled or weakly controlled indoor illumination.

The Core Set covers four dominant reconstruction challenges:

1. Fine surface patterns with thin boundaries
2. Weakly textured continuous curved surfaces
3. Strongly specular glazed surfaces
4. Complex local structures

The public dataset directory is organized as:

```text
Ceramic3D-General Core Set/
├── images_01/
├── images_02/
├── images_03/
├── images_04/
├── images_05/
├── images_06/
├── images_07/
├── images_08/
├── images_09/
├── images_10/
├── images_11/
└── images_12/
```

Each `images_XX/` directory contains the captured multi-view RGB images for one ceramic object.

### Important

The public Core Set in this lightweight repository contains the captured image data. Before S²A-GS training, the images must be converted into the input format expected by the underlying 3DGS pipeline, including camera registration and foreground masks.

The experiments reported in the manuscript use:

- sharp-frame selection from captured video;
- GLOMAP-based camera registration;
- foreground masks generated with a detection-prompted segmentation pipeline;
- a train/test split following the LLFFhold=8 protocol.

These preprocessing utilities are not included in this minimal core release.

---

## Training

### 3DGS-Mask baseline

After preparing a foreground-constrained 3DGS scene, a standard masked baseline can be trained with the underlying 3DGS training interface.

Example:

```bash
python train.py \
  -s /path/to/processed_scene \
  -i images_masked \
  -m outputs/3dgs_mask \
  --iterations 30000 \
  --eval
```

### Full S²A-GS

Example:

```bash
python train.py \
  -s /path/to/processed_scene \
  -i images_masked \
  -m outputs/s2ags_full \
  --iterations 30000 \
  --eval \
  --use_state_branch \
  --state_mode row_col \
  --foreground_mask_dir /path/to/masks \
  --lambda_dssim 0.2 \
  --lambda_state_structure 0.02 \
  --lambda_structure_target 0.05 \
  --lambda_highlight 0.02 \
  --lambda_highlight_confidence 0.01 \
  --highlight_quantile 0.97 \
  --guidance_refresh_interval 8 \
  --use_ms_ssim
```

The main S²A-GS-specific options include:

```text
--use_state_branch
--state_mode {none,row,row_col}
--foreground_mask_dir
--guidance_refresh_interval
--highlight_quantile
--lambda_state_structure
--lambda_structure_target
--lambda_highlight
--lambda_highlight_confidence
```

The default spatial-state configuration uses row-column bidirectional propagation.

---

## Rendering

Render the optimized model using:

```bash
python render.py \
  -m outputs/s2ags_full \
  --iteration 30000
```

To render only the test views:

```bash
python render.py \
  -m outputs/s2ags_full \
  --iteration 30000 \
  --skip_train
```

After training, the auxiliary spatial-state branch is not required for rendering.

---

## Evaluation

The released evaluation script supports the five object-level metrics used in the study:

- FG-PSNR ↑
- FG-SSIM ↑
- FG-LPIPS ↓
- Silhouette IoU ↑
- Projected Opacity Leakage ↓

Example:

```bash
python evaluate_fg_metrics.py \
  --model_path outputs/s2ags_full \
  --mask_dir /path/to/test_masks \
  --alpha_dir /path/to/accumulated_alpha \
  --hold 8 \
  --save_json outputs/s2ags_full/metrics.json
```

`Projected Opacity Leakage` should be computed from accumulated opacity / alpha maps. The optional RGB-luminance fallback is not equivalent to opacity-based leakage and should not be used when reporting the manuscript metric.

Silhouette IoU and Projected Opacity Leakage are image-space / projection-space indicators of object-level structural consistency; they are not direct measurements of 3D geometric accuracy.

---

## Citation

If you use S²A-GS or the Ceramic3D-General Core Set in your research, please cite:

**Spatial-State-Aware Gaussian Splatting for Ceramic Object Reconstruction under Uncontrolled Illumination**

Jie Yang, Kaihua Hu, Guangyang Chen

Citation information will be updated after publication.

---

## License and Third-Party Code

S²A-GS is built on the GraphDECO/Inria 3D Gaussian Splatting implementation. Modified upstream source files retain their original copyright notices.

Users are responsible for complying with the license terms of:

- the original 3DGS implementation;
- the Gaussian rasterization dependencies;
- mamba-ssm;
- LPIPS;
- any other third-party packages used with this repository.

The Ceramic3D-General Core Set should be used for research and academic purposes unless otherwise stated by the repository owner.

---

## Contact

For questions about the code or dataset, please contact:

**Kaihua Hu**  
School of Information Engineering  
Jingdezhen Ceramic University  
Jingdezhen, Jiangxi 333403, China  
Email: hukaihua@jci.edu.cn
