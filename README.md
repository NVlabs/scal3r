<div align="center">

# Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction

[Chin-Yang Lin](https://linjohnss.github.io/)<sup>1,2</sup>, Yang-Che Sun<sup>1</sup>, [Cheng Sun](https://sunset1995.github.io/)<sup>2</sup>, [Fu-En Yang](https://fuenyang1127.github.io/)<sup>2</sup><br>
[Min-Hung Chen](https://minhungchen.netlify.app/)<sup>2</sup>, [Yen-Yu Lin](https://sites.google.com/site/yylinweb/)<sup>1</sup>, [Wei-Chen Chiu](https://walonchiu.github.io/)<sup>1</sup>, [Yu-Lun Liu](https://yulunalexliu.github.io/)<sup>1</sup>

<sup>1</sup>National Yang Ming Chiao Tung University &nbsp;&nbsp; <sup>2</sup>NVIDIA

<!-- [![Project Page](https://img.shields.io/badge/Project-Page-blue)](URL) -->
<!-- [![Paper](https://img.shields.io/badge/arXiv-PDF-red)](URL) -->

</div>

## Overview

Scal3R freezes a pretrained online 3D reconstruction model and learns a small set of **pose query tokens** (~1% params) to extract multi-reference relative poses from frozen representations via **asymmetric attention**. Combined with keyframe selection and incremental **pose-graph optimization (PGO)**, Scal3R scales existing 3R models to large-scale scenes with significantly reduced drift.

## Repository Structure

```
scal3r_cut3r/       # Scal3R on CUT3R
scal3r_stream3r/    # Scal3R on STream3R
```

## Installation

### Scal3R-CUT3R

```bash
cd scal3r_cut3r
conda create -n scal3r_cut3r python=3.11 cmake=3.14.0
conda activate scal3r_cut3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Compile RoPE CUDA kernels (CroCo v2)
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Scal3R-STream3R

```bash
cd scal3r_stream3r
conda create -n scal3r_stream3r python=3.11 cmake=3.14.0
conda activate scal3r_stream3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install STream3R as a package
pip install -e .
```

### Optional: PGO & Loop Closure

```bash
pip install gtsam faiss-gpu
```

## Checkpoints

<!-- TODO: add download links -->

Download the Scal3R weights and the corresponding base model weights:

| Model | Base Weights | Scal3R Weights |
|-------|-------------|----------------|
| Scal3R-CUT3R | [cut3r_512_dpt_4_64.pth](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link) → `scal3r_cut3r/src/` | coming soon |
| Scal3R-STream3R | [stream3r](https://github.com/NiranjanLan/STream3R) → `scal3r_stream3r/weights/` | coming soon |

## Inference

The `demo.py` scripts run inference on an image folder (or video) and launch an interactive 3D point cloud viewer via [Viser](https://github.com/nerfstudio-project/viser).

### Scal3R-CUT3R

```bash
cd scal3r_cut3r

# Base CUT3R (no relative pose)
python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path <IMAGE_FOLDER> --size 512

# Scal3R-CUT3R with PGO
python demo.py --model_path src/checkpoints/scal3r/checkpoint-best.pth \
    --seq_path <IMAGE_FOLDER> --use_relative_pose \
    --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 \
    --no_kf_gate --reset_interval 10 --num_init_frames 2

# With loop closure (for long/revisiting sequences)
python demo.py --model_path src/checkpoints/scal3r/checkpoint-best.pth \
    --seq_path <IMAGE_FOLDER> --use_relative_pose \
    --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 \
    --no_kf_gate --reset_interval 10 --num_init_frames 2 \
    --loop_closure --loop_temporal_gap 200 --loop_similarity_threshold 0.60
```

<details>
<summary>Key arguments</summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `src/cut3r_512_dpt_4_64.pth` | Model checkpoint path |
| `--seq_path` | — | Image folder or video file |
| `--size` | `512` | Input image size (224 or 512) |
| `--use_relative_pose` | off | Enable Scal3R relative pose + PGO |
| `--no_relative_pose` | off | Force base CUT3R pose only |
| `--kf_window` | `4` | Keyframe buffer size |
| `--max_ref_frames` | model default | Max reference frames for multi-ref pose |
| `--nkf_buffer_size` | `0` | Non-keyframe buffer size |
| `--reset_interval` | `1000000` | Reset streaming state every N frames |
| `--num_init_frames` | `5` | Initial frames treated as keyframes |
| `--no_kf_gate` | off | Disable keyframe-gated state update |
| `--skip_pgo` | off | Skip PGO, use chain accumulation only |
| `--loop_closure` | off | Enable loop closure detection |
| `--downsample_factor` | `1` | Point cloud visualization downsample |
| `--vis_threshold` | `1.5` | Confidence threshold for visualization |

</details>

### Scal3R-STream3R

```bash
cd scal3r_stream3r

# Base STream3R (no relative pose)
python demo.py --model_path yslan/STream3R \
    --seq_path <IMAGE_FOLDER> --use_streaming --mode window

# Scal3R-STream3R with PGO
python demo.py --model_path weights/scal3r_stream3r/model.pt \
    --seq_path <IMAGE_FOLDER> --use_streaming --mode window \
    --use_rel_pose \
    --kf_window 8 --max_ref_frames 8 --nkf_buffer_size 8 \
    --num_init_frames 2 --reset_interval 10

# With loop closure
python demo.py --model_path weights/scal3r_stream3r/model.pt \
    --seq_path <IMAGE_FOLDER> --use_streaming --mode window \
    --use_rel_pose \
    --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 \
    --num_init_frames 2 --reset_interval 10 \
    --loop_closure --loop_similarity_threshold 0.80 --loop_temporal_gap 200
```

<details>
<summary>Key arguments</summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `yslan/STream3R` | Checkpoint path or HuggingFace model name |
| `--seq_path` | — | Image folder or video file |
| `--size` | `518` | Input image size |
| `--mode` | `causal` | Attention mode: `causal`, `window`, or `full` |
| `--use_streaming` | off | Enable streaming inference with KV cache |
| `--use_rel_pose` | off | Enable Scal3R relative pose + PGO |
| `--kf_window` | `4` | Keyframe buffer size |
| `--max_ref_frames` | `4` | Max reference frames for multi-ref pose |
| `--nkf_buffer_size` | `0` | Non-keyframe buffer size |
| `--reset_interval` | `1000000` | Reset streaming state every N frames |
| `--num_init_frames` | `5` | Initial frames treated as keyframes |
| `--skip_pgo` | off | Skip PGO, use chain accumulation only |
| `--loop_closure` | off | Enable loop closure detection |
| `--vis_stride` | `1` | Frame stride for visualization (saves memory) |

</details>

## Training

Scal3R freezes the pretrained base model and only trains the pose query tokens, projection layers, and relative pose decoder (~1% of total parameters). Both codebases use [Hydra](https://hydra.cc/) for configuration.

### Data Preparation

We use [TartanAir](https://theairlab.org/tartanair-dataset/) for training. Follow [CUT3R's preprocessing guide](scal3r_cut3r/docs/preprocess.md) to prepare the dataset, then update the dataset root path in the config files accordingly.

### Scal3R-CUT3R

Training uses [Accelerate](https://huggingface.co/docs/accelerate) for multi-GPU. The config is at `scal3r_cut3r/config/train_scal3r_stage1.yaml`.

```bash
cd scal3r_cut3r/src
accelerate launch --multi_gpu train.py --config-name train_scal3r_stage1
```

Key training settings (in `config/train_scal3r_stage1.yaml`):
- **Frozen**: encoder + decoder + reconstruction head (`freeze='encoder_and_decoder_and_head'`)
- **Trainable**: `relative_pose_token`, `prev_pose_proj`, `RelativePoseDecoder`
- **Loss**: relative pose only (`use_relative_pose_loss=True`, `use_pts_loss=False`)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.05
- **Schedule**: 40 epochs, 5 warmup epochs, batch_size=8

### Scal3R-STream3R

Training uses [PyTorch Lightning](https://lightning.ai/) + [DeepSpeed](https://www.deepspeed.ai/). The experiment config is at `scal3r_stream3r/configs/experiment/stream3r/stream3r_rel_pose.yaml`.

```bash
cd scal3r_stream3r
python stream3r/train.py experiment=stream3r/stream3r_rel_pose
```

Key training settings (in `configs/experiment/stream3r/stream3r_rel_pose.yaml`):
- **Frozen**: all except `rel_pose_token`, `prev_pose_proj`, `rel_pose_decoder` (`freeze='rel_pose_prompt'`)
- **Trainable**: pose query tokens + projection + decoder
- **Loss**: relative pose only (`use_rel_pose_loss=True`, all other loss weights=0)
- **Optimizer**: AdamW, lr=1e-4, weight_decay=0.05
- **Schedule**: 40 epochs, 5 warmup epochs, batch_size=8, DeepSpeed bf16-mixed precision

## Acknowledgements

This work builds on [CUT3R](https://github.com/CUT3R/CUT3R), [STream3R](https://github.com/NiranjanLan/STream3R), [GTSAM](https://gtsam.org/), and [DINOv2](https://github.com/facebookresearch/dinov2).

## Citation

```bibtex
@inproceedings{lin2026scal3r,
  title={Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction},
  author={Lin, Chin-Yang and Sun, Yang-Che and Sun, Cheng and Yang, Fu-En and Chen, Min-Hung and Lin, Yen-Yu and Chiu, Wei-Chen and Liu, Yu-Lun},
  year={2026}
}
```
