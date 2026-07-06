<p align="center">
  <h1 align="center">[ECCV 2026] Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction</h1>
  <p align="center">
    <a href="https://linjohnss.github.io/"><strong>Chin-Yang Lin</strong></a>
    ·
    <a href="https://tw.linkedin.com/in/yang-che-sun-6b9341282"><strong>Yang-Che Sun</strong></a>
    ·
    <a href="https://sunset1995.github.io/"><strong>Cheng Sun</strong></a>
    ·
    <a href="https://fuenyang1127.github.io/"><strong>Fu-En Yang</strong></a>
    <br>
    <a href="https://minhungchen.netlify.app/"><strong>Min-Hung Chen</strong></a>
    ·
    <a href="https://sites.google.com/site/yylinweb/"><strong>Yen-Yu Lin</strong></a>
    ·
    <a href="https://walonchiu.github.io/"><strong>Wei-Chen Chiu</strong></a>
    ·
    <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a>
  </p>
  <h3 align="center"><a href="https://linjohnss.github.io/scal3r/">Project Page</a> | <a href="https://linjohnss.github.io/scal3r/">Paper</a> | <a href="https://huggingface.co/nvidia/scal3r">🤗 Hugging Face</a></h3>
  <div align="center"></div>
</p>

<div align="center">
TL;DR: Scalable online 3D reconstruction on kilometer-scale sequences, with only ~1% extra parameters on a frozen backbone trained in 8 hours on a single GPU.
</div>
<br>

> This repository contains the **STream3R-based** implementation of Scal3R. For the CUT3R-based implementation, see `scal3r_cut3r`.

## Getting Started

### Installation

1. Clone Scal3R.
```bash
git clone https://github.com/NVlabs/scal3r.git
cd scal3r/scal3r_stream3r
```

2. Create the environment.
```bash
conda create -n scal3r_stream3r python=3.11 cmake=3.14.0 -y
conda activate scal3r_stream3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # use the correct version of cuda for your system
pip install -r requirements.txt
# install this repo as a package
pip install -e .
# for evaluation
pip install evo
# for online pose-graph optimization
pip install gtsam
# for loop closure (DINOv2 + SALAD retrieval)
pip install faiss-cpu pytorch_metric_learning
```

### Download Checkpoints

The released Scal3R checkpoint contains the full model (frozen STream3R backbone + relative pose query modules) in Hugging Face format, and is downloaded automatically on first use:

```python
from stream3r.models.stream3r import STream3R

model = STream3R.from_pretrained("nvidia/scal3r")
```

If you want to train Scal3R yourself (see [Training](#training)), additionally download the pretrained STream3R backbone from [Hugging Face](https://huggingface.co/yslan/STream3R) and place it at `weights/stream3r/model.pt`.

### Inference Demo

To run the inference demo, you can use the following command:
```bash
# input can be a folder or a video
# the following script will run streaming inference with multi-reference relative
# pose query + online PGO, and visualize the output with viser on port 8080
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path nvidia/scal3r \
    --seq_path SEQ_PATH --output_dir OUT_DIR \
    --use_streaming --use_rel_pose --vis_threshold 1.5

# Example: long outdoor sequence (vKITTI-style settings: sliding window attention,
# state reset every 10 frames)
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path nvidia/scal3r \
    --seq_path data/processed_vkitti/Scene01/clone/Camera_0 \
    --use_streaming --mode window \
    --kf_window 8 --max_ref_frames 8 --nkf_buffer_size 8 \
    --num_init_frames 2 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3 \
    --reset_interval 10 \
    --downsample_factor 100

# Example: KITTI odometry with loop closure
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path nvidia/scal3r \
    --seq_path data/kitti_data/sequences/00/image_2 \
    --use_streaming --mode window \
    --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 \
    --num_init_frames 2 --reset_interval 10 \
    --loop_closure --loop_similarity_threshold 0.80 --loop_temporal_gap 200 \
    --downsample_factor 40 --vis_threshold 1.5 --vis_stride 10
```

### Evaluation
Please refer to the [eval.md](docs/eval.md) for more details.

### Training
Please refer to the [train.md](docs/train.md) for dataset preparation ([preprocess.md](docs/preprocess.md)) and training commands.

## Acknowledgements
Our code is based on the following awesome repositories:

- [STream3R](https://github.com/NIRVANALAN/STream3R), [CUT3R](https://github.com/CUT3R/CUT3R), [VGG-T](https://github.com/facebookresearch/vggt), [Human3R](https://github.com/fanegg/Human3R), [DUSt3R](https://github.com/naver/dust3r), [MonST3R](https://github.com/Junyi42/monst3r.git), [Spann3R](https://github.com/HengyiWang/spann3r.git), [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), [GTSAM](https://github.com/borglab/gtsam), [SALAD](https://github.com/serizba/salad), [Viser](https://github.com/nerfstudio-project/viser)

We thank the authors for releasing their code!

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{lin2026scal3r,
  title={Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction},
  author={Lin, Chin-Yang and Sun, Yang-Che and Sun, Cheng and Yang, Fu-En and Chen, Min-Hung and Lin, Yen-Yu and Chiu, Wei-Chen and Liu, Yu-Lun},
  booktitle={ECCV},
  year={2026}
}
```
