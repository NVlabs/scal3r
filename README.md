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

<p align="center">
  <img src="./assets/teaser.gif" alt="Scal3R teaser" width="100%">
</p>

<div align="center">
TL;DR: Scalable online 3D reconstruction on kilometer-scale sequences, with only ~1% extra parameters on a frozen backbone trained in 8 hours on a single GPU.
</div>
<br>

> This repository contains the **CUT3R-based** implementation of Scal3R. For the STream3R-based implementation, see the [`scal3r_stream3r`](https://github.com/NVlabs/scal3r/tree/scal3r_stream3r) branch.

## Getting Started

### Installation

1. Clone Scal3R.
```bash
git clone https://github.com/NVlabs/scal3r.git
cd scal3r
```

2. Create the environment.
```bash
conda create -n scal3r python=3.11 cmake=3.14.0
conda activate scal3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # use the correct version of cuda for your system
pip install -r requirements.txt
# issues with pytorch dataloader, see https://github.com/pytorch/pytorch/issues/99625
conda install 'llvm-openmp<16'
# for evaluation
pip install evo
# for online pose-graph optimization
pip install gtsam
# for loop closure (DINOv2 + SALAD retrieval)
pip install faiss-cpu pytorch_lightning pytorch_metric_learning
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).
```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoints

The released Scal3R checkpoint contains the full model (frozen CUT3R backbone + relative pose query modules), so it is the only file needed for inference and evaluation:

```bash
# Scal3R (CUT3R backbone) checkpoint
huggingface-cli download nvidia/scal3r cut3r/scal3r_cut3r.pth --local-dir src/checkpoints
mv src/checkpoints/cut3r/scal3r_cut3r.pth src/checkpoints/scal3r_cut3r.pth
```

If you want to train Scal3R yourself (see [Training](#training)), additionally download the pretrained CUT3R backbone into `src/`:

```bash
cd src
# CUT3R 512 dpt ckpt (frozen backbone, training only)
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```

### Inference Demo

To run the inference demo, you can use the following command:
```bash
# input can be a folder or a video
# the following script will run streaming inference with multi-reference relative
# pose query + online PGO, and visualize the output with viser on port 8080
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path src/checkpoints/scal3r_cut3r.pth \
    --size 512 --seq_path SEQ_PATH --output_dir OUT_DIR \
    --use_relative_pose --vis_threshold 1.5

# Example: long outdoor sequence (vKITTI-style settings: K=12 references,
# state reset every 10 frames)
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path src/checkpoints/scal3r_cut3r.pth \
    --size 512 --seq_path data/processed_vkitti/Scene01/clone/Camera_0 \
    --use_relative_pose \
    --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 12 --no_kf_gate --reset_interval 10 \
    --downsample_factor 100

# Example: KITTI odometry with loop closure
CUDA_VISIBLE_DEVICES=0 python demo.py --model_path src/checkpoints/scal3r_cut3r.pth \
    --size 512 --seq_path data/kitti_data/sequences/07/image_2 \
    --use_relative_pose \
    --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 --no_kf_gate --reset_interval 10 --num_init_frames 2 \
    --loop_closure --loop_temporal_gap 200 --loop_similarity_threshold 0.60 \
    --downsample_factor 100
```

### Evaluation
Please refer to the [eval.md](docs/eval.md) for more details.

### Training
Please refer to the [train.md](docs/train.md) for dataset preparation ([preprocess.md](docs/preprocess.md)) and training commands.

## License

All [CUT3R](https://github.com/CUT3R/CUT3R) code and NVIDIA modifications of CUT3R are released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license (see [LICENSE](LICENSE)). The separable NVIDIA-authored files `src/dust3r/utils/alignment.py`, `src/dust3r/utils/loop_closure.py`, and `src/dust3r/utils/pgo.py` are released under the NVIDIA License (see [LICENSE_NVIDIA](LICENSE_NVIDIA)). Each source file carries a header identifying its applicable license.

## Acknowledgements
Our code is based on the following awesome repositories:

- [CUT3R](https://github.com/CUT3R/CUT3R), [STream3R](https://github.com/NIRVANALAN/STream3R), [Human3R](https://github.com/fanegg/Human3R), [DUSt3R](https://github.com/naver/dust3r), [MonST3R](https://github.com/Junyi42/monst3r.git), [Spann3R](https://github.com/HengyiWang/spann3r.git), [GTSAM](https://github.com/borglab/gtsam), [SALAD](https://github.com/serizba/salad), [Viser](https://github.com/nerfstudio-project/viser)

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
