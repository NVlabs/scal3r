# Datasets

We evaluate camera pose estimation on **TUM-dynamics**, **Sintel**, **ScanNet**, **Virtual KITTI 2**, and **KITTI odometry**.

Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare the **Sintel**, **TUM-dynamics**, and **ScanNet** datasets. For convenience, STream3R also provides processed versions on [Hugging Face](https://huggingface.co/datasets/yslan/pointmap_regression_evalsets).

For **Virtual KITTI 2**, download the preprocessed version released by [CUT3R](https://drive.google.com/file/d/1KdAH4ztRkzss1HCkGrPjQNnMg5c-f3aD/view?usp=sharing) into `data/processed_vkitti`.

For **KITTI odometry**, download the color images and ground-truth poses from the [official benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), organized as:

```
kitti_data/
├── sequences/
│   └── {00..10}/image_2/*.png
└── poses/
    └── {00..10}.txt
```

The datasets should be organized as follows (dataset roots are configured in `eval/relpose/metadata.py`; adjust the `kitti_odom` entry to point to your KITTI odometry path):

```
data/
├── processed_vkitti
├── scannetv2
├── sintel
└── tum
```

For loop closure on KITTI, additionally place the VGGT-Long VPR weights at `reference/VGGT-Long/weights/{dinov2_vitb14_pretrain.pth,dino_salad.ckpt}`.

# Evaluation

### Camera Pose Estimation

To evaluate on all benchmarks (TUM, Sintel, ScanNet, vKITTI, KITTI odometry 00–10), run:

```bash
# Scal3R weights are loaded automatically from Hugging Face (nvidia/scal3r)
bash eval/relpose/run.sh
```

Results (ATE / RPE trans / RPE rot) will be saved in `eval_results/relpose/${tag}/${data}/_error_log.txt`.

The script applies the per-benchmark inference settings used in the paper:

| Benchmark | Settings |
|-----------|----------|
| TUM / ScanNet | `--mode causal --kf_window 12 --max_ref_frames 12 --kf_only_cache --num_init_frames 1 --use_global_pose_init --global_pose_prior_sigma 0.01` |
| Sintel | `--mode window --kf_window 20 --max_ref_frames 20 --nkf_buffer_size 24 --num_init_frames 2` |
| vKITTI | `--mode window --kf_window 8 --max_ref_frames 8 --num_init_frames 2 --reset_interval 20` |
| KITTI odometry | `--mode window --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 --num_init_frames 2 --reset_interval 10`, plus `--loop_closure` with per-sequence similarity thresholds |

> **Important — Pose-Graph Optimization (PGO) & loop closure dependencies.**
> Relative-pose evaluation uses `gtsam` (iSAM2 PGO), and KITTI additionally
> uses a VGGT-Long DINOv2+SALAD model for loop closure. `eval/relpose/run.sh`
> already applies the required environment fixes, but if you call
> `eval/relpose/launch.py` directly, replicate them:
> ```bash
> # gtsam needs conda's libstdc++ (CXXABI_1.3.15); otherwise its import fails
> # and PGO is SILENTLY skipped (ATE degrades, e.g. Sintel 0.18 -> ~0.30).
> export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6"
> # Loop closure imports a VPR model via pytorch_lightning, which needs
> # pkg_resources (setuptools<71) and an importable wandb (remove a broken one).
> pip install 'setuptools<71' faiss-cpu pytorch-metric-learning
> pip uninstall -y wandb
> ```

### Multi-view Reconstruction

Since the backbone and reconstruction heads are frozen and pose tokens are injected via asymmetric attention, Scal3R's pointmap quality is identical to the original STream3R. To verify:

```bash
bash eval/mv_recon/run.sh
```

Results will be saved in `eval_results/mv_recon/stream3r/7scenes/logs_all.txt`.

The script uses `--kf_every 1 --max_frames 300` (dense consecutive frames). Using the default `kf_every=200` produces sparse sampling that does not match the paper's evaluation protocol.