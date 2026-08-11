# Datasets

We evaluate camera pose estimation on **TUM-dynamics**, **Sintel**, **ScanNet**, **Virtual KITTI 2**, and **KITTI odometry**, and multi-view reconstruction on **7-Scenes**.

Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare the **Sintel**, **TUM-dynamics**, **ScanNet**, and **7-Scenes** datasets.

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
├── 7scenes
├── processed_vkitti
├── scannetv2
├── sintel
└── tum
```

# Evaluation

### Camera Pose Estimation

To evaluate on all benchmarks (TUM, Sintel, ScanNet, vKITTI, KITTI odometry 00–10), run:

```bash
# bash eval/relpose/run.sh <checkpoint_path> [model_name]
bash eval/relpose/run.sh src/checkpoints/scal3r_cut3r.pth

# override GPU count / port if needed
NUM_GPUS=1 bash eval/relpose/run.sh src/checkpoints/scal3r_cut3r.pth
```

Results (ATE / RPE trans / RPE rot) will be saved in `eval_results/relpose/${data}_${model_name}/_error_log.txt`.

The script applies the per-benchmark inference settings used in the paper:

| Benchmark | Settings |
|-----------|----------|
| TUM / ScanNet | default (K=4 keyframe references) |
| Sintel | `--kf_window 8 --max_ref_frames 8 --nkf_buffer_size 8` |
| vKITTI | `--kf_window 12 --max_ref_frames 12 --nkf_buffer_size 12 --no_kf_gate --reset_interval 10` |
| KITTI odometry | `--kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 --no_kf_gate --reset_interval 10 --num_init_frames 2`, plus `--loop_closure` with per-sequence similarity thresholds |

Useful flags (pass through `EXTRA_ARGS` or append to `eval/relpose/launch.py`):

- `--no_relative_pose`: evaluate the backbone's original absolute pose regression (CUT3R baseline).
- `--skip_pgo`: chain the predicted relative poses without pose-graph optimization (ablation).
- `--loop_closure --loop_similarity_threshold T --loop_temporal_gap G`: enable online loop closure (DINOv2 + SALAD retrieval).

### Multi-view Reconstruction

Since the backbone and reconstruction heads are frozen and pose tokens are injected via asymmetric attention, Scal3R's pointmap quality is identical to the original CUT3R. To verify on 7-Scenes:

```bash
bash eval/mv_recon/run.sh
```

Results will be saved in `eval_results/mv_recon/scal3r_scal3r_cut3r/7scenes/logs_all.txt`.

The script uses `--kf_every 1 --max_frames 300` (dense consecutive frames) and runs on a single GPU (`--num_processes 1`) to avoid OOM. Using the default `kf_every=200` produces sparse sampling that does not match the paper's evaluation protocol.