# Dataset Preparation for Training

Scal3R is fine-tuned on a single dataset, [TartanAir](https://theairlab.org/tartanair-dataset/), with the STream3R backbone entirely frozen. Please download the dataset from its official source and ensure compliance with the respective licensing agreements.

## TartanAir

1. Download the **TartanAir** dataset (left RGB images, depth, and optical flow). You can follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/prepare_training.md) to download it, or use the [official download tool](https://github.com/castacks/tartanair_tools). The raw data should be organized as:

```
data/tartanair/
└── {env}/                  # e.g. abandonedfactory, amusement, ...
    └── {Easy,Hard}/
        └── {Pxxx}/
            ├── image_left/
            ├── depth_left/
            ├── flow/
            └── pose_left.txt
```

2. Process the raw data into the CUT3R training format (per-frame RGB / depth / `_cam.npz` with camera pose and intrinsics), using the preprocessing script shared with the CUT3R-based implementation:

```bash
python ../scal3r_cut3r/datasets_preprocess/preprocess_tartanair.py \
    --tartanair_dir data/tartanair \
    --output_dir data/mast3r_data/processed_tartanair
```

3. The training config `configs/experiment/stream3r/finetune_scal3r.yaml` reads TartanAir through `data.train_datasets`. By default it uses `TartanAirWds_Multi`, which reads WebDataset-style tar shards for faster random access on network filesystems (e.g. Lustre). If you train on a standard local filesystem, simply switch the entry to the plain directory-based loader:

```yaml
train_datasets:
  - 4000 @ TartanAir_Multi(..., ROOT="${data.data_root}/processed_tartanair", ...)
```

keeping all other arguments (resolutions, `num_views`, etc.) unchanged.
