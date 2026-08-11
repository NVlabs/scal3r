# Datasets

We freeze all weights of the pretrained CUT3R backbone (encoder, decoder, and reconstruction heads) and fine-tune only the relative pose query modules — the learnable base query token, the per-reference projection MLPs, and the relative pose head — which together account for only ~1% of the total parameters.

Fine-tuning uses **TartanAir** only. For each training sample, we randomly select 4 views from a sequence: one serves as the current frame and the remaining three as reference frames (K=3). A Random Interval Sampling strategy perturbs the temporal intervals between the selected frames, forcing the model to extract stable relative pose representations under varying motion velocities and baseline lengths. Since each pose query token attends independently, the number of references can be freely scaled at inference time (we use K=12 on KITTI/vKITTI) without retraining.

Please refer to [preprocess.md](preprocess.md) for downloading and processing TartanAir, and set the dataset path in `config/finetune_scal3r.yaml`.

# Training

Download the pretrained CUT3R checkpoint into `src/` (see [README](../README.md#download-checkpoints)), then run:

```bash
# Remember to replace the dataset path in config/finetune_scal3r.yaml with your own path
cd src/

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    train.py --config-name finetune_scal3r \
    pretrained=cut3r_512_dpt_4_64.pth hydra.job.chdir=False
```

Training runs for 40 epochs with batch size 8 (AdamW, lr 1e-4, weight decay 0.05, 5 warmup epochs) and converges in approximately 8 hours on a single NVIDIA A100 GPU. Checkpoints and logs are written to `src/checkpoints/finetune_scal3r/`.

Notes on the config (`config/finetune_scal3r.yaml`):

- `freeze='encoder_and_decoder_and_head'` keeps the entire backbone frozen; `output_mode='pts3d+pose+relative_pose'` and `num_prompt_tokens=1` enable the relative pose query tokens.
- The training criterion supervises **relative poses only** (`use_rel_pose_loss=True`, with `use_pts_loss` / `use_pose_loss` disabled): rotation and translation are penalized separately, and translations are scale-aligned before the loss to handle monocular scale ambiguity (`use_align_scale=True`). Adjust `rel_rot_loss_weight` / `rel_trans_loss_weight` if either term dominates.
- `num_views: 4` controls the views per training sample (1 current + 3 references).
