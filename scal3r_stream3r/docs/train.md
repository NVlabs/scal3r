# Datasets

We freeze all weights of the pretrained STream3R backbone (aggregator and reconstruction heads) and fine-tune only the relative pose query modules — the learnable base query token, the per-reference projection MLP, and the relative pose decoder (`freeze: "rel_pose_prompt"` in the config) — which together account for only ~1% of the total parameters.

Fine-tuning uses **TartanAir** only. For each training sample, we randomly select 4 views from a sequence: one serves as the current frame and the remaining three as reference frames (K=3). A Random Interval Sampling strategy perturbs the temporal intervals between the selected frames, forcing the model to extract stable relative pose representations under varying motion velocities and baseline lengths. Since each pose query token attends independently, the number of references can be freely scaled at inference time (we use K=12 on KITTI) without retraining.

Please refer to [preprocess.md](preprocess.md) for downloading and processing TartanAir, and set `data.data_root` in `configs/experiment/stream3r/finetune_scal3r.yaml`.

# Training

1. Download the pretrained STream3R backbone from [Hugging Face](https://huggingface.co/yslan/STream3R) and place it at `weights/stream3r/model.pt` (or set `model.pretrained` in the config).

2. Launch training:

```bash
# Remember to set data.data_root in configs/experiment/stream3r/finetune_scal3r.yaml
python stream3r/train.py experiment=stream3r/finetune_scal3r slurm_job_id=0
```

Training runs for 40 epochs with batch size 8 (lr 1e-4, 5 warmup epochs) and converges in approximately 8 hours on a single NVIDIA A100 GPU. Checkpoints and logs are written to `logs/stream3r/runs/finetune_scal3r_${slurm_job_id}/`. If interrupted, re-running the same command auto-resumes from `checkpoints/last.ckpt`.

3. After training, convert the DeepSpeed checkpoint into a `state_dict` file:

```python
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir="logs/stream3r/runs/finetune_scal3r_0/checkpoints/last.ckpt",
    output_file="logs/stream3r/runs/finetune_scal3r_0/checkpoints/last_aggregated.ckpt",
    tag=None,
)
```

Notes on the config (`configs/experiment/stream3r/finetune_scal3r.yaml`):

- `use_rel_pose_prompt: true` and `num_rel_pose_tokens: 1` enable the shared learnable base query token; `max_ref_frames: 4` sets the reference buffer size during training (freely overridable at inference).
- The training loss supervises **relative poses only**: rotation and translation are penalized separately, and translations are scale-aligned before the loss (`use_align_scale: true`) to handle monocular scale ambiguity.
- `data.num_views: 4` controls the views per training sample (1 current + 3 references).
