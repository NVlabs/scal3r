#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set -e

workdir='.'
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# ============================================================
# Environment fixes (REQUIRED for PGO + loop closure)
# ------------------------------------------------------------
# 1) gtsam (iSAM2 PGO) only imports if conda's libstdc++ (providing
#    CXXABI_1.3.15) is preloaded. Without this, gtsam import fails and PGO is
#    *silently* disabled -> the trajectory falls back to the raw chain and ATE
#    degrades a lot on drifty scenes (e.g. Sintel 0.157 -> ~0.30).
# 2) Loop closure needs faiss (descriptor index) and the VGGT-Long VPR model,
#    which is loaded through pytorch_lightning + pytorch_metric_learning.
#    pytorch_lightning additionally needs pkg_resources (setuptools<71) and an
#    importable wandb; a pinned wandb can ship a broken protobuf, so removing it
#    lets PL skip the wandb logger. Without faiss/pytorch_metric_learning the
#    KITTI loop-closure runs abort with ImportError.
# 3) The container image bakes in an older stream3r install; editable-install
#    this repo so `import stream3r` resolves to the local (modified) code, then
#    fail fast if it still resolves elsewhere.
# ============================================================
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
pip install -q 'setuptools<71' faiss-cpu pytorch-metric-learning >/dev/null 2>&1 || true
pip uninstall -y wandb >/dev/null 2>&1 || true
python -c "import gtsam" 2>/dev/null && echo "[env] gtsam import OK -> PGO enabled" \
    || echo "[env] WARNING: gtsam import FAILED -> PGO disabled (check LD_PRELOAD / libstdc++)"
python -c "import faiss, pytorch_metric_learning" 2>/dev/null \
    && echo "[env] faiss + pytorch_metric_learning OK -> loop closure enabled" \
    || echo "[env] WARNING: faiss/pytorch_metric_learning FAILED -> KITTI loop closure will error out"
unset PYTHONPATH   # the container bakes /workspace/CUT3R into PYTHONPATH, shadowing eval/
pip install -q -e . --no-deps >/dev/null 2>&1 || true
python scripts/check_local_stream3r.py

# ============================================================
# Helper function
# ============================================================
# Model weights are loaded from HF Hub (nvidia/scal3r) inside launch.py.
run_eval() {
    local dataset=$1
    local tag=$2
    shift 2
    local output_dir="${workdir}/eval_results/relpose/${tag}/${dataset}"
    echo "=========================================="
    echo ">>> ${tag} / ${dataset}"
    echo "=========================================="
    accelerate launch --num_processes 1 --main_process_port ${MASTER_PORT} \
        eval/relpose/launch.py \
        --output_dir "${output_dir}/" \
        --eval_dataset "${dataset}" \
        --use_rel_pose \
        "$@"
}

# ============================================================
# TUM (ATE: 0.0197, verified 2026-07-05 with crop + HF weights)
#   Best: causal, kf12, nkf0, kfc, nif1, global_pose_init σ=0.01
# ============================================================
run_eval tum best_camtoken_align \
    --mode causal --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 0 --kf_only_cache \
    --num_init_frames 1 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3 \
    --use_global_pose_init --global_pose_prior_sigma 0.01

# ============================================================
# ScanNet (ATE: 0.0494, verified 2026-07-05 with crop + HF weights)
#   Best: causal, kf12, nkf0, kfc, nif1, global_pose_init σ=0.01
# ============================================================
run_eval scannet best_camtoken_align \
    --mode causal --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 0 --kf_only_cache \
    --num_init_frames 1 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3 \
    --use_global_pose_init --global_pose_prior_sigma 0.01

# ============================================================
# Sintel (ATE: 0.1808, verified 2026-07-05 with crop + HF weights; was 0.157 no-crop)
#   Best: window, kf20, nkf24, NO kfc, NO global_pose_init
# ============================================================
run_eval sintel best_camtoken_align \
    --mode window --kf_window 20 --max_ref_frames 20 --nkf_buffer_size 24 \
    --num_init_frames 2 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3

# ============================================================
# vKITTI (ATE: 4.48, verified 2026-07-05 with crop + HF weights)
#   Best: window, kf8, nkf0, reset_interval=20, NO kfc, NO global_pose_init
# ============================================================
run_eval vkitti best_camtoken_align \
    --mode window --kf_window 8 --max_ref_frames 8 --nkf_buffer_size 0 \
    --num_init_frames 2 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3 \
    --reset_interval 20

# ============================================================
# KITTI odometry (00-10)
#   Base: window, kf12, nkf8, reset_interval=10, default sigma
#   No-loop best on 01,03,04,10: kf12, nkf8, r40 (avg ATE 57.62)
#   But r10 is used for loop closure compatibility
#
#   All seven invocations below share one output dir on purpose: the reported
#   average is recomputed from every <seq>/<seq>_eval_metric.txt under it, so
#   only the LAST run's "Average ATE" line covers all 11 sequences.
#   => the dir must be EMPTY before the first KITTI run. Leftover metric files
#      from an earlier run (e.g. an older seq00/ layout) are silently folded
#      into the average.
# ============================================================
rm -rf "${workdir}/eval_results/relpose/best_camtoken_align/kitti_odom"
KITTI_COMMON="--mode window --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 --num_init_frames 2"

# Per-seq loop closure with optimal thresholds (sweep results, r10 for loop propagation)
KITTI_LOOP="$KITTI_COMMON --reset_interval 10 --loop_closure --loop_temporal_gap 200"
run_eval kitti_odom best_camtoken_align --seq_list 00 $KITTI_LOOP --loop_similarity_threshold 0.85
run_eval kitti_odom best_camtoken_align --seq_list 02 $KITTI_LOOP --loop_similarity_threshold 0.45
run_eval kitti_odom best_camtoken_align --seq_list 05 $KITTI_LOOP --loop_similarity_threshold 0.70
run_eval kitti_odom best_camtoken_align --seq_list 06 $KITTI_LOOP --loop_similarity_threshold 0.80
run_eval kitti_odom best_camtoken_align --seq_list 07 $KITTI_LOOP --loop_similarity_threshold 0.60
run_eval kitti_odom best_camtoken_align --seq_list 08 $KITTI_LOOP --loop_similarity_threshold 0.45
# Sequences without loop closure (kf12 nkf4 r40 optimal from sweep)
KITTI_NOLOOP="--mode window --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 4 --num_init_frames 2 --reset_interval 40"
run_eval kitti_odom best_camtoken_align --seq_list 01 03 04 09 10 $KITTI_NOLOOP

echo ">>> All done"
