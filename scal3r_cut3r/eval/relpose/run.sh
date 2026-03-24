#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: bash eval/relpose/run.sh <checkpoint_path> [model_name]"
    echo "  e.g. bash eval/relpose/run.sh src/checkpoints/scal3r/checkpoint-best.pth"
    exit 1
fi

workdir='.'
model_weights="$1"
model_name="${2:-$(basename $(dirname $1))}"

NUM_GPUS=${NUM_GPUS:-8}
PORT=${PORT:-29558}
EXTRA_ARGS=${EXTRA_ARGS:-}

run_eval() {
    local data=$1
    shift
    local output_dir="${workdir}/eval_results/relpose/${data}_${model_name}"
    echo ""
    echo "========== ${data} =========="
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port ${PORT} \
        eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --use_relative_pose \
        $EXTRA_ARGS \
        "$@"
    if [ -f "${output_dir}/_error_log.txt" ]; then
        tail -2 "${output_dir}/_error_log.txt"
    fi
}

# TUM: default kf4
run_eval tum

# Sintel: kf8 + nkf8
run_eval sintel --kf_window 8 --max_ref_frames 8 --nkf_buffer_size 8

# ScanNet: default kf4
run_eval scannet

# vkitti: kf12 + nkf12 + no_kf_gate + reset_interval 10
run_eval vkitti --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 12 --no_kf_gate --reset_interval 10

# KITTI odometry (00-10): per-sequence optimal loop closure thresholds
# V10: Huber robust kernel on loop noise, iSAM2 finalize reads all poses,
#      FAISS top-20 with temporal_gap pre-filter, fixed loop sigma_scale=1.0
KITTI_BASE="--kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 --no_kf_gate --reset_interval 10 --num_init_frames 2"
KITTI_LOOP="$KITTI_BASE --loop_closure --loop_temporal_gap 200"

# Sequences with loop closure (per-seq optimal threshold from sweep)
run_eval kitti_odom --seq_list 00 $KITTI_LOOP --loop_similarity_threshold 0.85
run_eval kitti_odom --seq_list 02 $KITTI_LOOP --loop_similarity_threshold 0.50
run_eval kitti_odom --seq_list 05 $KITTI_LOOP --loop_similarity_threshold 0.75
run_eval kitti_odom --seq_list 06 $KITTI_LOOP --loop_similarity_threshold 0.85
run_eval kitti_odom --seq_list 07 $KITTI_LOOP --loop_similarity_threshold 0.60
run_eval kitti_odom --seq_list 08 $KITTI_BASE --loop_closure --loop_temporal_gap 500 --loop_similarity_threshold 0.45
run_eval kitti_odom --seq_list 09 $KITTI_LOOP --loop_similarity_threshold 0.70

# Sequences without loop (no revisit / too short)
run_eval kitti_odom --seq_list 01 03 04 10 $KITTI_BASE
