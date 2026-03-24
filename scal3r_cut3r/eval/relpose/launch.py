# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import math
import cv2
import numpy as np
import torch
import argparse

from copy import deepcopy
from eval.relpose.metadata import dataset_metadata
from eval.relpose.utils import *

from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r

from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help="model type for state update strategy: cut3r or ttt3r",
    )
    parser.add_argument(
        "--ttt3r_bias",
        type=float,
        default=0.0,
        help="bias added to cross-attn logits before sigmoid in ttt3r (0=original, positive=more update)",
    )
    parser.add_argument(
        "--ttt3r_mode",
        type=str,
        default="attn",
        help="ttt3r signal mode: attn (cross-attn maps, manual attention) or delta (state feat change, Flash Attention)",
    )
    parser.add_argument(
        "--ttt3r_scale",
        type=float,
        default=10.0,
        help="scale factor for delta mode: sigmoid((1-cos_sim)*scale + bias)",
    )

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
    parser.add_argument(
        "--use_relative_pose",
        action="store_true",
        default=False,
        help="Use relative pose accumulation + PGO instead of absolute camera_pose. "
        "If not set, auto-detects based on model output.",
    )
    parser.add_argument(
        "--no_relative_pose",
        action="store_true",
        default=False,
        help="Force absolute camera_pose even if relative_pose is available.",
    )
    parser.add_argument(
        "--skip_pgo",
        action="store_true",
        default=False,
        help="Skip PGO optimization, use chain accumulation only.",
    )
    parser.add_argument(
        "--pgo_sigma_rot",
        type=float,
        default=None,
        help="Override base sigma for rotation in PGO (default: 0.5)",
    )
    parser.add_argument(
        "--pgo_sigma_trans",
        type=float,
        default=None,
        help="Override base sigma for translation in PGO (default: 0.5)",
    )
    parser.add_argument(
        "--pgo_position_scale",
        type=float,
        default=0,
        help="Position-dependent sigma: sigma *= (1 + frame_idx / position_scale). 0=disabled.",
    )
    parser.add_argument(
        "--pgo_max_edges",
        type=int,
        default=0,
        help="Max PGO constraints per frame (0=unlimited, 2=only gap-1 and gap-2).",
    )
    parser.add_argument(
        "--pgo_mode",
        type=str,
        default=None,
        choices=["huber", "dcs", "cauchy", "tukey", "irls"],
        help="PGO robust kernel mode (default: huber)",
    )
    parser.add_argument(
        "--max_ref_frames",
        type=int,
        default=None,
        help="Override model's max_ref_frames at inference time (default: model config)",
    )
    parser.add_argument(
        "--kf_window",
        type=int,
        default=4,
        help="Number of keyframes to keep in buffer (default: 4)",
    )
    parser.add_argument(
        "--nkf_buffer_size",
        type=int,
        default=0,
        help="Number of non-keyframes to keep in buffer (default: 0)",
    )
    parser.add_argument(
        "--keyframe_overlap_thr",
        type=float,
        default=None,
        help="Override keyframe overlap threshold (default: use inference.py default)",
    )
    # Phase 3 features
    parser.add_argument(
        "--pgo_adaptive_sigma",
        action="store_true",
        default=False,
        help="Adapt PGO sigma based on sequence length (short=tighter, long=looser)",
    )
    parser.add_argument(
        "--pgo_warmstart",
        action="store_true",
        default=False,
        help="Use PGO #1 (iSAM2) result as initial guess for PGO #2",
    )
    parser.add_argument(
        "--kf_edges_only",
        action="store_true",
        default=False,
        help="Only keep PGO edges where at least one endpoint is a keyframe",
    )
    parser.add_argument(
        "--no_correct_scale",
        action="store_true",
        default=False,
        help="Disable scale correction in eval alignment (SE(3) instead of Sim(3))",
    )
    parser.add_argument(
        "--no_finalize",
        action="store_true",
        default=False,
        help="Skip iSAM2 finalize (extra iterations + read all poses) for ablation",
    )
    parser.add_argument(
        "--rot_gap_power",
        type=float,
        default=None,
        help="Gap scaling power for rotation sigma (default: 0.5 = sqrt)",
    )
    parser.add_argument(
        "--trans_gap_power",
        type=float,
        default=None,
        help="Gap scaling power for translation sigma (default: 0.5 = sqrt)",
    )
    parser.add_argument(
        "--use_pgo1_poses",
        action="store_true",
        default=False,
        help="Use PGO #1 (iSAM2 incremental) poses directly, skip chain + PGO #2",
    )
    parser.add_argument(
        "--use_online_pgo",
        action="store_true",
        default=False,
        help="Use online PGO (sliding window) poses instead of global PGO",
    )
    parser.add_argument(
        "--force_kf_gate",
        action="store_true",
        default=False,
        help="Force keyframe-gated state update regardless of sequence length",
    )
    parser.add_argument(
        "--no_kf_gate",
        action="store_true",
        default=False,
        help="Disable keyframe-gated state update (all frames update state)",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=1000000,
        help="Reset recurrent state every N frames with 1 overlap frame (default: disabled)",
    )
    parser.add_argument(
        "--final_batch_opt",
        action="store_true",
        default=False,
        help="Run final Levenberg-Marquardt batch optimization after incremental iSAM2",
    )
    # Loop closure arguments
    parser.add_argument(
        "--loop_closure",
        action="store_true",
        default=False,
        help="Enable online loop closure detection during inference",
    )
    parser.add_argument(
        "--loop_similarity_threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for loop detection (higher = fewer but more reliable)",
    )
    parser.add_argument(
        "--loop_temporal_gap",
        type=int,
        default=300,
        help="Minimum frame gap for loop closure candidates",
    )
    parser.add_argument(
        "--loop_max_per_frame",
        type=int,
        default=1,
        help="Maximum number of loop closure frames to inject per keyframe",
    )
    parser.add_argument(
        "--loop_nms_window",
        type=int,
        default=50,
        help="NMS window: suppress loop if query/target both within this many frames of a recent loop",
    )
    parser.add_argument(
        "--loop_max_translation",
        type=float,
        default=20.0,
        help="Geometric verification: reject loop edges with predicted translation > this (meters)",
    )
    parser.add_argument(
        "--loop_gt",
        action="store_true",
        default=False,
        help="Use GT poses for loop detection (ablation only, not for real eval)",
    )
    parser.add_argument(
        "--loop_sigma_scale",
        type=float,
        default=1.0,
        help="Sigma scale for loop closure identity constraint (lower = tighter snap)",
    )
    parser.add_argument(
        "--reset_kf_interval",
        type=int,
        default=0,
        help="Reset buffer every N keyframes (0=disabled, use static reset_interval)",
    )
    parser.add_argument(
        "--kf_ref_only",
        action="store_true",
        default=False,
        help="Only use keyframe indices as references (for noref ablation: accumulate from last KF)",
    )
    parser.add_argument(
        "--num_init_frames",
        type=int,
        default=5,
        help="Number of initial frames treated as keyframes before overlap-based selection (default: 5)",
    )
    parser.add_argument(
        "--save_relative_poses",
        action="store_true",
        default=False,
        help="Save per-frame raw relative poses to CSV (pred_relative_poses.csv per sequence)",
    )
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    if metadata is None:
        raise ValueError(f"Unknown dataset {args.eval_dataset!r}. Available: {sorted(dataset_metadata.keys())}")
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    from dust3r.inference import inference, inference_recurrent, make_kf_only_callbacks

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    # Loop closure detector (loaded once, reused across sequences)
    loop_detector = None
    if args.loop_closure and not getattr(args, 'loop_gt', False):
        from dust3r.loop_closure import OnlineLoopDetector
        loop_detector = OnlineLoopDetector(
            device=device,
            similarity_threshold=args.loop_similarity_threshold,
            temporal_gap=args.loop_temporal_gap,
            max_loops_per_frame=args.loop_max_per_frame,
            nms_window=args.loop_nms_window,
        )
        print(f"Loop closure enabled: threshold={args.loop_similarity_threshold}, "
              f"gap={args.loop_temporal_gap}, max_per_frame={args.loop_max_per_frame}")
    elif args.loop_closure and getattr(args, 'loop_gt', False):
        print("Loop closure enabled: GT detection mode (ablation)")

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        fps_list = []
        mem_list = []
        total_frames = 0
        total_infer_time = 0.0
        load_img_size = args.size
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                img_filter = metadata.get("img_filter", None)
                filelist = [
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    if img_filter is None or img_filter(name)
                ]
                filelist.sort()
                filelist = filelist[:: args.pose_eval_stride]

                # Filter out frames with -inf GT poses (e.g. missing tracking in ScanNet)
                gt_valid_mask = None
                gt_traj_file_early = metadata["gt_traj_func"](img_path, anno_path, seq)
                if gt_traj_file_early is not None and os.path.isfile(gt_traj_file_early):
                    with open(gt_traj_file_early, 'r') as f:
                        gt_lines = f.readlines()
                    gt_lines = gt_lines[:: args.pose_eval_stride]
                    gt_valid_mask = ['-inf' not in line for line in gt_lines[:len(filelist)]]
                    n_invalid = sum(1 for v in gt_valid_mask if not v)
                    if n_invalid > 0:
                        print(f"Filtering {n_invalid}/{len(filelist)} frames with -inf GT poses in {seq}")
                        filelist = [f for f, v in zip(filelist, gt_valid_mask) if v]

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                    revisit=args.revisit,
                    update=not args.freeze_state,
                    reset_interval=args.reset_interval,
                )
                # Keyframe selection (works for both CUT3R and Scal3R:
                # CUT3R uses camera_pose for c2w, Scal3R uses relative_poses)
                # Reset / create loop detector for new sequence
                seq_loop_detector = loop_detector  # SALAD detector (shared)
                if args.loop_closure and getattr(args, 'loop_gt', False):
                    from dust3r.loop_closure import GTLoopDetector
                    gt_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                    if gt_file and os.path.isfile(gt_file):
                        seq_loop_detector = GTLoopDetector(
                            gt_poses_file=gt_file,
                            temporal_gap=args.loop_temporal_gap,
                            max_loops_per_frame=args.loop_max_per_frame,
                            nms_window=args.loop_nms_window,
                            reset_interval=args.reset_interval,
                        )
                        print(f"  GT loop detector for {seq}: {gt_file}")
                if seq_loop_detector is not None:
                    seq_loop_detector.reset()
                kf_extra_params = {}
                if args.keyframe_overlap_thr is not None:
                    kf_extra_params['keyframe_overlap_thr'] = args.keyframe_overlap_thr
                if args.no_finalize:
                    kf_extra_params['do_finalize'] = False
                ref_frame_indices_fn, on_frame_processed, keyframe_indices, buffer_pruning_fn = make_kf_only_callbacks(
                    kf_window=args.kf_window,
                    nkf_buffer_size=args.nkf_buffer_size,
                    pgo_sigma_rot=args.pgo_sigma_rot,
                    pgo_sigma_trans=args.pgo_sigma_trans,
                    pgo_position_scale=args.pgo_position_scale,
                    pgo_max_edges=args.pgo_max_edges,
                    final_batch_opt=getattr(args, 'final_batch_opt', False),
                    loop_detector=seq_loop_detector,
                    loop_image_paths=_build_view_image_paths(filelist, args.reset_interval),
                    loop_max_translation=args.loop_max_translation,
                    loop_sigma_scale=args.loop_sigma_scale,
                    reset_kf_interval=args.reset_kf_interval,
                    kf_ref_only=getattr(args, 'kf_ref_only', False),
                    num_init_frames=args.num_init_frames,
                    **kf_extra_params,
                )
                # Override model's max_ref_frames at inference time
                if args.max_ref_frames is not None:
                    model.max_ref_frames = args.max_ref_frames
                # State gating: for Scal3R on long sequences (>60 frames),
                # keyframe-based state update reduces noise from redundant frames.
                # Short sequences need every frame's state update.
                # CUT3R always passes None (matches training).
                if args.no_kf_gate:
                    kf_for_state = None
                elif args.force_kf_gate or (args.use_relative_pose and len(views) > 60):
                    kf_for_state = keyframe_indices
                else:
                    kf_for_state = None
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                t_infer_start = time.perf_counter()
                outputs, _ = inference_recurrent(
                    views, model, device,
                    ref_frame_indices_fn=ref_frame_indices_fn,
                    keyframe_indices=kf_for_state,
                    on_frame_processed=on_frame_processed,
                    buffer_pruning_fn=buffer_pruning_fn,
                )
                torch.cuda.synchronize(device)
                t_infer_end = time.perf_counter()
                seq_infer_time = t_infer_end - t_infer_start
                seq_n_frames = len(views)
                seq_fps = seq_n_frames / seq_infer_time if seq_infer_time > 0 else 0
                seq_peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
                total_frames += seq_n_frames
                total_infer_time += seq_infer_time

                # Save raw relative poses if requested
                if getattr(args, 'save_relative_poses', False):
                    relpose_path = f"{save_dir}/{seq}/pred_relative_poses.csv"
                    _save_raw_relative_poses(
                        outputs, relpose_path,
                        revisit=args.revisit,
                        reset_interval=args.reset_interval,
                    )

                # Determine use_relative_pose setting
                if args.use_relative_pose and args.no_relative_pose:
                    print("Warning: Both --use_relative_pose and --no_relative_pose specified. --no_relative_pose takes precedence.")
                    use_rel = False
                elif args.no_relative_pose:
                    use_rel = False
                elif args.use_relative_pose:
                    use_rel = True
                else:
                    use_rel = None  # auto-detect

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(
                    outputs, revisit=args.revisit, solve_pose=args.solve_pose,
                    use_relative_pose=use_rel, skip_pgo=args.skip_pgo,
                    pgo_sigma_rot=args.pgo_sigma_rot,
                    pgo_sigma_trans=args.pgo_sigma_trans,
                    pgo_mode=args.pgo_mode,
                    pgo_adaptive_sigma=args.pgo_adaptive_sigma,
                    pgo_warmstart=args.pgo_warmstart,
                    kf_edges_only=args.kf_edges_only,
                    rot_gap_power=args.rot_gap_power,
                    trans_gap_power=args.trans_gap_power,
                    use_pgo1_poses=args.use_pgo1_poses,
                    use_online_pgo=args.use_online_pgo,
                )

                # Extract timestamps from image filenames for TUM format
                traj_format = metadata.get("traj_format", None)
                if traj_format == "tum":
                    img_timestamps = [float(os.path.splitext(os.path.basename(f))[0]) for f in filelist]
                else:
                    img_timestamps = None
                pred_traj = get_tum_poses(pr_poses, timestamps=img_timestamps)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                save_intrinsics(cam_dict, f"{save_dir}/{seq}/pred_intrinsics.txt")
                # save_depth_maps(pts3ds_self,f'{save_dir}/{seq}', conf_self=conf_self)
                # save_conf_maps(conf_self,f'{save_dir}/{seq}')
                # save_rgb_imgs(colors,f'{save_dir}/{seq}')

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    _correct_scale = not args.no_correct_scale
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                        correct_scale=_correct_scale,
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png",
                        correct_scale=_correct_scale,
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)
                fps_list.append(seq_fps)
                mem_list.append(seq_peak_mem)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}, FPS: {seq_fps:.2f} ({seq_n_frames} frames / {seq_infer_time:.2f}s), Peak Mem: {seq_peak_mem:.0f}MB\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e) or "IndeterminantLinearSystemException" in str(type(e).__name__) or "Indeterminant" in str(e):
                    # Handle numerical issues from GTSAM/eigensolve
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Numerical error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
    avg_fps = total_frames / total_infer_time if total_infer_time > 0 else 0
    avg_peak_mem = sum(mem_list) / len(mem_list) if mem_list else 0

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )
            f.write(
                f"Average FPS: {avg_fps:.2f} ({total_frames} frames / {total_infer_time:.2f}s), Average Peak Mem: {avg_peak_mem:.0f}MB\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.geometry import weighted_procrustes, geotrf

    args.full_seq = False
    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 1, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        # weighted procrustes
        c2w = weighted_procrustes(
            pts3ds_self,
            pts3ds_other,
            torch.log(conf_self) * torch.log(conf_other),
            use_weights=True,
            return_T=True,
        )
        return c2w, focal, pp.reshape(B, 2)

    def _rotation_matrix_to_quaternion(R):
        """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return qw, qx, qy, qz

    def _save_raw_relative_poses(outputs, save_path, revisit=1, reset_interval=1000000):
        """Extract and save raw model-predicted relative poses to CSV.

        Each row: frame_idx, ref_idx, tx, ty, tz, qw, qx, qy, qz
        """
        preds = outputs["pred"]
        views = outputs["views"]

        # Handle revisit
        valid_length = len(preds) // max(revisit, 1)
        preds = preds[-valid_length:]
        views = views[-valid_length:]

        # Remove overlap frames (same logic as prepare_output)
        has_reset_key = len(views) > 0 and "reset" in views[0]
        if has_reset_key:
            reset_mask = torch.cat([v["reset"] for v in views], 0)
            shifted = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], 0)
            if shifted.any():
                preds = [p for p, m in zip(preds, shifted) if not m]
                views = [v for v, m in zip(views, shifted) if not m]

        rows = []
        for i, pred in enumerate(preds):
            rel_poses = pred.get("relative_poses")        # (B, N, 4, 4) or None
            ref_indices = pred.get("ref_frame_indices")    # List[int] or None
            if rel_poses is None or ref_indices is None:
                continue
            rel_poses_np = rel_poses[0].cpu().float().numpy()  # (N, 4, 4)
            N = rel_poses_np.shape[0]
            for k, ref_idx in enumerate(ref_indices):
                if k >= N:
                    break
                T = rel_poses_np[k]
                tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
                R = T[:3, :3]
                qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
                rows.append([i, ref_idx, tx, ty, tz, qw, qx, qy, qz])

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "ref_idx", "tx", "ty", "tz", "qw", "qx", "qy", "qz"])
            writer.writerows(rows)
        print(f"  Saved {len(rows)} relative poses to {save_path}")

    def _build_view_image_paths(img_paths, reset_interval):
        """Build view-indexed image path list that accounts for overlap frames.
        prepare_input inserts an overlap frame after every reset_interval frames.
        This maps view_idx -> original image path."""
        result = []
        for i, path in enumerate(img_paths):
            result.append(path)
            if (i + 1) % reset_interval == 0:
                result.append(path)  # overlap frame uses same image
        return result

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
        reset_interval=1000000,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
                }
                views.append(view)
                # Insert overlap frame after reset for state re-initialization
                if (i + 1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1, solve_pose=False, use_relative_pose=None, skip_pgo=False,
                        pgo_sigma_rot=None, pgo_sigma_trans=None, pgo_mode=None,
                        pgo_adaptive_sigma=False, pgo_warmstart=False,
                        kf_edges_only=False, rot_gap_power=None, trans_gap_power=None,
                        use_pgo1_poses=False, use_online_pgo=False):
        from dust3r.inference import accumulate_poses

        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        # Remove overlap frames inserted after reset
        has_reset_key = len(outputs["views"]) > 0 and "reset" in outputs["views"][0]
        if has_reset_key:
            reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
            # shifted_reset_mask marks overlap frames (frame after reset=True)
            shifted_reset_mask = torch.cat(
                [torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0
            )
            if shifted_reset_mask.any():
                outputs["pred"] = [
                    pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask
                ]
                outputs["views"] = [
                    view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask
                ]

        pts3ds_self = [
            output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
        ]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]

        if solve_pose:
            pr_poses, focal, pp = recover_cam_params(
                torch.cat(pts3ds_self, 0),
                torch.cat(pts3ds_other, 0),
                torch.cat(conf_self, 0),
                torch.cat(conf_other, 0),
            )
            pts3ds_self = torch.cat(pts3ds_self, 0)
        else:
            pts3ds_self = torch.cat(pts3ds_self, 0)
            pr_poses = accumulate_poses(
                outputs["pred"],
                views=outputs["views"],
                use_relative_pose=use_relative_pose,
                skip_pgo=skip_pgo,
                pgo_sigma_rot=pgo_sigma_rot,
                pgo_sigma_trans=pgo_sigma_trans,
                pgo_mode=pgo_mode,
                pgo_adaptive_sigma=pgo_adaptive_sigma,
                pgo_warmstart=pgo_warmstart,
                kf_edges_only=kf_edges_only,
                rot_gap_power=rot_gap_power,
                trans_gap_power=trans_gap_power,
                use_pgo1_poses=use_pgo1_poses,
                use_online_pgo=use_online_pgo,
            )
            pr_poses = torch.cat(pr_poses, 0)

            B, H, W, _ = pts3ds_self.shape
            pp = (
                torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
                .float()
                .repeat(B, 1)
                .reshape(B, 2)
            )
            focal = estimate_focal_knowing_depth(
                pts3ds_self, pp, focal_mode="weiszfeld"
            )

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            colors,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    model = ARCroco3DStereo.from_pretrained(args.weights)
    model.config.model_update_type = args.model_update_type
    model.config.ttt3r_bias = args.ttt3r_bias
    model.config.ttt3r_mode = args.ttt3r_mode
    model.config.ttt3r_scale = args.ttt3r_scale
    eval_pose_estimation(args, model, save_dir=args.output_dir)
