#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
3D Point Cloud Inference and Visualization Script using STream3R

This script performs inference using the STream3R model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Supports PGO (Pose Graph
Optimization), keyframe selection, loop closure, and reset_interval for
large-scale scenes.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                   [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]
                   [--mode MODE] [--use_streaming] [--use_rel_pose]

Examples:
    # Original STream3R (pose_enc)
    python demo.py \
    --model_path weights/stream3r/model.pt \
    --seq_path data/processed_vkitti/Scene01/clone/Camera_0 \
    --use_streaming --mode window \
    --downsample_factor 100

    # Relative pose with PGO
    python demo.py --model_path weights/scal3r_stream3r/model.pt \
        --seq_path data/processed_vkitti/Scene01/clone/Camera_0 \
        --use_streaming --mode window \
        --kf_window 8 --max_ref_frames 8 --nkf_buffer_size 8 \
        --num_init_frames 2 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3 \
        --reset_interval 10 \
        --downsample_factor 100

    # Long sequence with reset interval
    python demo.py --model_path weights/scal3r_stream3r/model.pt \
      --seq_path data/kitti_data/sequences/00/image_2 \
      --use_streaming --mode window \
      --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 \
      --num_init_frames 2 --reset_interval 10 \
      --loop_closure --loop_similarity_threshold 0.80 --loop_temporal_gap 200 \
      --downsample_factor 40 --vis_threshold 1.5 --vis_stride 10
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
import imageio.v2 as iio

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using STream3R."
    )

    # ── Core ──
    parser.add_argument(
        "--model_path", type=str, default="yslan/STream3R",
        help="Path to the pretrained model checkpoint or Hugging Face model name.",
    )
    parser.add_argument(
        "--seq_path", type=str, default="",
        help="Path to the directory containing the image sequence or video file.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size", type=int, default=518,
        help="Target image size for preprocessing (default: 518).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./demo_tmp",
        help="Output directory for saving results.",
    )
    parser.add_argument(
        "--mode", type=str, default="causal", choices=["causal", "window", "full"],
        help="Inference mode: 'causal', 'window', or 'full'.",
    )
    parser.add_argument(
        "--use_streaming", action="store_true",
        help="Use streaming inference with KV cache.",
    )

    # ── Frame control ──
    parser.add_argument(
        "--max_images", type=int, default=100000,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--frame_interval", type=int, default=1,
        help="Frame interval for reading images/video (1 = every frame).",
    )
    parser.add_argument(
        "--reset_interval", type=int, default=1000000,
        help="Reset streaming state every N frames for long sequences.",
    )
    parser.add_argument(
        "--img_filter", type=str, default=None,
        help="Suffix filter for image files (e.g. '_rgb.jpg'). Auto-detected if not set.",
    )

    # ── Visualization ──
    parser.add_argument(
        "--vis_threshold", type=float, default=1.5,
        help="Confidence threshold for point cloud visualization (1 to INF).",
    )
    parser.add_argument(
        "--downsample_factor", type=int, default=1,
        help="Downsample factor for point cloud visualization.",
    )
    parser.add_argument(
        "--vis_stride", type=int, default=1,
        help="Frame stride for visualization (e.g. 10 = show every 10th frame). "
             "Inference runs on all frames for correct PGO, only visualization is subsampled.",
    )

    # ── Camera color ──
    parser.add_argument(
        "--cam_color_mode",
        type=str,
        default=None,
        choices=["split", "rainbow"],
        help="Camera pose color mode: 'split' colors by frame ID threshold, 'rainbow' colors by time. Default: keyframe-based coloring.",
    )
    parser.add_argument(
        "--cam_color_split_frame",
        type=int,
        default=0,
        help="Frame ID for split mode: frames before this are blue, after are orange.",
    )

    # ── Relative pose ──
    parser.add_argument(
        "--use_rel_pose", action="store_true",
        help="Use relative pose accumulation with PGO (primary switch).",
    )
    parser.add_argument(
        "--use_rel_pose_prompt", action="store_true",
        help="(Backward compat) Alias for --use_rel_pose.",
    )
    parser.add_argument(
        "--ref_feat_type", type=str, default="img_feat",
        choices=["img_feat", "camera_token"],
        help="Reference feature type for rel_pose conditioning.",
    )
    parser.add_argument(
        "--rel_pose_global_only", action="store_true",
        help="Only use global-path features (1024d) for rel_pose decoder.",
    )
    parser.add_argument(
        "--max_ref_frames", type=int, default=4,
        help="Max reference frames for multi-ref pose.",
    )

    # ── PGO ──
    parser.add_argument(
        "--skip_pgo", action="store_true",
        help="Skip PGO, use chain accumulation only.",
    )
    parser.add_argument(
        "--kf_window", type=int, default=4,
        help="Number of keyframes to keep in buffer.",
    )
    parser.add_argument(
        "--nkf_buffer_size", type=int, default=0,
        help="Number of non-keyframes to keep in buffer.",
    )
    parser.add_argument(
        "--num_init_frames", type=int, default=5,
        help="Number of initial frames treated as keyframes.",
    )
    parser.add_argument(
        "--pgo_sigma_rot", type=float, default=None,
        help="Override base sigma for rotation in PGO (default: 0.5).",
    )
    parser.add_argument(
        "--pgo_sigma_trans", type=float, default=None,
        help="Override base sigma for translation in PGO (default: 0.5).",
    )
    parser.add_argument(
        "--pgo_mode", type=str, default=None,
        choices=["huber", "dcs", "cauchy", "tukey"],
        help="PGO robust kernel mode.",
    )
    parser.add_argument(
        "--pgo_max_edges", type=int, default=0,
        help="Max PGO constraints per frame (0=unlimited).",
    )
    parser.add_argument(
        "--final_batch_opt", action="store_true",
        help="Run final Levenberg-Marquardt batch optimization after iSAM2.",
    )
    parser.add_argument(
        "--kf_only_cache", action="store_true",
        help="Only keyframes update KV cache.",
    )
    parser.add_argument(
        "--use_global_pose_init", action="store_true",
        help="Use pose_enc (global pose) as PGO initialization.",
    )
    parser.add_argument(
        "--global_pose_prior_sigma", type=float, default=None,
        help="If set, add PriorFactorPose3 with this sigma for global pose.",
    )

    # ── Loop closure ──
    parser.add_argument(
        "--loop_closure", action="store_true",
        help="Enable online loop closure detection.",
    )
    parser.add_argument(
        "--loop_similarity_threshold", type=float, default=0.85,
        help="Cosine similarity threshold for loop detection.",
    )
    parser.add_argument(
        "--loop_temporal_gap", type=int, default=300,
        help="Minimum frame gap for loop closure candidates.",
    )
    parser.add_argument(
        "--loop_max_per_frame", type=int, default=1,
        help="Max loop closure frames to inject per keyframe.",
    )
    parser.add_argument(
        "--loop_nms_window", type=int, default=50,
        help="NMS window for loop closure suppression.",
    )
    parser.add_argument(
        "--loop_max_translation", type=float, default=20.0,
        help="Reject loop edges with predicted translation > this (meters).",
    )
    parser.add_argument(
        "--loop_sigma_scale", type=float, default=1.0,
        help="Sigma scale for loop closure identity constraint.",
    )

    return parser.parse_args()


def parse_seq_path(p, frame_interval=1, img_filter=None):
    """Parse sequence path and return image paths."""
    if os.path.isdir(p):
        # Auto-detect img_filter: if directory has *_rgb.jpg files, only use those
        if img_filter is None:
            rgb_files = glob.glob(f"{p}/*_rgb.jpg")
            if rgb_files:
                img_filter = "_rgb.jpg"
                print(f"Auto-detected img_filter='{img_filter}' (found {len(rgb_files)} RGB files)")

        if img_filter is not None:
            img_paths = sorted(glob.glob(f"{p}/*{img_filter}"))
        else:
            img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            img_paths = sorted(glob.glob(f"{p}/*"))
            img_paths = [x for x in img_paths if os.path.splitext(x)[1].lower() in img_extensions]
        # Apply frame_interval
        img_paths = img_paths[::frame_interval]
        tmpdirname = None
    else:
        # Video file
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, "
            f"Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def _build_view_image_paths(img_paths, reset_interval):
    """Build view-indexed image path list that accounts for overlap frames.
    reset logic inserts an overlap frame after every reset_interval frames.
    This maps view_idx -> original image path."""
    result = []
    for i, path in enumerate(img_paths):
        result.append(path)
        if (i + 1) % reset_interval == 0:
            result.append(path)  # overlap frame uses same image
    return result


def prepare_output(predictions, outdir, use_relative_pose=False, pgo_poses=None, save=True):
    """
    Process STream3R predictions to generate point clouds and camera parameters.

    Args:
        predictions (dict): STream3R model predictions.
        outdir (str): Output directory for saving results.
        use_relative_pose (bool): If True, use accumulated relative poses.
        pgo_poses (list, optional): PGO-optimized c2w poses (list of [4,4] or [1,4,4] tensors).

    Returns:
        tuple: (points, colors, confidence, cam_dict, rel_pose_dict)
    """
    from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
    from stream3r.models.components.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

    # Extract predictions
    depth = predictions["depth"]  # [B, S, H, W, 1]
    depth_conf = predictions["depth_conf"]  # [B, S, H, W]
    pose_enc = predictions["pose_enc"]  # [B, S, 9]
    images = predictions["images"]  # [B, S, 3, H, W]

    # Handle streaming vs batch case
    is_streaming = len(images.shape) == 5
    if is_streaming:
        B, S = images.shape[0], images.shape[1]
        depth = depth.view(B * S, *depth.shape[2:])
        depth_conf = depth_conf.view(B * S, *depth_conf.shape[2:])
        images = images.view(B * S, *images.shape[2:])
        N = B * S
    else:
        B = images.shape[0]
        N = B

    depth = depth.cpu()
    depth_conf = depth_conf.cpu()
    images = images.cpu()
    H, W = images.shape[2], images.shape[3]
    image_size_hw = (H, W)

    # ── Determine c2w poses ──
    if pgo_poses is not None:
        # PGO-optimized c2w poses
        print("Using PGO-optimized poses for visualization...")
        c2w_list = []
        for p in pgo_poses:
            if isinstance(p, torch.Tensor):
                if p.dim() == 3:
                    p = p.squeeze(0)
                c2w_list.append(p.cpu().numpy())
            else:
                c2w_list.append(np.array(p, dtype=np.float32))
        se3_c2w = np.stack(c2w_list, axis=0)  # [N, 4, 4]
        w2c_poses = closed_form_inverse_se3(se3_c2w)
        extrinsics_np = w2c_poses[:, :3, :4]

        # Intrinsics from pose_enc
        pose_enc_cpu = pose_enc.cpu()
        if is_streaming:
            pose_enc_cpu = pose_enc_cpu.view(N, pose_enc_cpu.shape[-1])
        if len(pose_enc_cpu.shape) == 2:
            pose_enc_cpu = pose_enc_cpu.unsqueeze(0)
        _, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_cpu, image_size_hw=image_size_hw,
            pose_encoding_type="absT_quaR_FoV", build_intrinsics=True
        )
        if intrinsics.shape[0] == 1:
            intrinsics = intrinsics.squeeze(0)
        elif intrinsics.shape[1] == 1:
            intrinsics = intrinsics.squeeze(1)
        intrinsics_np = intrinsics.numpy()
        print(f"PGO: {N} optimized camera poses")

    elif use_relative_pose:
        # Chain accumulation from rel_pose predictions
        has_rel_pose = "rel_pose" in predictions and predictions["rel_pose"] is not None
        if not has_rel_pose:
            print("Warning: use_relative_pose=True but rel_pose not found. Falling back to pose_enc.")
            return prepare_output(predictions, outdir, use_relative_pose=False)

        print("Using accumulated relative poses for visualization...")
        rel_pose = predictions["rel_pose"]

        # Handle streaming list format vs batch tensor format
        if isinstance(rel_pose, list):
            # Streaming: list of per-frame dicts
            se3_c2w = _chain_accumulate_streaming(rel_pose)
        else:
            # Batch: single dict with tensors
            rel_trans = rel_pose["rel_trans"].cpu().numpy()
            rel_rot = rel_pose["rel_rot"].cpu().numpy()
            if rel_trans.shape[0] == 1:
                rel_trans = rel_trans.squeeze(0)
                rel_rot = rel_rot.squeeze(0)
            se3_c2w = _chain_accumulate_batch(rel_trans, rel_rot)

        w2c_poses = closed_form_inverse_se3(se3_c2w)
        extrinsics_np = w2c_poses[:, :3, :4]

        # Intrinsics from pose_enc
        pose_enc_cpu = pose_enc.cpu()
        if is_streaming:
            pose_enc_cpu = pose_enc_cpu.view(N, pose_enc_cpu.shape[-1])
        if len(pose_enc_cpu.shape) == 2:
            pose_enc_cpu = pose_enc_cpu.unsqueeze(0)
        _, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_cpu, image_size_hw=image_size_hw,
            pose_encoding_type="absT_quaR_FoV", build_intrinsics=True
        )
        if intrinsics.shape[0] == 1:
            intrinsics = intrinsics.squeeze(0)
        elif intrinsics.shape[1] == 1:
            intrinsics = intrinsics.squeeze(1)
        intrinsics_np = intrinsics.numpy()
        print(f"Accumulated {N} camera poses from relative poses")

    else:
        # Direct pose_enc decoding
        print("Using direct pose_enc for visualization...")
        pose_enc_cpu = pose_enc.cpu()
        if is_streaming:
            pose_enc_cpu = pose_enc_cpu.view(N, pose_enc_cpu.shape[-1])
        if len(pose_enc_cpu.shape) == 2:
            pose_enc_cpu = pose_enc_cpu.unsqueeze(0)
        elif len(pose_enc_cpu.shape) == 3:
            pass
        else:
            pose_enc_cpu = pose_enc_cpu.unsqueeze(1)

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_cpu, image_size_hw=image_size_hw,
            pose_encoding_type="absT_quaR_FoV", build_intrinsics=True
        )
        if extrinsics.shape[0] == 1:
            extrinsics = extrinsics.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
        elif extrinsics.shape[1] == 1:
            extrinsics = extrinsics.squeeze(1)
            intrinsics = intrinsics.squeeze(1)

        extrinsics_np = extrinsics.numpy()
        intrinsics_np = intrinsics.numpy()

        se3_w2c = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
        se3_w2c[:, :3, :4] = extrinsics_np
        se3_c2w = closed_form_inverse_se3(se3_w2c)

    # ── Generate world points from depth ──
    print("Computing world points from depth map...")
    depth_np = depth.numpy()
    world_points = unproject_depth_map_to_point_map(depth_np, extrinsics_np, intrinsics_np)

    R_c2w = se3_c2w[:, :3, :3]
    t_c2w = se3_c2w[:, :3, 3]

    pts3ds_other = [torch.from_numpy(world_points[i]) for i in range(N)]
    R_c2w_t = torch.from_numpy(R_c2w)
    t_c2w_t = torch.from_numpy(t_c2w)

    focal = torch.from_numpy(intrinsics_np[:, 0, 0])
    pp = torch.from_numpy(np.stack([intrinsics_np[:, 0, 2], intrinsics_np[:, 1, 2]], axis=1))

    # images from load_and_preprocess_images are in [0, 1] range
    colors = images.permute(0, 2, 3, 1).clamp(0, 1)
    colors = [colors[i:i+1] for i in range(N)]

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w_t.numpy(),
        "t": t_c2w_t.numpy(),
    }

    # ── Save outputs ──
    rel_pose_dict = None
    if save:
        os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)

        for f_id in range(N):
            depth_save = depth[f_id].numpy()
            conf_save = depth_conf[f_id].numpy()
            color_save = colors[f_id].numpy().squeeze(0)
            c2w_save = se3_c2w[f_id]
            intrins_save = intrinsics_np[f_id]

            np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth_save)
            np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf_save)
            iio.imwrite(
                os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color_save * 255).astype(np.uint8),
            )
            np.savez(
                os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                pose=c2w_save, intrinsics=intrins_save,
            )

        # ── Save rel_pose if available ──
        has_rel_pose = "rel_pose" in predictions and predictions["rel_pose"] is not None
        if has_rel_pose:
            rel_pose = predictions["rel_pose"]
            if isinstance(rel_pose, list):
                all_trans, all_rot = [], []
                for rp in rel_pose:
                    if rp is not None and "rel_trans" in rp:
                        all_trans.append(rp["rel_trans"].cpu().numpy())
                        all_rot.append(rp["rel_rot"].cpu().numpy())
                trans_arr = np.empty(len(all_trans), dtype=object)
                rot_arr = np.empty(len(all_rot), dtype=object)
                for i in range(len(all_trans)):
                    trans_arr[i] = all_trans[i]
                    rot_arr[i] = all_rot[i]
                rel_pose_dict = {"rel_trans_list": trans_arr, "rel_rot_list": rot_arr}
            else:
                rel_trans = rel_pose["rel_trans"].cpu().numpy()
                rel_rot = rel_pose["rel_rot"].cpu().numpy()
                if rel_trans.shape[0] == 1:
                    rel_trans = rel_trans.squeeze(0)
                    rel_rot = rel_rot.squeeze(0)
                rel_pose_dict = {"rel_trans": rel_trans, "rel_rot": rel_rot}

            os.makedirs(os.path.join(outdir, "rel_pose"), exist_ok=True)
            np.savez(os.path.join(outdir, "rel_pose", "rel_poses.npz"), allow_pickle=True, **rel_pose_dict)
            print(f"Relative pose predictions saved to {os.path.join(outdir, 'rel_pose')}")

    return pts3ds_other, colors, depth_conf, cam_dict, rel_pose_dict


def _chain_accumulate_batch(rel_trans, rel_rot):
    """Chain accumulation for batch format. rel_trans: [S, 3], rel_rot: [S, 3, 3]."""
    S = rel_trans.shape[0]
    c2w_poses = np.zeros((S, 4, 4), dtype=np.float32)
    c2w_poses[0] = np.eye(4, dtype=np.float32)

    for i in range(1, S):
        T_rel_inv = np.eye(4, dtype=np.float32)
        T_rel_inv[:3, :3] = rel_rot[i].T
        T_rel_inv[:3, 3] = -rel_rot[i].T @ rel_trans[i]
        c2w_poses[i] = c2w_poses[i-1] @ T_rel_inv

    return c2w_poses


def _chain_accumulate_streaming(rel_pose_list):
    """Chain accumulation for streaming format: list of per-frame rel_pose dicts."""
    N = len(rel_pose_list)
    c2w_poses = np.zeros((N, 4, 4), dtype=np.float32)
    c2w_poses[0] = np.eye(4, dtype=np.float32)
    c2w_history = {0: c2w_poses[0]}

    for fi in range(1, N):
        rp = rel_pose_list[fi]
        if rp is None or "rel_trans" not in rp:
            c2w_poses[fi] = c2w_history.get(fi - 1, np.eye(4, dtype=np.float32))
            c2w_history[fi] = c2w_poses[fi]
            continue

        rt = rp["rel_trans"].cpu().numpy()  # [B, 1, K_i, 3]
        rr = rp["rel_rot"].cpu().numpy()    # [B, 1, K_i, 3, 3]
        K_i = rt.shape[2]

        c2w_i = None
        for k in range(K_i):
            ref_idx = fi - k - 1
            if ref_idx < 0 or ref_idx not in c2w_history:
                continue
            R_inv = rr[0, 0, k].T
            t_inv = -R_inv @ rt[0, 0, k]
            T_inv = np.eye(4, dtype=np.float32)
            T_inv[:3, :3] = R_inv
            T_inv[:3, 3] = t_inv
            c2w_i = c2w_history[ref_idx] @ T_inv
            break

        if c2w_i is None:
            c2w_i = c2w_history.get(fi - 1, np.eye(4, dtype=np.float32)).copy()

        c2w_poses[fi] = c2w_i
        c2w_history[fi] = c2w_i

    return c2w_poses


def _remove_overlap_predictions(predictions, overlap_indices, total_frames):
    """Remove overlap frame predictions (inserted by reset_interval)."""
    if not overlap_indices:
        return predictions

    keep_mask = [True] * total_frames
    for oi in overlap_indices:
        if oi < total_frames:
            keep_mask[oi] = False
    keep_indices = [i for i, m in enumerate(keep_mask) if m]

    for k in ['pose_enc', 'depth', 'depth_conf', 'images', 'world_points', 'world_points_conf']:
        if k in predictions and predictions[k] is not None:
            predictions[k] = predictions[k][:, keep_indices]

    if 'rel_pose' in predictions and isinstance(predictions['rel_pose'], list):
        predictions['rel_pose'] = [
            rp for rp, m in zip(predictions['rel_pose'], keep_mask) if m
        ]

    return predictions


def run_inference(args):
    """Execute the full inference and visualization pipeline."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    from stream3r.models.stream3r import STream3R
    from stream3r.stream_session import StreamSession
    from stream3r.models.components.utils.load_fn import load_and_preprocess_images

    # ── Resolve use_rel_pose ──
    use_rel_pose = args.use_rel_pose or args.use_rel_pose_prompt

    # ── Parse image paths ──
    img_paths, tmpdirname = parse_seq_path(
        args.seq_path, frame_interval=args.frame_interval, img_filter=args.img_filter
    )
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    if len(img_paths) > args.max_images:
        print(f"Found {len(img_paths)} images, limiting to {args.max_images}.")
        img_paths = img_paths[:args.max_images]
    else:
        print(f"Found {len(img_paths)} images in {args.seq_path}.")

    # ── Load model ──
    print(f"Loading STream3R model from {args.model_path}...")
    if args.model_path.endswith(('.pt', '.pth', '.bin')):
        raw = torch.load(args.model_path, map_location=device, weights_only=False)

        # New format: {'state_dict': ..., 'config': ...}; old format: plain state_dict
        if isinstance(raw, dict) and 'state_dict' in raw and 'config' in raw:
            checkpoint = raw['state_dict']
            ckpt_config = raw['config']
            print(f"Loaded config from checkpoint: {ckpt_config}")
        else:
            checkpoint = raw
            ckpt_config = {}

        # Handle Lightning checkpoint format (keys with 'net.' prefix)
        if any(k.startswith('net.') for k in checkpoint.keys()):
            checkpoint = {k.replace('net.', ''): v for k, v in checkpoint.items() if k.startswith('net.')}

        # Auto-detect rel_pose from checkpoint
        has_rel_pose_keys = any("rel_pose" in k for k in checkpoint.keys())
        model_use_rel_pose = use_rel_pose or has_rel_pose_keys

        # Model config: checkpoint config > CLI override > defaults
        num_rel_pose_tokens = ckpt_config.get('num_rel_pose_tokens', 4)
        rel_pose_token_key = "aggregator.rel_pose_token"
        if rel_pose_token_key in checkpoint:
            num_rel_pose_tokens = checkpoint[rel_pose_token_key].shape[1]
            print(f"Detected num_rel_pose_tokens={num_rel_pose_tokens} from checkpoint.")

        ref_feat_type = ckpt_config.get('ref_feat_type', args.ref_feat_type)
        rel_pose_global_only = ckpt_config.get('rel_pose_global_only', args.rel_pose_global_only)

        model = STream3R(
            use_rel_pose_prompt=model_use_rel_pose,
            num_rel_pose_tokens=num_rel_pose_tokens,
            ref_feat_type=ref_feat_type,
            rel_pose_global_only=rel_pose_global_only,
        )
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f"Model: use_rel_pose={model_use_rel_pose}, num_tokens={num_rel_pose_tokens}, "
              f"ref_feat_type={ref_feat_type}, global_only={rel_pose_global_only}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        if model_use_rel_pose:
            model.aggregator.max_ref_frames = args.max_ref_frames
            print(f"  Inference max_ref_frames={args.max_ref_frames}")

        model = model.to(device)
        # Update use_rel_pose based on what was actually loaded
        use_rel_pose = model_use_rel_pose
    else:
        # Load from HuggingFace
        model = STream3R.from_pretrained(args.model_path).to(device)
        if use_rel_pose:
            print("Warning: Relative pose requested but HuggingFace model may not support it.")
            use_rel_pose = False
    model.eval()

    # ── Load images ──
    print("Loading and preprocessing images...")
    images = load_and_preprocess_images(img_paths, mode="crop").to(device)
    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")

    # Clean up temp dir now (images are in memory)
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # ── Run inference ──
    print(f"Running inference with mode: {args.mode}")
    start_time = time.time()

    keyframe_indices = set()
    pgo_poses = None
    overlap_indices = []

    if args.use_streaming:
        if args.mode == "full":
            print("Warning: Streaming mode does not support 'full'. Switching to 'causal'.")
            args.mode = "causal"

        # ── Build PGO config ──
        no_pgo = args.skip_pgo
        use_pgo = use_rel_pose and not no_pgo
        use_pgo_session = use_pgo or args.kf_only_cache

        pgo_config = None
        if use_pgo_session:
            pgo_config = dict(
                kf_pgo=use_pgo_session,
                kf_window=args.kf_window,
                nkf_buffer_size=args.nkf_buffer_size,
                num_init_frames=args.num_init_frames,
                kf_only_cache=args.kf_only_cache,
            )
            if args.pgo_sigma_rot is not None:
                pgo_config['pgo_sigma_rot'] = args.pgo_sigma_rot
            if args.pgo_sigma_trans is not None:
                pgo_config['pgo_sigma_trans'] = args.pgo_sigma_trans
            if args.pgo_max_edges > 0:
                pgo_config['pgo_max_edges'] = args.pgo_max_edges
            if args.pgo_mode is not None:
                pgo_config['pgo_mode'] = args.pgo_mode
            if args.use_global_pose_init:
                pgo_config['use_global_pose_init'] = True
                if args.global_pose_prior_sigma is not None:
                    pgo_config['global_pose_prior_sigma'] = args.global_pose_prior_sigma

            # Loop closure
            if args.loop_closure:
                try:
                    from stream3r.utils.loop_closure import OnlineLoopDetector
                    loop_detector = OnlineLoopDetector(
                        device=device,
                        similarity_threshold=args.loop_similarity_threshold,
                        temporal_gap=args.loop_temporal_gap,
                        max_loops_per_frame=args.loop_max_per_frame,
                        nms_window=args.loop_nms_window,
                    )
                    loop_detector.reset()
                    pgo_config['loop_detector'] = loop_detector
                    pgo_config['loop_image_paths'] = _build_view_image_paths(img_paths, args.reset_interval)
                    pgo_config['loop_max_translation'] = args.loop_max_translation
                    pgo_config['loop_sigma_scale'] = args.loop_sigma_scale
                    print(f"Loop closure enabled: threshold={args.loop_similarity_threshold}, "
                          f"gap={args.loop_temporal_gap}")
                except ImportError as e:
                    print(f"Warning: Loop closure not available ({e}). Continuing without it.")

        # ── Create session ──
        print(f"Streaming inference: use_pgo={use_pgo}, kf_window={args.kf_window}, "
              f"nkf_buffer_size={args.nkf_buffer_size}")
        session = StreamSession(model, mode=args.mode, use_pgo=use_pgo_session, pgo_config=pgo_config)

        # ── Process frames ──
        num_frames = images.shape[0]
        with torch.no_grad():
                for i in range(num_frames):
                    image = images[i:i+1]
                    predictions = session.forward_stream(image)

                    # Reset streaming state after every reset_interval frames
                    if (i + 1) % args.reset_interval == 0 and (i + 1) < num_frames:
                        print(f"  Resetting streaming state at frame {i+1}")
                        session.reset_streaming_state()
                        # Re-feed current frame as overlap (first frame of new segment)
                        predictions = session.forward_stream(image)
                        overlap_indices.append(session.frame_count - 1)

        # Print keyframe stats
        if use_pgo_session and hasattr(session, 'on_frame_processed'):
            if hasattr(session.on_frame_processed, 'print_kf_stats'):
                session.on_frame_processed.print_kf_stats()

        keyframe_indices = getattr(session, 'keyframe_indices', set()) if use_pgo_session else set()

        # Get PGO poses BEFORE removing overlaps (finalize needs all frames)
        if use_pgo and use_rel_pose:
            pgo_poses = session.get_pgo_poses()
            if pgo_poses is not None:
                # Remove overlap frames from PGO poses
                if overlap_indices:
                    keep_mask = [True] * session.frame_count
                    for oi in overlap_indices:
                        keep_mask[oi] = False
                    pgo_poses = [p for p, keep in zip(pgo_poses, keep_mask) if keep]
                print(f"  PGO: {len(pgo_poses)} optimized poses")

        predictions = session.get_all_predictions()

        # Remove overlap frame predictions
        if overlap_indices:
            predictions = _remove_overlap_predictions(predictions, overlap_indices, session.frame_count)

        if use_pgo_session:
            n_kf = len(keyframe_indices)
            total_f = session.frame_count - len(overlap_indices)
            print(f"  Keyframes: {n_kf}/{total_f} frames")

        session.clear()
    else:
        # Batch inference
        if use_rel_pose and not args.skip_pgo:
            print("Warning: PGO is only supported in streaming mode (--use_streaming). "
                  "Using chain accumulation.")
        print("Using batch inference...")
        with torch.no_grad():
            predictions = model(images, mode=args.mode)

    total_time = time.time() - start_time
    per_frame_time = total_time / images.shape[0]
    print(f"Inference completed in {total_time:.2f}s ({per_frame_time:.2f}s/frame)")

    # ── Prepare output ──
    print("Preparing output for visualization...")

    # Debug: save PGO trajectory in TUM format for comparison with eval
    if pgo_poses is not None:
        from scipy.spatial.transform import Rotation as _Rot
        os.makedirs(args.output_dir, exist_ok=True)
        tum_path = os.path.join(args.output_dir, "pgo_traj_tum.txt")
        with open(tum_path, "w") as f:
            for i, p in enumerate(pgo_poses):
                c2w = p.squeeze(0).cpu().numpy() if isinstance(p, torch.Tensor) else np.array(p)
                t = c2w[:3, 3]
                qx, qy, qz, qw = _Rot.from_matrix(c2w[:3, :3]).as_quat()
                f.write(f"{i} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}\n")
        print(f"  Saved PGO trajectory to {tum_path} ({len(pgo_poses)} poses)")
        # Debug: print first/last 3 poses
        for idx in [0, 1, 2, len(pgo_poses)//2, -2, -1]:
            p = pgo_poses[idx].squeeze(0) if isinstance(pgo_poses[idx], torch.Tensor) else pgo_poses[idx]
            t_dbg = p[:3, 3] if isinstance(p, torch.Tensor) else p[:3, 3]
            print(f"  pose[{idx}] t={t_dbg}")

    # Also save pose_enc trajectory for comparison
    if 'pose_enc' in predictions:
        from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri as _pe2ei
        from stream3r.dust3r.utils.geometry import inv as _inv
        _pe = predictions["pose_enc"]
        _img_hw = predictions["images"].shape[-2:]
        _extr, _ = _pe2ei(_pe, _img_hw)
        os.makedirs(args.output_dir, exist_ok=True)
        pe_tum_path = os.path.join(args.output_dir, "pose_enc_traj_tum.txt")
        with open(pe_tum_path, "w") as f:
            for i in range(_extr.shape[1]):
                w2c_34 = _extr[0, i]
                w2c = torch.cat([w2c_34, torch.tensor([[0, 0, 0, 1]], device=w2c_34.device)], dim=0)
                c2w_pe = _inv(w2c).cpu().numpy()
                t = c2w_pe[:3, 3]
                from scipy.spatial.transform import Rotation as _Rot2
                qx, qy, qz, qw = _Rot2.from_matrix(c2w_pe[:3, :3]).as_quat()
                f.write(f"{i} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}\n")
        print(f"  Saved pose_enc trajectory to {pe_tum_path} ({_extr.shape[1]} poses)")

    if pgo_poses is not None:
        # PGO mode
        pts3ds_other, colors, conf, cam_dict, rel_pose_dict = prepare_output(
            predictions, args.output_dir, pgo_poses=pgo_poses
        )
    elif use_rel_pose:
        # Chain accumulation (--skip_pgo or batch mode)
        pts3ds_other, colors, conf, cam_dict, rel_pose_dict = prepare_output(
            predictions, args.output_dir, use_relative_pose=True
        )
    else:
        # Original pose_enc
        pts3ds_other, colors, conf, cam_dict, rel_pose_dict = prepare_output(
            predictions, args.output_dir, use_relative_pose=False
        )

    if rel_pose_dict is not None:
        print("Relative pose predictions saved.")

    # ── Visualization (apply vis_stride to reduce frame count) ──
    vis_stride = args.vis_stride
    if vis_stride > 1:
        n_total = len(pts3ds_other)
        vis_indices = list(range(0, n_total, vis_stride))
        pts3ds_other = [pts3ds_other[i] for i in vis_indices]
        colors = [colors[i] for i in vis_indices]
        conf = [conf[i] for i in vis_indices]
        cam_dict = {k: v[vis_indices] for k, v in cam_dict.items()}
        keyframe_indices = {i // vis_stride for i in keyframe_indices if i % vis_stride == 0}
        print(f"  vis_stride={vis_stride}: {n_total} -> {len(vis_indices)} frames for visualization")

    pts3ds_to_vis = [p.numpy() for p in pts3ds_other]
    colors_to_vis = [c.numpy() for c in colors]
    conf_to_vis = [c.numpy() for c in conf]
    edge_colors = [None] * len(pts3ds_to_vis)

    print("Launching point cloud viewer...")
    try:
        from viser_utils import PointCloudViewer
        viewer = PointCloudViewer(
            model,
            None,
            pts3ds_to_vis,
            colors_to_vis,
            conf_to_vis,
            cam_dict,
            device=device,
            edge_color_list=edge_colors,
            show_camera=True,
            show_camera_image=False,
            vis_threshold=args.vis_threshold,
            size=args.size,
            downsample_factor=args.downsample_factor,
            keyframe_indices=keyframe_indices,
            cam_color_mode=args.cam_color_mode,
            cam_color_split_frame=args.cam_color_split_frame,
        )
        viewer.run()
    except ImportError as e:
        print(f"PointCloudViewer not available: {e}")
        print(f"Results saved to: {args.output_dir}")
    except Exception as e:
        print(f"Error launching viewer: {e}")
        import traceback
        traceback.print_exc()
        print(f"Results saved to: {args.output_dir}")


def main():
    args = parse_args()
    if not args.seq_path:
        print("No inputs found! Please provide a sequence path using --seq_path.")
        return
    run_inference(args)


if __name__ == "__main__":
    main()
