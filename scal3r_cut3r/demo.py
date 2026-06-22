# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]
                            [--downsample_factor FACTOR] [--use_relative_pose] [--no_relative_pose]
                            [--frame_interval INTERVAL] [--update_interval INTERVAL]
                            [--max_frame MAX_FRAME]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path data/kitti_data/sequences/07/image_2 \
        --no_auto_keyframe \
        --downsample_factor 100 \
        --pts_brightness 0.5 \
        --cam_color_mode split --cam_color_split_frame 113 \
        --max_frame 245
        

    # Use relative pose accumulation (if available in model output)
    python demo.py --model_path src/checkpoints/scal3r/checkpoint-best.pth \
        --seq_path data/vkitti/Scene01/clone/Camera_0 \
        --use_relative_pose \
        --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 12 --no_kf_gate --reset_interval 10 \
        --downsample_factor 100

    python demo.py --model_path src/checkpoints/scal3r/checkpoint-best.pth \
        --seq_path data/kitti_data/sequences/07/image_2 \
        --use_relative_pose \
        --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 8 --no_kf_gate --reset_interval 10 --num_init_frames 2 \
        --downsample_factor 100 --cam_color_mode rainbow \
        --loop_closure --loop_temporal_gap 200 --loop_similarity_threshold 0.60

    # Force to use direct camera_pose (ignore relative_pose)
    python demo.py --model_path src/checkpoints/scal3r/checkpoint-best.pth \
        --seq_path examples/vkitti_scene01 --device cuda --no_relative_pose
"""

import argparse
import glob
import os
import random
import shutil
import tempfile
import time
from copy import deepcopy

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from add_ckpt_path import add_path_to_dust3r

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample factor for the point cloud viewer",
    )
    parser.add_argument(
        "--use_relative_pose",
        action="store_true",
        help="Use relative pose accumulation instead of direct camera_pose. If not set, will auto-detect based on model output.",
    )
    parser.add_argument(
        "--no_relative_pose",
        action="store_true",
        help="Force to use direct camera_pose instead of relative pose accumulation.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Frame interval for reading images from video or image sequence (default: 1, use every frame).",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=1,
        help="Update interval for state update (default: 1, update every frame). "
        "Note: relative_pose is predicted relative to the previous updated frame.",
    )
    parser.add_argument(
        "--max_frame",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: None, process all frames).",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=1000000,
        help="Only used for demo, reset state for extremely long sequence, chunks are aligned via global camera poses",
    )
    parser.add_argument(
        "--no_auto_keyframe",
        action="store_true",
        help="Disable automatic keyframe selection (default: auto keyframe enabled).",
    )
    parser.add_argument(
        "--img_filter",
        type=str,
        default=None,
        help="Suffix filter for image files (e.g. '_rgb.jpg'). Auto-detected if not set.",
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
        "--max_ref_frames",
        type=int,
        default=None,
        help="Override model's max_ref_frames at inference time (default: model config)",
    )
    parser.add_argument(
        "--no_kf_gate",
        action="store_true",
        help="Disable keyframe-gated state update (all frames update state)",
    )
    parser.add_argument(
        "--force_kf_gate",
        action="store_true",
        help="Force keyframe-gated state update regardless of sequence length",
    )
    parser.add_argument(
        "--skip_pgo",
        action="store_true",
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
        "--pgo_mode",
        type=str,
        default=None,
        choices=["huber", "dcs", "cauchy", "tukey", "irls"],
        help="PGO robust kernel mode (default: huber)",
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
        help="Max PGO constraints per frame (0=unlimited).",
    )
    parser.add_argument(
        "--final_batch_opt",
        action="store_true",
        help="Run final Levenberg-Marquardt batch optimization after incremental iSAM2",
    )
    # Loop closure arguments
    parser.add_argument(
        "--loop_closure",
        action="store_true",
        help="Enable online loop closure detection during inference",
    )
    parser.add_argument(
        "--loop_similarity_threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for loop detection",
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
        help="Reject loop edges with predicted translation > this (meters)",
    )
    parser.add_argument(
        "--loop_sigma_scale",
        type=float,
        default=1.0,
        help="Sigma scale for loop closure identity constraint",
    )
    parser.add_argument(
        "--num_init_frames",
        type=int,
        default=5,
        help="Number of initial frames treated as keyframes before overlap-based selection",
    )
    parser.add_argument(
        "--reset_kf_interval",
        type=int,
        default=0,
        help="Reset buffer every N keyframes (0=disabled)",
    )
    parser.add_argument(
        "--kf_ref_only",
        action="store_true",
        help="Only use keyframe indices as references",
    )
    # GT pose overlay
    parser.add_argument(
        "--gt_pose_path",
        type=str,
        default=None,
        help="Path to GT poses. For vkitti: directory with *_cam.npz files. Aligned to pred at first frame.",
    )
    # Camera color arguments
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
    parser.add_argument(
        "--max_total_points",
        type=int,
        default=0,
        help="Maximum total points in viewer (0=unlimited). Points are uniformly subsampled if over limit.",
    )
    parser.add_argument(
        "--pts_opacity",
        type=float,
        default=1.0,
        help="Point cloud opacity (default: 1.0). Range: 0.0 (transparent) to 1.0 (opaque).",
    )
    return parser.parse_args()


def prepare_input(
    img_paths,
    img_mask,
    size,
    raymaps=None,
    raymap_mask=None,
    revisit=1,
    update=True,
    update_interval=1,
    reset_interval=1000000,
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.
        update_interval (int): Interval for state updates (1 = update every frame).

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            # Determine if this frame should update the state
            # First frame always updates, then every update_interval frames
            should_update = (i == 0) or (i % update_interval == 0)
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
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(should_update).unsqueeze(0),
                "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
            }
            views.append(view)
            if (i + 1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
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
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i + 1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(
    outputs, outdir, revisit=1, use_pose=True, use_relative_pose=None,
    skip_pgo=False, pgo_sigma_rot=None, pgo_sigma_trans=None, pgo_mode=None,
):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        use_relative_pose (bool, optional): Whether to use relative pose accumulation.
            If None, will auto-detect based on model output.
            If True, will use relative pose if available.
            If False, will force to use direct camera_pose.
        skip_pgo (bool): Skip PGO optimization, use chain accumulation only.
        pgo_sigma_rot (float, optional): Override base sigma for rotation in PGO.
        pgo_sigma_trans (float, optional): Override base sigma for translation in PGO.
        pgo_mode (str, optional): PGO robust kernel mode.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf
    from src.dust3r.inference import accumulate_poses

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # Handle reset frames: delete overlap frames (reset_mask=True followed by reset_mask=False)
    # Check if reset key exists in views (it may not be preserved by inference)
    has_reset_key = len(outputs["views"]) > 0 and "reset" in outputs["views"][0]
    if has_reset_key:
        reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
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

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    pr_poses = accumulate_poses(
        outputs["pred"],
        views=outputs["views"],
        use_relative_pose=use_relative_pose,
        skip_pgo=skip_pgo,
        pgo_sigma_rot=pgo_sigma_rot,
        pgo_sigma_trans=pgo_sigma_trans,
        pgo_mode=pgo_mode,
    )
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    return pts3ds_other, colors, conf_other, cam_dict


def parse_seq_path(p, frame_interval=1, img_filter=None):
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
            img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
            img_paths = []
            for ext in img_extensions:
                img_paths.extend(glob.glob(f"{p}/{ext}"))
            img_paths = sorted(img_paths)
        # Apply frame_interval to image sequence
        img_paths = img_paths[::frame_interval]
        tmpdirname = None
    else:
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
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
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
    prepare_input inserts an overlap frame after every reset_interval frames.
    This maps view_idx -> original image path."""
    result = []
    for i, path in enumerate(img_paths):
        result.append(path)
        if (i + 1) % reset_interval == 0:
            result.append(path)  # overlap frame uses same image
    return result


def load_gt_poses(gt_pose_path, pred_cam_dict, frame_interval=1):
    """Load GT poses and align to predicted poses at first frame.

    Supports vkitti format: directory with *_cam.npz files containing 'camera_pose' (4x4 c2w).
    Returns a cam_dict with R and t arrays aligned to pred coordinate system.
    """
    npz_files = sorted(glob.glob(os.path.join(gt_pose_path, "*_cam.npz")))
    if not npz_files:
        print(f"Warning: No *_cam.npz files found in {gt_pose_path}")
        return None

    # Load all GT c2w poses, apply frame_interval
    npz_files = npz_files[::frame_interval]
    num_pred = len(pred_cam_dict["R"])
    npz_files = npz_files[:num_pred]

    gt_c2w_list = []
    for f in npz_files:
        c2w = np.load(f)["camera_pose"].astype(np.float32)
        gt_c2w_list.append(c2w)
    gt_c2w = np.stack(gt_c2w_list, axis=0)  # (N, 4, 4)

    # Build predicted first-frame c2w
    pred_R0 = pred_cam_dict["R"][0]  # (3, 3)
    pred_t0 = pred_cam_dict["t"][0]  # (3,)
    pred_c2w_0 = np.eye(4, dtype=np.float32)
    pred_c2w_0[:3, :3] = pred_R0
    pred_c2w_0[:3, 3] = pred_t0

    # Align: T_align = pred_c2w_0 @ gt_c2w_0^{-1}
    gt_c2w_0_inv = np.linalg.inv(gt_c2w[0])
    T_align = pred_c2w_0 @ gt_c2w_0_inv

    # Apply alignment to all GT poses
    aligned_c2w = T_align @ gt_c2w  # (N, 4, 4) via broadcasting

    print(f"Loaded {len(aligned_c2w)} GT poses from {gt_pose_path}, aligned at first frame.")
    return {
        "focal": pred_cam_dict["focal"][:len(aligned_c2w)],
        "pp": pred_cam_dict["pp"][:len(aligned_c2w)],
        "R": aligned_c2w[:, :3, :3],
        "t": aligned_c2w[:, :3, 3],
    }


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval, img_filter=args.img_filter)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    # Apply max_frame limit if specified
    if args.max_frame is not None and args.max_frame > 0:
        img_paths = img_paths[: args.max_frame]
        print(f"Limited to {len(img_paths)} frames (max_frame={args.max_frame}).")
    else:
        print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        update_interval=args.update_interval,
        reset_interval=args.reset_interval,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    use_keyframes = not getattr(args, 'no_auto_keyframe', False)

    # Loop closure detector
    loop_detector = None
    if getattr(args, 'loop_closure', False):
        from dust3r.utils.loop_closure import OnlineLoopDetector
        loop_detector = OnlineLoopDetector(
            device=device,
            similarity_threshold=args.loop_similarity_threshold,
            temporal_gap=args.loop_temporal_gap,
            max_loops_per_frame=args.loop_max_per_frame,
            nms_window=args.loop_nms_window,
        )
        loop_detector.reset()
        print(f"Loop closure enabled: threshold={args.loop_similarity_threshold}, "
              f"gap={args.loop_temporal_gap}, max_per_frame={args.loop_max_per_frame}")

    # When --no_auto_keyframe, still build callbacks for chain accumulation / PGO,
    # but set num_init_frames to total frame count so every frame is a keyframe
    # (skipping overlap-based filtering).
    num_init = len(views) if not use_keyframes else getattr(args, 'num_init_frames', 5)

    from src.dust3r.inference import make_kf_only_callbacks
    ref_frame_indices_fn, on_frame_processed, keyframe_indices, buffer_pruning_fn = make_kf_only_callbacks(
        kf_window=args.kf_window,
        nkf_buffer_size=args.nkf_buffer_size,
        pgo_sigma_rot=args.pgo_sigma_rot,
        pgo_sigma_trans=args.pgo_sigma_trans,
        pgo_position_scale=getattr(args, 'pgo_position_scale', 0),
        pgo_max_edges=getattr(args, 'pgo_max_edges', 0),
        final_batch_opt=getattr(args, 'final_batch_opt', False),
        loop_detector=loop_detector,
        loop_image_paths=_build_view_image_paths(img_paths, args.reset_interval),
        loop_max_translation=getattr(args, 'loop_max_translation', 20.0),
        loop_sigma_scale=getattr(args, 'loop_sigma_scale', 1.0),
        reset_kf_interval=getattr(args, 'reset_kf_interval', 0),
        kf_ref_only=getattr(args, 'kf_ref_only', False),
        num_init_frames=num_init,
    )

    # Override model's max_ref_frames at inference time
    if args.max_ref_frames is not None:
        model.max_ref_frames = args.max_ref_frames

    # State gating: match eval logic
    if args.no_kf_gate:
        kf_for_state = None
    elif use_keyframes and (args.force_kf_gate or (args.use_relative_pose and len(views) > 60)):
        kf_for_state = keyframe_indices
    else:
        kf_for_state = None

    outputs, state_args = inference_recurrent(
        views, model, device,
        ref_frame_indices_fn=ref_frame_indices_fn,
        keyframe_indices=kf_for_state,
        on_frame_processed=on_frame_processed,
        buffer_pruning_fn=buffer_pruning_fn,
    )
    print(f"  Keyframes: {len(keyframe_indices)}/{len(views)} frames"
          f"{' (all frames, no filtering)' if not use_keyframes else ''}")
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    # Determine pose accumulation mode based on command-line arguments
    if args.use_relative_pose and args.no_relative_pose:
        print(
            "Warning: Both --use_relative_pose and --no_relative_pose specified. --no_relative_pose takes precedence."
        )
        use_relative_pose = False
    elif args.use_relative_pose:
        use_relative_pose = True
    elif args.no_relative_pose:
        use_relative_pose = False
    else:
        use_relative_pose = None  # Auto-detect

    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs,
        args.output_dir,
        1,
        True,
        use_relative_pose=use_relative_pose,
        skip_pgo=args.skip_pgo,
        pgo_sigma_rot=args.pgo_sigma_rot,
        pgo_sigma_trans=args.pgo_sigma_trans,
        pgo_mode=args.pgo_mode,
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Load GT poses if provided
    gt_cam_dict = None
    if args.gt_pose_path is not None:
        gt_cam_dict = load_gt_poses(args.gt_pose_path, cam_dict, args.frame_interval)

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size=args.size,
        downsample_factor=args.downsample_factor,
        keyframe_indices=keyframe_indices,
        cam_color_mode=args.cam_color_mode,
        cam_color_split_frame=args.cam_color_split_frame,
        gt_cam_dict=gt_cam_dict,
        max_total_points=args.max_total_points,
        pts_opacity=args.pts_opacity,
    )
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
