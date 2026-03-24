#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Render a follow-camera video from STream3R 3D point cloud inference results.

Each frame is rendered from a third-person follow camera viewpoint,
showing the accumulated point cloud, camera frustums (rainbow colored),
and a bird's-eye-view (BEV) trajectory.

Usage:
    python render_video.py \
        --model_path weights/scal3r_stream3r/model.pt \
        --seq_path data/tum/rgbd_dataset_freiburg3_walking_xyz/rgb_90 \
        --use_rel_pose --use_streaming \
        --mode causal --kf_window 12 --max_ref_frames 12 --nkf_buffer_size 16 \
        --num_init_frames 2 --pgo_sigma_rot 0.1 --pgo_sigma_trans 0.3 \
        --downsample_factor 4 --vis_threshold 1.0 \
        --output_video render.mp4 --video_fps 24 \
        --cam_offset_back 3.0 --cam_offset_up 1.5 --render_fov 90 \
        --frustum_scale 0.3
"""

import argparse
import os
import shutil
import time

import cv2
import imageio.v2 as iio
import matplotlib.colors as mcolors
import numpy as np
import torch

from demo import parse_seq_path, prepare_output, _build_view_image_paths, _remove_overlap_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Render follow-camera video from STream3R inference.")

    # ── Core (same as demo.py) ──
    parser.add_argument("--model_path", type=str, default="yslan/STream3R")
    parser.add_argument("--seq_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--output_dir", type=str, default="./demo_tmp")
    parser.add_argument("--mode", type=str, default="causal", choices=["causal", "window", "full"])
    parser.add_argument("--use_streaming", action="store_true")

    # ── Frame control ──
    parser.add_argument("--max_images", type=int, default=100000)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=1000000)
    parser.add_argument("--img_filter", type=str, default=None)

    # ── Visualization ──
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=1)

    # ── Relative pose ──
    parser.add_argument("--use_rel_pose", action="store_true")
    parser.add_argument("--use_rel_pose_prompt", action="store_true")
    parser.add_argument("--ref_feat_type", type=str, default="img_feat",
                        choices=["img_feat", "camera_token"])
    parser.add_argument("--rel_pose_global_only", action="store_true")
    parser.add_argument("--max_ref_frames", type=int, default=4)

    # ── PGO ──
    parser.add_argument("--skip_pgo", action="store_true")
    parser.add_argument("--kf_window", type=int, default=4)
    parser.add_argument("--nkf_buffer_size", type=int, default=0)
    parser.add_argument("--num_init_frames", type=int, default=5)
    parser.add_argument("--pgo_sigma_rot", type=float, default=None)
    parser.add_argument("--pgo_sigma_trans", type=float, default=None)
    parser.add_argument("--pgo_mode", type=str, default=None,
                        choices=["huber", "dcs", "cauchy", "tukey"])
    parser.add_argument("--pgo_max_edges", type=int, default=0)
    parser.add_argument("--final_batch_opt", action="store_true")
    parser.add_argument("--kf_only_cache", action="store_true")
    parser.add_argument("--use_global_pose_init", action="store_true")
    parser.add_argument("--global_pose_prior_sigma", type=float, default=None)

    # ── Loop closure ──
    parser.add_argument("--loop_closure", action="store_true")
    parser.add_argument("--loop_similarity_threshold", type=float, default=0.85)
    parser.add_argument("--loop_temporal_gap", type=int, default=300)
    parser.add_argument("--loop_max_per_frame", type=int, default=1)
    parser.add_argument("--loop_nms_window", type=int, default=50)
    parser.add_argument("--loop_max_translation", type=float, default=20.0)
    parser.add_argument("--loop_sigma_scale", type=float, default=1.0)

    # ── Video rendering args ──
    parser.add_argument("--output_video", type=str, default="render_video.mp4",
                        help="Output video path (default: render_video.mp4)")
    parser.add_argument("--render_height", type=int, default=360,
                        help="Render height (default: 360)")
    parser.add_argument("--render_width", type=int, default=640,
                        help="Render width (default: 640, 360p 16:9)")
    parser.add_argument("--point_radius", type=int, default=1,
                        help="Pixel radius for splatted points (default: 1)")
    parser.add_argument("--video_fps", type=int, default=15,
                        help="Output video FPS (default: 15)")
    parser.add_argument("--bg_color", type=str, default="255,255,255",
                        help="Background color as R,G,B (default: 255,255,255 white)")
    parser.add_argument("--no_accumulate", action="store_true",
                        help="Only show current frame's points (don't accumulate)")

    # ── Camera follow offset ──
    parser.add_argument("--cam_offset_back", type=float, default=2.0,
                        help="Offset behind the camera along -Z_cam (meters)")
    parser.add_argument("--cam_offset_up", type=float, default=1.0,
                        help="Offset above the camera along -Y_cam (meters)")
    parser.add_argument("--render_fov", type=float, default=90.0,
                        help="Render camera FOV in degrees (default: 90)")

    # ── Frustum drawing ──
    parser.add_argument("--frustum_scale", type=float, default=0.001,
                        help="Camera frustum size in world units (default: 0.001)")
    parser.add_argument("--frustum_line_width", type=int, default=2,
                        help="Line width for frustum edges (default: 2)")
    parser.add_argument("--current_frustum_line_width", type=int, default=3,
                        help="Line width for current frame's frustum (default: 3)")
    parser.add_argument("--render_every", type=int, default=4,
                        help="Render every N-th frame (default: 4). "
                             "E.g., --render_every 3 renders 1/3 of frames.")

    return parser.parse_args()


# ── Point cloud helpers ──

def filter_frame_points(pts, colors, conf, vis_threshold, downsample_factor):
    """Filter points by confidence and downsample. Returns (N,3) pts and (N,3) colors."""
    pts = pts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    conf = conf.reshape(-1)
    mask = conf > vis_threshold
    pts, colors = pts[mask], colors[mask]
    if downsample_factor > 1 and len(pts) > 0:
        idx = np.arange(0, len(pts), downsample_factor)
        pts, colors = pts[idx], colors[idx]
    return pts, colors


def render_point_cloud(pts_world, colors, R_w2c, t_w2c, focal, cx, cy, H, W,
                       point_radius=1, bg_color=(255, 255, 255)):
    """Render point cloud onto a canvas using z-buffer splatting."""
    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)
    if len(pts_world) == 0:
        return canvas

    p_cam = pts_world @ R_w2c.T + t_w2c[None, :]
    z = p_cam[:, 2]
    valid = z > 0.01
    p_cam = p_cam[valid]
    z = z[valid]
    colors_valid = colors[valid]

    if len(p_cam) == 0:
        return canvas

    u = (focal * p_cam[:, 0] / z + cx).astype(np.float32)
    v = (focal * p_cam[:, 1] / z + cy).astype(np.float32)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    m = point_radius
    keep = (ui >= m) & (ui < W - m) & (vi >= m) & (vi < H - m)
    ui, vi, z_v, c_rgb = ui[keep], vi[keep], z[keep], \
        (colors_valid[keep] * 255).clip(0, 255).astype(np.uint8)
    if len(ui) == 0:
        return canvas

    if point_radius <= 1:
        order = np.argsort(-z_v)
        canvas[vi[order], ui[order]] = c_rgb[order]
    else:
        depth_buf = np.full((H, W), np.inf, dtype=np.float32)
        order = np.argsort(z_v)
        for idx in order:
            y, x, d = vi[idx], ui[idx], z_v[idx]
            if d < depth_buf[y, x]:
                cv2.circle(canvas, (x, y), point_radius, c_rgb[idx].tolist(), -1)
                cv2.circle(depth_buf, (x, y), point_radius, float(d), -1)
    return canvas


# ── Frustum helpers ──

def _frustum_corners_world(R_c2w, t_c2w, focal, pp, img_H, img_W, scale):
    """5 frustum corners in world coords: [apex, tl, tr, br, bl]."""
    hw, hh = img_W / 2.0, img_H / 2.0
    z = scale
    corners_cam = np.array([
        [-hw / focal * z, -hh / focal * z, z],
        [ hw / focal * z, -hh / focal * z, z],
        [ hw / focal * z,  hh / focal * z, z],
        [-hw / focal * z,  hh / focal * z, z],
    ], dtype=np.float64)
    pts_cam = np.vstack([[[0, 0, 0]], corners_cam])  # (5, 3)
    return pts_cam @ R_c2w.T + t_c2w[None, :]


_FRUSTUM_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]


def draw_frustum(canvas, corners_world, R_w2c, t_w2c, focal, cx, cy, H, W,
                 color, line_width=2):
    """Draw a camera frustum wireframe on canvas."""
    p_cam = corners_world @ R_w2c.T + t_w2c[None, :]
    z = p_cam[:, 2]
    u = np.where(z > 0.01, focal * p_cam[:, 0] / z + cx, -1e6)
    v = np.where(z > 0.01, focal * p_cam[:, 1] / z + cy, -1e6)

    for i0, i1 in _FRUSTUM_EDGES:
        if z[i0] <= 0.01 or z[i1] <= 0.01:
            continue
        p0 = (int(round(u[i0])), int(round(v[i0])))
        p1 = (int(round(u[i1])), int(round(v[i1])))
        cv2.line(canvas, p0, p1, color, line_width, cv2.LINE_AA)


def rainbow_color(idx, total):
    """HSV rainbow color as (R, G, B) uint8 tuple."""
    hue = idx / max(total - 1, 1)
    rgb = mcolors.hsv_to_rgb((hue, 1.0, 1.0))
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


# ── Follow camera ──

def compute_follow_camera(R_c2w, t_c2w, offset_back, offset_up):
    """Third-person follow camera: behind and above the target camera, looking forward."""
    forward = R_c2w[:, 2]
    up_cam = -R_c2w[:, 1]

    render_pos = t_c2w - forward * offset_back + up_cam * offset_up
    look_at = t_c2w + forward * 0.5
    fwd = look_at - render_pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

    right = np.cross(fwd, up_cam)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.cross(fwd, np.array([0, 0, 1]))
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    down = np.cross(fwd, right)

    R_render = np.stack([right, down, fwd], axis=1)
    return R_render, render_pos


# ── Inference (mirrors demo.py run_inference) ──

def run_inference(args):
    """Run STream3R inference and return (pts3ds, colors, conf, cam_dict, keyframe_indices)."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    from stream3r.models.stream3r import STream3R
    from stream3r.stream_session import StreamSession
    from stream3r.models.components.utils.load_fn import load_and_preprocess_images

    use_rel_pose = args.use_rel_pose or args.use_rel_pose_prompt

    # ── Parse image paths ──
    img_paths, tmpdirname = parse_seq_path(
        args.seq_path, frame_interval=args.frame_interval, img_filter=args.img_filter
    )
    if not img_paths:
        print(f"No images found in {args.seq_path}.")
        return None
    if len(img_paths) > args.max_images:
        img_paths = img_paths[:args.max_images]
    print(f"Found {len(img_paths)} images in {args.seq_path}.")

    # ── Load model ──
    print(f"Loading STream3R model from {args.model_path}...")
    if args.model_path.endswith(('.pt', '.pth', '.bin')):
        raw = torch.load(args.model_path, map_location=device, weights_only=False)

        if isinstance(raw, dict) and 'state_dict' in raw and 'config' in raw:
            checkpoint = raw['state_dict']
            ckpt_config = raw['config']
            print(f"Loaded config from checkpoint: {ckpt_config}")
        else:
            checkpoint = raw
            ckpt_config = {}

        if any(k.startswith('net.') for k in checkpoint.keys()):
            checkpoint = {k.replace('net.', ''): v for k, v in checkpoint.items() if k.startswith('net.')}

        has_rel_pose_keys = any("rel_pose" in k for k in checkpoint.keys())
        model_use_rel_pose = use_rel_pose or has_rel_pose_keys

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
        use_rel_pose = model_use_rel_pose
    else:
        model = STream3R.from_pretrained(args.model_path).to(device)
        if use_rel_pose:
            print("Warning: Relative pose requested but HuggingFace model may not support it.")
            use_rel_pose = False
    model.eval()

    # ── Load images ──
    print("Loading and preprocessing images...")
    images = load_and_preprocess_images(img_paths, mode="crop").to(device)
    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")

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

        no_pgo = args.skip_pgo
        use_pgo = use_rel_pose and not no_pgo
        use_pgo_session = use_pgo or getattr(args, 'kf_only_cache', False)

        pgo_config = None
        if use_pgo_session:
            pgo_config = dict(
                kf_pgo=use_pgo_session,
                kf_window=args.kf_window,
                nkf_buffer_size=args.nkf_buffer_size,
                num_init_frames=args.num_init_frames,
                kf_only_cache=getattr(args, 'kf_only_cache', False),
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

        print(f"Streaming inference: use_pgo={use_pgo}, kf_window={args.kf_window}, "
              f"nkf_buffer_size={args.nkf_buffer_size}")
        session = StreamSession(model, mode=args.mode, use_pgo=use_pgo_session, pgo_config=pgo_config)

        num_frames = images.shape[0]
        with torch.no_grad():
            for i in range(num_frames):
                image = images[i:i+1]
                predictions = session.forward_stream(image)

                if (i + 1) % args.reset_interval == 0 and (i + 1) < num_frames:
                    print(f"  Resetting streaming state at frame {i+1}")
                    session.reset_streaming_state()
                    predictions = session.forward_stream(image)
                    overlap_indices.append(session.frame_count - 1)

        keyframe_indices = getattr(session, 'keyframe_indices', set()) if use_pgo_session else set()

        if use_pgo and use_rel_pose:
            pgo_poses = session.get_pgo_poses()
            if pgo_poses is not None and overlap_indices:
                keep_mask = [True] * session.frame_count
                for oi in overlap_indices:
                    keep_mask[oi] = False
                pgo_poses = [p for p, keep in zip(pgo_poses, keep_mask) if keep]
            if pgo_poses is not None:
                print(f"  PGO: {len(pgo_poses)} optimized poses")

        predictions = session.get_all_predictions()

        if overlap_indices:
            predictions = _remove_overlap_predictions(predictions, overlap_indices, session.frame_count)

        if use_pgo_session:
            n_kf = len(keyframe_indices)
            total_f = session.frame_count - len(overlap_indices)
            print(f"  Keyframes: {n_kf}/{total_f} frames")

        session.clear()
    else:
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
    print("Preparing output...")
    if pgo_poses is not None:
        pts3ds_other, colors, conf, cam_dict, _ = prepare_output(
            predictions, args.output_dir, pgo_poses=pgo_poses, save=False
        )
    elif use_rel_pose:
        pts3ds_other, colors, conf, cam_dict, _ = prepare_output(
            predictions, args.output_dir, use_relative_pose=True, save=False
        )
    else:
        pts3ds_other, colors, conf, cam_dict, _ = prepare_output(
            predictions, args.output_dir, use_relative_pose=False, save=False
        )

    return pts3ds_other, colors, conf, cam_dict, keyframe_indices


# ── Main render pipeline ──

def run_render_video(args):
    result = run_inference(args)
    if result is None:
        return
    pts3ds_other, colors, conf, cam_dict, keyframe_indices = result

    N = len(pts3ds_other)
    pts_np = [p.cpu().numpy() if isinstance(p, torch.Tensor) else p for p in pts3ds_other]
    col_np = [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in colors]
    conf_np = [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in conf]

    # ── Build filtered point clouds per frame ──
    print("Building full point cloud...")
    all_pts_list, all_col_list = [], []
    for i in range(N):
        pts_i, col_i = filter_frame_points(
            pts_np[i], col_np[i], conf_np[i],
            args.vis_threshold, args.downsample_factor,
        )
        all_pts_list.append(pts_i)
        all_col_list.append(col_i)
    all_pts = np.concatenate(all_pts_list, axis=0)
    all_col = np.concatenate(all_col_list, axis=0)
    print(f"  Total points: {len(all_pts)}")

    # ── Pre-compute frustum corners ──
    sample_shape = pts_np[0].shape  # (H, W, 3) or (1, H, W, 3)
    if len(sample_shape) == 4:
        img_H, img_W = sample_shape[1], sample_shape[2]
    else:
        img_H, img_W = sample_shape[0], sample_shape[1]
    frustum_corners = []
    for i in range(N):
        corners = _frustum_corners_world(
            cam_dict["R"][i], cam_dict["t"][i],
            float(cam_dict["focal"][i]), cam_dict["pp"][i],
            img_H, img_W, args.frustum_scale,
        )
        frustum_corners.append(corners)

    # ── Render resolution and intrinsics ──
    H = args.render_height if args.render_height > 0 else img_H
    W = args.render_width if args.render_width > 0 else img_W
    bg_color = tuple(int(x) for x in args.bg_color.split(","))

    fov_rad = np.deg2rad(args.render_fov)
    render_focal = (W / 2.0) / np.tan(fov_rad / 2.0)
    render_cx, render_cy = W / 2.0, H / 2.0

    # ── Video paths ──
    base, ext = os.path.splitext(args.output_video)
    input_video_path = base + "_input" + ext
    bev_video_path = base + "_bev" + ext

    # ── Frame indices to render ──
    render_indices = list(range(0, N, args.render_every))
    n_render = len(render_indices)

    # ── BEV setup: fixed top-down view covering all poses + points ──
    bev_H, bev_W = 360, 360
    all_cam_positions = cam_dict["t"]  # (N, 3)
    all_xz = all_cam_positions[:, [0, 2]]
    pts_xz = all_pts[::max(1, len(all_pts) // 10000), [0, 2]]  # subsample for bounds
    combined_xz = np.vstack([all_xz, pts_xz])
    xz_min = combined_xz.min(axis=0)
    xz_max = combined_xz.max(axis=0)
    xz_center = (xz_min + xz_max) / 2.0
    xz_range = (xz_max - xz_min).max() * 1.2  # 20% margin
    if xz_range < 1e-3:
        xz_range = 10.0
    bev_scale = min(bev_W, bev_H) / xz_range
    bev_cx = bev_W / 2.0
    bev_cy = bev_H / 2.0

    def world_to_bev(x, z):
        px = (x - xz_center[0]) * bev_scale + bev_cx
        py = (z - xz_center[1]) * bev_scale + bev_cy
        return int(round(px)), int(round(py))

    # Pre-compute per-frame BEV points (downsampled for speed)
    bev_frame_pts = []
    bev_frame_cols = []
    for i in range(N):
        pts_i, col_i = all_pts_list[i], all_col_list[i]
        if len(pts_i) == 0:
            bev_frame_pts.append(np.zeros((0, 2), dtype=np.int32))
            bev_frame_cols.append(np.zeros((0, 3), dtype=np.uint8))
            continue
        ds = max(1, len(pts_i) // 100)
        pts_ds = pts_i[::ds]
        col_ds = col_i[::ds]
        px = ((pts_ds[:, 0] - xz_center[0]) * bev_scale + bev_cx).astype(np.int32)
        py = ((pts_ds[:, 2] - xz_center[1]) * bev_scale + bev_cy).astype(np.int32)
        coords = np.stack([px, py], axis=1)
        colors_u8 = (col_ds * 255).clip(0, 255).astype(np.uint8)
        bev_frame_pts.append(coords)
        bev_frame_cols.append(colors_u8)

    bev_pts_canvas = np.full((bev_H, bev_W, 3), bg_color, dtype=np.uint8)

    # ── Render videos ──
    print(f"Rendering {n_render}/{N} frames to {args.output_video} "
          f"({W}x{H} @ {args.video_fps}fps, every {args.render_every} frames)...")
    print(f"Input video: {input_video_path}")
    print(f"BEV video: {bev_video_path}")
    codec_params = ["-crf", "28", "-preset", "medium"]
    writer = iio.get_writer(args.output_video, fps=args.video_fps, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=1,
                            output_params=codec_params)
    writer_input = iio.get_writer(input_video_path, fps=args.video_fps, codec="libx264",
                                  pixelformat="yuv420p", macro_block_size=1,
                                  output_params=codec_params)
    writer_bev = iio.get_writer(bev_video_path, fps=args.video_fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=1,
                                output_params=codec_params)

    for idx, t in enumerate(render_indices):
        # Follow camera with offset
        R_render, t_render = compute_follow_camera(
            cam_dict["R"][t], cam_dict["t"][t],
            args.cam_offset_back, args.cam_offset_up,
        )
        R_w2c = R_render.T
        t_w2c = -R_w2c @ t_render

        # 1) Render full point cloud
        canvas = render_point_cloud(
            all_pts, all_col, R_w2c, t_w2c,
            render_focal, render_cx, render_cy, H, W,
            point_radius=args.point_radius, bg_color=bg_color,
        )

        # 2) Draw all camera frustums (rainbow)
        for i in range(N):
            color = rainbow_color(i, N)
            draw_frustum(canvas, frustum_corners[i],
                         R_w2c, t_w2c, render_focal, render_cx, render_cy, H, W,
                         color=color, line_width=args.frustum_line_width)

        # 3) Highlight current frustum: white outline + rainbow fill
        draw_frustum(canvas, frustum_corners[t],
                     R_w2c, t_w2c, render_focal, render_cx, render_cy, H, W,
                     color=(255, 255, 255), line_width=args.current_frustum_line_width + 2)
        draw_frustum(canvas, frustum_corners[t],
                     R_w2c, t_w2c, render_focal, render_cx, render_cy, H, W,
                     color=rainbow_color(t, N), line_width=args.current_frustum_line_width)

        writer.append_data(canvas)

        # 4) Write input image
        input_img = col_np[t].squeeze(0) if len(col_np[t].shape) == 4 else col_np[t]
        input_img = (input_img * 255).clip(0, 255).astype(np.uint8)
        writer_input.append_data(input_img)

        # 5) BEV frame: accumulate pts + trajectory up to t + current marker
        prev_t = render_indices[idx - 1] + 1 if idx > 0 else 0
        for fi in range(prev_t, t + 1):
            coords = bev_frame_pts[fi]
            cols = bev_frame_cols[fi]
            for k in range(len(coords)):
                px, py = coords[k, 0], coords[k, 1]
                if 0 <= px < bev_W and 0 <= py < bev_H:
                    bev_pts_canvas[py, px] = cols[k]

        bev_frame = bev_pts_canvas.copy()

        # Draw trajectory up to current frame (rainbow colored)
        for i in range(t + 1):
            px, py = world_to_bev(all_cam_positions[i, 0], all_cam_positions[i, 2])
            if 0 <= px < bev_W and 0 <= py < bev_H:
                color = rainbow_color(i, N)
                cv2.circle(bev_frame, (px, py), 2, color, -1)
            if i > 0:
                px0, py0 = world_to_bev(all_cam_positions[i-1, 0], all_cam_positions[i-1, 2])
                cv2.line(bev_frame, (px0, py0), (px, py), rainbow_color(i, N), 1, cv2.LINE_AA)

        # Draw forward direction arrow for current frame
        pos_t = all_cam_positions[t]
        fwd_t = cam_dict["R"][t][:, 2]
        arrow_len = xz_range * 0.03
        px_t, py_t = world_to_bev(pos_t[0], pos_t[2])
        px_fwd, py_fwd = world_to_bev(pos_t[0] + fwd_t[0] * arrow_len,
                                       pos_t[2] + fwd_t[2] * arrow_len)
        cv2.circle(bev_frame, (px_t, py_t), 6, (255, 255, 255), -1)
        cv2.circle(bev_frame, (px_t, py_t), 4, rainbow_color(t, N), -1)
        cv2.arrowedLine(bev_frame, (px_t, py_t), (px_fwd, py_fwd),
                        (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

        writer_bev.append_data(bev_frame)

        if (idx + 1) % 50 == 0 or idx == n_render - 1:
            print(f"  [{idx+1}/{n_render}] frame {t}")

    writer.close()
    writer_input.close()
    writer_bev.close()
    print(f"Render video saved to {args.output_video}")
    print(f"Input video saved to {input_video_path}")
    print(f"BEV video saved to {bev_video_path}")


def main():
    args = parse_args()
    run_render_video(args)


if __name__ == "__main__":
    main()
