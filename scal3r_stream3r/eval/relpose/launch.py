# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import torch
import argparse

# Ensure project root is in sys.path before importing stream3r (local code over pip-installed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from accelerate import PartialState
from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.dust3r.utils.image import load_images_for_eval as load_images
from stream3r.dust3r.utils.device import collate_with_cat
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.dust3r.utils.geometry import inv
from stream3r.utils.utils import ImgDust3r2Stream3r
from eval.relpose.metadata import dataset_metadata
from eval.relpose.utils import *
from tqdm import tqdm


torch.backends.cuda.matmul.allow_tf32 = True

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ===========================================


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--no_crop",
                        type=bool,
                        default=True,
                        help="whether to crop input data")

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument("--pose_eval_stride",
                        default=1,
                        type=int,
                        help="stride for pose evaluation")
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

    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        type=str,
        default="causal",
        choices=["causal", "window", "full"],
        help="attention mode for StreamSession",
    )
    parser.add_argument(
        "--use_rel_pose",
        action="store_true",
        default=False,
        help="Use accumulated relative poses instead of pose_enc for evaluation",
    )
    parser.add_argument(
        "--max_ref_frames",
        type=int,
        default=4,
        help="Max reference frames for CUT3R-style multi-ref pose",
    )
    parser.add_argument(
        "--use_pgo",
        action="store_true",
        default=False,
        help="Use PGO for pose optimization during streaming inference",
    )
    parser.add_argument("--kf_window", type=int, default=4, help="Keyframe window size for PGO buffer")
    parser.add_argument("--nkf_buffer_size", type=int, default=0, help="Number of recent non-keyframes to keep in ref buffer (default: 0)")
    parser.add_argument("--no_pgo", action="store_true", default=False, help="Disable PGO, use chain accumulation only")
    parser.add_argument("--kf_only_cache", action="store_true", default=False, help="Only keyframes update KV cache (window mode)")
    parser.add_argument("--num_init_frames", type=int, default=5, help="Number of initial frames treated as keyframes")
    parser.add_argument("--pgo_sigma_rot", type=float, default=None, help="PGO rotation sigma (default: 0.5)")
    parser.add_argument("--pgo_sigma_trans", type=float, default=None, help="PGO translation sigma (default: 0.5)")
    parser.add_argument("--pgo_mode", type=str, default=None, choices=["huber", "cauchy", "tukey", "dcs", None], help="PGO robust kernel")
    parser.add_argument("--pgo_max_edges", type=int, default=0, help="Max PGO edges per frame (0=unlimited)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights (.pt file)",
    )
    parser.add_argument(
        "--ref_feat_type",
        type=str,
        default="img_feat",
        choices=["img_feat", "camera_token"],
        help="Reference feature type for rel_pose conditioning",
    )
    parser.add_argument(
        "--rel_pose_global_only",
        action="store_true",
        default=False,
        help="Only use global-path features (1024d) for rel_pose decoder",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=1000000,
        help="Reset recurrent state every N frames with 1 overlap frame (default: disabled)",
    )
    parser.add_argument(
        "--use_global_pose_init",
        action="store_true",
        default=False,
        help="Use pose_enc (global pose) as PGO initialization",
    )
    parser.add_argument(
        "--global_pose_prior_sigma",
        type=float,
        default=None,
        help="If set, add PriorFactorPose3 with this sigma for global pose (e.g. 0.1)",
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
        help="Sigma scale for loop closure constraints (lower = tighter snap)",
    )
    parser.add_argument(
        "--no_correct_scale",
        action="store_true",
        default=False,
        help="Disable scale correction in ATE/RPE evaluation (align=True, correct_scale=False)",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="Enable center cropping for input images (default: no crop)",
    )
    return parser


def eval_pose_estimation_dist(args,
                              model,
                              img_path,
                              save_dir=None,
                              mask_path=None):

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
                seq for seq in seq_list
                if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    # Loop closure detector (loaded once, reused across sequences)
    loop_detector = None
    if getattr(args, 'loop_closure', False) and not getattr(args, 'loop_gt', False):
        from stream3r.utils.loop_closure import OnlineLoopDetector
        loop_detector = OnlineLoopDetector(
            device=device,
            similarity_threshold=args.loop_similarity_threshold,
            temporal_gap=args.loop_temporal_gap,
            max_loops_per_frame=args.loop_max_per_frame,
            nms_window=args.loop_nms_window,
        )
        print(f"Loop closure enabled: threshold={args.loop_similarity_threshold}, "
              f"gap={args.loop_temporal_gap}, max_per_frame={args.loop_max_per_frame}")
    elif getattr(args, 'loop_closure', False) and getattr(args, 'loop_gt', False):
        print("Loop closure enabled: GT detection mode (ablation)")

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

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        os.makedirs(save_dir, exist_ok=True)
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(
                        save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get("mask_path_seq_func",
                                                  lambda mask_path, seq: None)
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                img_filter = metadata.get("img_filter", None)
                filelist = [
                    os.path.join(dir_path, name)
                    for name in os.listdir(dir_path)
                    if img_filter is None or img_filter(name)
                ]
                filelist.sort()
                filelist = filelist[::args.pose_eval_stride]

                # Filter out frames with non-finite GT poses (e.g. ScanNet tracking failures)
                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)
                if traj_format == "replica" and gt_traj_file is not None and os.path.exists(gt_traj_file):
                    gt_poses_raw = np.loadtxt(gt_traj_file)
                    valid_mask = np.isfinite(gt_poses_raw).all(axis=1)
                    if not valid_mask.all():
                        n_invalid = (~valid_mask).sum()
                        print(f"Warning: {seq} has {n_invalid} non-finite GT poses, skipping those frames.")
                        filelist = [f for f, v in zip(filelist, valid_mask) if v]

                images = load_images(
                    filelist,
                    size=518,
                    verbose=True,
                    crop=getattr(args, 'crop', False),
                    patch_size=14,
                )

                images = collate_with_cat([tuple(images)])
                images = torch.stack([view["img"] for view in images], dim=1)
                images = ImgDust3r2Stream3r(images).to(device)

                with torch.no_grad():
                    # Enable PGO by default when using rel_pose (matches CUT3R)
                    no_pgo = getattr(args, 'no_pgo', False)
                    kf_only_cache = getattr(args, 'kf_only_cache', False)
                    use_pgo = (args.use_pgo or args.use_rel_pose) and not no_pgo
                    # kf_only_cache needs PGO callbacks for keyframe selection,
                    # even when --no_pgo is set (no_pgo only disables PGO poses)
                    use_pgo_session = use_pgo or kf_only_cache
                    pgo_config = dict(
                        kf_pgo=use_pgo_session,
                        kf_window=args.kf_window,
                        nkf_buffer_size=getattr(args, 'nkf_buffer_size', 0),
                        num_init_frames=args.num_init_frames,
                        kf_only_cache=kf_only_cache,
                    ) if use_pgo_session else None
                    if pgo_config is not None:
                        if getattr(args, 'pgo_sigma_rot', None) is not None:
                            pgo_config['pgo_sigma_rot'] = args.pgo_sigma_rot
                        if getattr(args, 'pgo_sigma_trans', None) is not None:
                            pgo_config['pgo_sigma_trans'] = args.pgo_sigma_trans
                        if getattr(args, 'pgo_max_edges', 0) > 0:
                            pgo_config['pgo_max_edges'] = args.pgo_max_edges
                        if getattr(args, 'pgo_mode', None) is not None:
                            pgo_config['pgo_mode'] = args.pgo_mode
                        if getattr(args, 'use_global_pose_init', False):
                            pgo_config['use_global_pose_init'] = True
                            if getattr(args, 'global_pose_prior_sigma', None) is not None:
                                pgo_config['global_pose_prior_sigma'] = args.global_pose_prior_sigma

                    # Loop closure: set up per-sequence detector and image paths
                    seq_loop_detector = loop_detector  # SALAD detector (shared across seqs)
                    reset_interval = getattr(args, 'reset_interval', 1000000)
                    if getattr(args, 'loop_closure', False) and getattr(args, 'loop_gt', False):
                        from stream3r.utils.loop_closure import GTLoopDetector
                        gt_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                        if gt_file and os.path.isfile(gt_file):
                            seq_loop_detector = GTLoopDetector(
                                gt_poses_file=gt_file,
                                temporal_gap=args.loop_temporal_gap,
                                max_loops_per_frame=args.loop_max_per_frame,
                                nms_window=args.loop_nms_window,
                                reset_interval=reset_interval,
                            )
                            print(f"  GT loop detector for {seq}: {gt_file}")
                    if seq_loop_detector is not None:
                        seq_loop_detector.reset()
                    if pgo_config is not None and seq_loop_detector is not None:
                        pgo_config['loop_detector'] = seq_loop_detector
                        pgo_config['loop_image_paths'] = _build_view_image_paths(filelist, reset_interval)
                        pgo_config['loop_max_translation'] = getattr(args, 'loop_max_translation', 20.0)
                        pgo_config['loop_sigma_scale'] = getattr(args, 'loop_sigma_scale', 1.0)

                    session = StreamSession(model, mode=args.mode,
                                            use_pgo=use_pgo_session, pgo_config=pgo_config)
                    overlap_indices = []  # global prediction indices of overlap frames
                    num_frames = images.shape[1]
                    for i in range(num_frames):
                        image = images[:, i:i+1]
                        predictions = session.forward_stream(image)

                        # Reset streaming state after every reset_interval frames
                        if (i + 1) % reset_interval == 0 and (i + 1) < num_frames:
                            session.reset_streaming_state()
                            # Re-feed current frame as overlap (first frame of new segment)
                            predictions = session.forward_stream(image)
                            # Mark this overlap prediction for removal
                            overlap_indices.append(session.frame_count - 1)

                # Print keyframe stats for this sequence
                if use_pgo_session and hasattr(session, 'on_frame_processed') and hasattr(session.on_frame_processed, 'print_kf_stats'):
                    session.on_frame_processed.print_kf_stats()

                if args.use_rel_pose and "rel_pose" in predictions:
                    # Try PGO poses first, unless --no_pgo forces chain accumulation
                    # NOTE: get_pgo_poses() calls finalize() BEFORE overlap removal
                    # (matching CUT3R: finalize with all frames, then remove overlaps)
                    pgo_poses = session.get_pgo_poses() if not no_pgo else None

                    # Remove overlap frames AFTER finalize (CUT3R: prepare_output removes after finalize)
                    if overlap_indices and pgo_poses is not None:
                        keep_mask = [True] * session.frame_count
                        for oi in overlap_indices:
                            keep_mask[oi] = False
                        pgo_poses = [p for p, keep in zip(pgo_poses, keep_mask) if keep]

                    if pgo_poses is not None:
                        pr_poses = [p[0] for p in pgo_poses]
                    else:
                        # Fallback: multi-ref chain accumulation (K varies per frame)
                        from eval.relpose.utils import _se3_inverse_batch, _reorthogonalize_c2w
                        rel_pose_list = predictions["rel_pose"]
                        if not isinstance(rel_pose_list, list):
                            rel_pose_list = [rel_pose_list]
                        c2w_init = torch.eye(4, device=device).unsqueeze(0)
                        c2w_history = {0: c2w_init}
                        pr_poses = [c2w_init[0]]
                        for fi in range(1, len(rel_pose_list)):
                            rp = rel_pose_list[fi]
                            rt = rp["rel_trans"]  # [B, 1, K_i, 3]
                            rr = rp["rel_rot"]    # [B, 1, K_i, 3, 3]
                            K_i = rt.shape[2]
                            c2w_i = None
                            for k in range(K_i):
                                ref_idx = fi - k - 1
                                if ref_idx < 0 or ref_idx not in c2w_history:
                                    continue
                                inv_rel = _se3_inverse_batch(
                                    rr[0, 0, k:k+1], rt[0, 0, k:k+1])
                                c2w_i = c2w_history[ref_idx] @ inv_rel
                                c2w_i[0] = _reorthogonalize_c2w(c2w_i[0])
                                break
                            if c2w_i is None:
                                c2w_i = c2w_history.get(fi - 1, c2w_init).clone()
                            c2w_history[fi] = c2w_i
                            pr_poses.append(c2w_i[0])
                else:
                    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], predictions["images"].shape[-2:])
                    pr_poses = []
                    for i in range(extrinsic.shape[1]):
                        pr_poses.append(inv(torch.cat([extrinsic[0, i], torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)))

                # Extract timestamps from image filenames for TUM format
                traj_format = metadata.get("traj_format", None)
                if traj_format == "tum":
                    img_timestamps = [float(os.path.splitext(os.path.basename(f))[0]) for f in filelist]
                else:
                    img_timestamps = None
                pred_traj = get_tum_poses(pr_poses, timestamps=img_timestamps)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path,
                                                        seq)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(gt_traj_file=gt_traj_file,
                                        stride=args.pose_eval_stride)
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                        correct_scale=not args.no_correct_scale,
                    )
                    plot_trajectory(pred_traj,
                                    gt_traj,
                                    title=seq,
                                    filename=f"{save_dir}/{seq}.png")
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
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
                        e) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(
                        f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

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

    return avg_ate, avg_rpe_trans, avg_rpe_rot


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path)
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def main():
    args = get_args_parser()
    args = args.parse_args()

    args.full_seq = False
    args.no_crop = False

    if args.pretrained is not None:
        raw = torch.load(args.pretrained, map_location=args.device, weights_only=False)
        # New format: {'state_dict': ..., 'config': ...}; old format: plain state_dict
        if isinstance(raw, dict) and 'state_dict' in raw and 'config' in raw:
            checkpoint = raw['state_dict']
            config = raw['config']
            print(f"Loaded config from checkpoint: {config}")
        else:
            checkpoint = raw
            config = {}

        # Build model config: checkpoint config > CLI override > defaults
        # NOTE: ref_feat_type / rel_pose_global_only are no longer model params
        # (the model is locked to camera_token + concat). The CLI flags are kept
        # for backward-compatible invocation but are not passed to the model.
        use_rel_pose = config.get('use_rel_pose_prompt', args.use_rel_pose)
        num_rel_pose_tokens = config.get('num_rel_pose_tokens', 4)

        model = STream3R(
            use_rel_pose_prompt=use_rel_pose,
            num_rel_pose_tokens=num_rel_pose_tokens,
        )
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained from {args.pretrained}")
        print(f"  use_rel_pose={use_rel_pose}, num_rel_pose_tokens={num_rel_pose_tokens}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        # Override max_ref_frames at inference time (buffer-window cap; CUT3R-style)
        if args.use_rel_pose:
            model.max_ref_frames = args.max_ref_frames
            print(f"Inference max_ref_frames={args.max_ref_frames}")
        model = model.to(args.device)
    else:
        model = STream3R.from_pretrained("yslan/STream3R").to(args.device)
    model.eval()

    eval_pose_estimation(args, model, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
