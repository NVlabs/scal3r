# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=None,
                        help="datasets to evaluate on (default: all)")
    parser.add_argument("--kf_every", type=int, default=200,
                        help="keyframe sampling interval for 7scenes/NRGBD (1=dense video)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames per sequence (0=unlimited)")
    # scal3r-specific arguments
    parser.add_argument("--use_relative_pose", action="store_true", default=False,
                        help="Use relative pose accumulation + PGO")
    parser.add_argument("--skip_pgo", action="store_true", default=False,
                        help="Skip PGO, use chain accumulation only")
    parser.add_argument("--kf_window", type=int, default=4,
                        help="Number of keyframes in buffer")
    parser.add_argument("--nkf_buffer_size", type=int, default=0)
    parser.add_argument("--pgo_sigma_rot", type=float, default=None)
    parser.add_argument("--pgo_sigma_trans", type=float, default=None)
    parser.add_argument("--max_ref_frames", type=int, default=None)
    parser.add_argument("--use_pgo1_poses", action="store_true", default=False)
    parser.add_argument("--use_online_pgo", action="store_true", default=False)
    parser.add_argument("--final_batch_opt", action="store_true", default=False)
    parser.add_argument("--reset_interval", type=int, default=1000000)
    parser.add_argument("--scene_start", type=int, default=0,
                        help="Start scene index (inclusive)")
    parser.add_argument("--scene_end", type=int, default=999,
                        help="End scene index (exclusive)")
    return parser


def main(args):
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    eval_datasets = args.eval_datasets
    datasets_all = {}
    if eval_datasets is None or "7scenes" in eval_datasets:
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT="./data/7scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
        )
    if eval_datasets is None or "NRGBD" in eval_datasets:
        datasets_all["NRGBD"] = NRGBD(
            split="test",
            ROOT="./data/neural_rgbd",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
        )

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name in ("ours", "cut3r", "scal3r"):
        from dust3r.model import ARCroco3DStereo
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy

        model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
        model.eval()

        from dust3r.inference import inference_recurrent, make_kf_only_callbacks, accumulate_poses
    else:
        raise NotImplementedError
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            all_idxs = [i for i in range(len(dataset)) if args.scene_start <= i < args.scene_end]
            with accelerator.split_between_processes(all_idxs) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    if args.max_frames > 0:
                        batch = batch[:args.max_frames]
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():  # pseudo_focal
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    if model_name in ("ours", "cut3r"):
                        # CUT3R: inference_recurrent WITHOUT callbacks (vanilla, no buffer pruning)
                        inf_views = []
                        for i, view in enumerate(batch):
                            inf_view = {
                                "img": view["img"].cpu(),
                                "ray_map": view["ray_map"].cpu(),
                                "true_shape": view["true_shape"].cpu(),
                                "idx": i,
                                "instance": str(i),
                                "camera_pose": view["camera_pose"].cpu(),
                                "img_mask": view["img_mask"].cpu(),
                                "ray_mask": view["ray_mask"].cpu(),
                                "update": view["update"].cpu(),
                                "reset": view.get("reset", torch.tensor(False).unsqueeze(0)),
                            }
                            inf_views.append(inf_view)

                        with torch.cuda.amp.autocast(enabled=False):
                            start = time.time()
                            outputs, _ = inference_recurrent(
                                inf_views, model, device,
                            )
                            end = time.time()
                            preds = outputs["pred"]

                        # Move preds to device
                        for i in range(len(preds)):
                            for key in preds[i]:
                                if isinstance(preds[i][key], torch.Tensor):
                                    preds[i][key] = preds[i][key].to(device)

                    elif model_name == "scal3r":
                        # Scal3r: inference_recurrent + relative pose accumulation
                        inf_views = []
                        for i, view in enumerate(batch):
                            inf_view = {
                                "img": view["img"].cpu(),
                                "ray_map": view["ray_map"].cpu(),
                                "true_shape": view["true_shape"].cpu(),
                                "idx": i,
                                "instance": str(i),
                                "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                                "img_mask": view["img_mask"].cpu(),
                                "ray_mask": view["ray_mask"].cpu(),
                                "update": view["update"].cpu(),
                                "reset": torch.tensor((i + 1) % args.reset_interval == 0).unsqueeze(0),
                            }
                            inf_views.append(inf_view)

                        ref_frame_indices_fn, on_frame_processed, keyframe_indices, buffer_pruning_fn = \
                            make_kf_only_callbacks(
                                kf_window=args.kf_window,
                                nkf_buffer_size=args.nkf_buffer_size,
                                pgo_sigma_rot=args.pgo_sigma_rot,
                                pgo_sigma_trans=args.pgo_sigma_trans,
                                final_batch_opt=args.final_batch_opt,
                            )

                        if args.max_ref_frames is not None:
                            model.max_ref_frames = args.max_ref_frames

                        with torch.cuda.amp.autocast(enabled=False):
                            start = time.time()
                            outputs, _ = inference_recurrent(
                                inf_views, model, device,
                                ref_frame_indices_fn=ref_frame_indices_fn,
                                keyframe_indices=keyframe_indices,
                                on_frame_processed=on_frame_processed,
                                buffer_pruning_fn=buffer_pruning_fn,
                            )
                            end = time.time()
                            preds = outputs["pred"]

                        # Accumulate relative poses → global c2w
                        accumulated_c2w = accumulate_poses(
                            preds,
                            views=outputs["views"],
                            use_relative_pose=True if args.use_relative_pose else None,
                            skip_pgo=args.skip_pgo,
                            use_pgo1_poses=args.use_pgo1_poses,
                            use_online_pgo=args.use_online_pgo,
                            pgo_sigma_rot=args.pgo_sigma_rot,
                            pgo_sigma_trans=args.pgo_sigma_trans,
                        )
                        # Replace pts3d_in_other_view with c2w-transformed pts
                        for i in range(len(preds)):
                            pts3d_self = preds[i]["pts3d_in_self_view"]
                            c2w_i = accumulated_c2w[i]
                            preds[i]["pts3d_in_other_view"] = geotrf(c2w_i, pts3d_self)

                        # Move preds to device (criterion expects same device as batch)
                        for i in range(len(preds)):
                            for key in preds[i]:
                                if isinstance(preds[i][key], torch.Tensor):
                                    preds[i][key] = preds[i][key].to(device)

                    fps = len(batch) / (end - start)
                    print(
                        f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                    )
                    fps_all.append(fps)
                    time_all.append(end - start)

                    # Evaluation (shared for all models)
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        pts = pred_pts[j].cpu().numpy()[0]
                        conf = preds[j]["conf"].cpu().data.numpy()[0]

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        #### Align predicted 3D points to the ground truth
                        pts[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts = geotrf(in_camera1, pts)

                        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts_gt = geotrf(in_camera1, pts_gt)

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}

                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all

                    np.save(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                        save_params,
                    )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                        ),
                        pcd,
                    )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                        pcd_gt,
                    )

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    pcd = pcd.transform(transformation)
                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    comp, comp_med, nc2, nc2_med = completion(
                        pcd_gt.points, pcd.points, gt_normal, pred_normal
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
