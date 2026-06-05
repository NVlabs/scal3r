# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.model import ARCroco3DStereo
from accelerate import Accelerator
import re
import numpy as np

try:
    import gtsam
    _HAS_GTSAM = True
except ImportError:
    _HAS_GTSAM = False


def custom_sort_key(key):
    text = key.split("/")
    if len(text) > 1:
        text, num = text[0], text[-1]
        return (text, int(num))
    else:
        return (key, -1)


def merge_chunk_dict(old_dict, curr_dict, add_number):
    new_dict = {}
    for key, value in curr_dict.items():

        match = re.search(r"(\d+)$", key)
        if match:

            num_part = int(match.group()) + add_number

            new_key = re.sub(r"(\d+)$", str(num_part), key, 1)
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    new_dict = old_dict | new_dict
    return {k: new_dict[k] for k in sorted(new_dict.keys(), key=custom_sort_key)}


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(
    batch,
    model,
    criterion,
    accelerator: Accelerator,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)

    with torch.amp.autocast('cuda', enabled=not inference):
        if inference:
            output, state_args = model(batch, ret_state=True)
            preds, batch = output.ress, output.views
            result = dict(views=batch, pred=preds)
            return result[ret] if ret else result, state_args
        else:
            output = model(batch)
            preds, batch = output.ress, output.views

        with torch.amp.autocast('cuda', enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None

    result = dict(views=batch, pred=preds, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch_tbptt(
    batch,
    model,
    criterion,
    chunk_size,
    loss_scaler,
    optimizer,
    accelerator: Accelerator,
    log_writer=None,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)
    all_preds = []
    all_loss = 0.0
    all_loss_details = {}
    with torch.amp.autocast('cuda', enabled=not inference):
        with torch.no_grad():
            (feat, pos, shape), (
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            ) = accelerator.unwrap_model(model)._forward_encoder(batch)
        feat = [f.detach() for f in feat]
        pos = [p.detach() for p in pos]
        shape = [s.detach() for s in shape]
        init_state_feat = init_state_feat.detach()
        init_mem = init_mem.detach()
        pose_token_buffer = []  # Sliding window buffer across chunks

        for chunk_id in range((len(batch) - 1) // chunk_size + 1):
            preds = []
            chunk = []
            state_feat = state_feat.detach()
            state_pos = state_pos.detach()
            mem = mem.detach()
            pose_token_buffer = [(idx, t.detach()) for idx, t in pose_token_buffer]  # Detach buffer at chunk boundary
            if chunk_id < ((len(batch) - 1) // chunk_size + 1) - 4:
                with torch.no_grad():
                    for in_chunk_idx in range(chunk_size):
                        i = chunk_id * chunk_size + in_chunk_idx
                        if i >= len(batch):
                            break
                        res, (state_feat, mem), pose_token_buffer = accelerator.unwrap_model(
                            model
                        )._forward_decoder_step(
                            batch,
                            i,
                            feat_i=feat[i],
                            pos_i=pos[i],
                            shape_i=shape[i],
                            init_state_feat=init_state_feat,
                            init_mem=init_mem,
                            state_feat=state_feat,
                            state_pos=state_pos,
                            mem=mem,
                            pose_token_buffer=pose_token_buffer,
                        )
                        preds.append(res)
                        all_preds.append({k: v.detach() if hasattr(v, 'detach') else v for k, v in res.items()})
                        chunk.append(batch[i])
                # Remap ref_frame_indices from global to local chunk indices
                chunk_start = chunk_id * chunk_size
                for pred in preds:
                    if "ref_frame_indices" in pred and pred["ref_frame_indices"] is not None:
                        pred["ref_frame_indices"] = [
                            ref_idx - chunk_start if chunk_start <= ref_idx < chunk_start + len(preds) else -1
                            for ref_idx in pred["ref_frame_indices"]
                        ]
                with torch.amp.autocast('cuda', enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, chunk_id * chunk_size
                    )
                    del loss
            else:
                for in_chunk_idx in range(chunk_size):
                    i = chunk_id * chunk_size + in_chunk_idx
                    if i >= len(batch):
                        break
                    res, (state_feat, mem), pose_token_buffer = accelerator.unwrap_model(
                        model
                    )._forward_decoder_step(
                        batch,
                        i,
                        feat_i=feat[i],
                        pos_i=pos[i],
                        shape_i=shape[i],
                        init_state_feat=init_state_feat,
                        init_mem=init_mem,
                        state_feat=state_feat,
                        state_pos=state_pos,
                        mem=mem,
                        pose_token_buffer=pose_token_buffer,
                    )
                    preds.append(res)
                    all_preds.append({k: v.detach() if hasattr(v, 'detach') else v for k, v in res.items()})
                    chunk.append(batch[i])
                # Remap ref_frame_indices from global to local chunk indices
                chunk_start = chunk_id * chunk_size
                for pred in preds:
                    if "ref_frame_indices" in pred and pred["ref_frame_indices"] is not None:
                        pred["ref_frame_indices"] = [
                            ref_idx - chunk_start if chunk_start <= ref_idx < chunk_start + len(preds) else -1
                            for ref_idx in pred["ref_frame_indices"]
                        ]
                with torch.amp.autocast('cuda', enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, chunk_id * chunk_size
                    )
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                        update_grad=True,
                        clip_grad=1.0,
                    )
                    optimizer.zero_grad()
                    del loss
    result = dict(
        views=batch,
        pred=all_preds,
        loss=(all_loss / ((len(batch) - 1) // chunk_size + 1), all_loss_details),
        already_backprop=True,
    )
    return result[ret] if ret else result


@torch.no_grad()
def inference(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    res, state_args = loss_of_one_batch(groups, model, None, None, inference=True)
    result = to_cpu(res)
    return result, state_args


@torch.no_grad()
def inference_step(view, state_args, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for name in view.keys():  # pseudo_focal
        if name in ignore_keys:
            continue
        if isinstance(view[name], tuple) or isinstance(view[name], list):
            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
        else:
            view[name] = view[name].to(device, non_blocking=True)

    with torch.amp.autocast('cuda', enabled=False):
        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        pred, _, _ = model.inference_step(
            view, state_feat, state_pos, init_state_feat, mem, init_mem
        )

    res = dict(pred=pred)
    result = to_cpu(res)
    return result


@torch.no_grad()
def inference_recurrent(groups, model, device, verbose=True, ref_frame_indices_fn=None,
                        keyframe_indices=None, on_frame_processed=None,
                        buffer_pruning_fn=None):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps (one at a time)")
    # Keep views on CPU initially
    cpu_views = []
    for view in groups:
        cpu_view = {}
        for name, value in view.items():
            if name in ignore_keys:
                cpu_view[name] = value
            else:
                # Keep on CPU for now
                cpu_view[name] = value
        cpu_views.append(cpu_view)
    with torch.amp.autocast('cuda', enabled=False):
        preds, batch, state_args = model.forward_recurrent(
            cpu_views, device, ret_state=True,
            ref_frame_indices_fn=ref_frame_indices_fn,
            keyframe_indices=keyframe_indices,
            on_frame_processed=on_frame_processed,
            buffer_pruning_fn=buffer_pruning_fn,
        )
        # Write final PGO #1 poses to predictions (not intermediate snapshots)
        if on_frame_processed is not None and hasattr(on_frame_processed, 'finalize'):
            on_frame_processed.finalize(preds)
        res = dict(views=batch, pred=preds)
    result = to_cpu(res)
    return result, state_args


def check_if_same_size(pairs):
    shapes1 = [img1["img"].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2["img"].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2
    )


def get_pred_pts3d(gt, pred, use_pose=False, inplace=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:

        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:

        assert use_pose is True
        return (
            pred["pts3d_in_other_view"]
            if inplace
            else pred["pts3d_in_other_view"].clone()
        )

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)

        for iter in range(10):

            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)

            w = dis.clip_(min=1e-8).reciprocal()

            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)

    return scaling


# =====================================================================
# Keyframe-only buffer: MUSt3R-style overlap NN
# =====================================================================

def make_kf_only_callbacks(**params):
    """Sliding-window keyframe buffer with 3D overlap score.

    Params:
        kf_window: 4                — number of keyframes to keep in buffer
        nkf_buffer_size: 0          — number of non-keyframes to keep (0 = pure KF)
        keyframe_overlap_thr: 0.1   — NN distance threshold for new area
        overlap_percentile: 85      — percentile of NN distances as score
        min_conf_keyframe: 1.2      — confidence gate for keyframe + point filter
        kf_x_subsamp: 4             — spatial subsampling for speed
        depth_normalize: True       — divide distances by depth (nn-norm mode)
        kf_pgo: True                — use iSAM2 PGO when GTSAM available

    Returns:
        (ref_frame_indices_fn, on_frame_processed, keyframe_indices, buffer_pruning_fn)
    """
    num_init_frames = params.get('num_init_frames', 2)
    kf_window = params.get('kf_window', 4)
    nkf_buffer_size = params.get('nkf_buffer_size', 0)
    max_ref_frames = params.get('max_ref_frames', 4)
    keyframe_indices = set()
    for i in range(num_init_frames):
        keyframe_indices.add(i)

    kf_pgo = params.get('kf_pgo', True)
    nkf_sigma_scale = params.get('nkf_sigma_scale', 1.0)
    pgo_position_scale = params.get('pgo_position_scale', 0)  # sigma grows as (1 + frame_idx / position_scale)
    pgo_max_edges = params.get('pgo_max_edges', 0)  # max constraints per frame for PGO (0=unlimited)

    # Local PGO sigma (do NOT mutate module-level globals)
    base_sigma_rot = params.get('pgo_sigma_rot') or _BASE_SIGMA_ROT
    base_sigma_trans = params.get('pgo_sigma_trans') or _BASE_SIGMA_TRANS

    # ── Loop closure ──
    loop_detector = params.get('loop_detector', None)
    loop_image_paths = params.get('loop_image_paths', None)
    _kf_feature_archive = {}  # frame_idx → (frame_idx, feat) — survives buffer pruning
    _pending_loop_frames = []  # (loop_frame_idx, score) to inject into next ref selection
    _active_loop_refs = {}    # frame_idx → score, for geometric verification in _accumulate_c2w
    loop_max_translation = params.get('loop_max_translation', 20.0)
    loop_sigma_scale = params.get('loop_sigma_scale', 1.0)
    loop_max_rot_deg = params.get('loop_max_rot_deg', 45.0)  # reject loop edges with rot > this
    loop_max_trans = params.get('loop_max_trans', 20.0)       # reject loop edges with trans > this

    # Local noise functions bound to local sigma values
    def _local_loop_noise(frame_gap, sigma_scale=1.0):
        return _make_loop_noise(frame_gap, sigma_scale, base_sigma_rot, base_sigma_trans)

    def _local_gap_noise(frame_gap, sigma_scale=1.0):
        return _make_gap_noise(frame_gap, sigma_scale, base_sigma_rot, base_sigma_trans)

    _buffer_indices = set()  # frame indices currently in the buffer
    _isam2_pgo = _ISAM2PGO(noise_fns=(_local_loop_noise, _local_gap_noise)) if (_HAS_GTSAM and kf_pgo) else None
    _constraint_buffer = []  # (j, ref_idx, rel_pose_1x4x4, sigma_scale, forced_gap_or_None)
    _pose_list = []          # indexed by frame_idx for finalize

    # ── Buffer pruning: keep most recent N keyframes + recent NKFs ──
    def kf_only_buffer_pruning(pose_token_buffer, _kf_indices, **kwargs):
        # Archive keyframe features before pruning (for loop closure re-injection)
        for idx, feat in pose_token_buffer:
            if idx in keyframe_indices and idx not in _kf_feature_archive:
                _kf_feature_archive[idx] = (idx, feat)

        if not pose_token_buffer:
            return []
        latest_idx = pose_token_buffer[-1][0]
        kf_entries = [(idx, f) for idx, f in pose_token_buffer
                      if idx in keyframe_indices]
        kf_entries = kf_entries[-kf_window:]
        non_kf = [(idx, f) for idx, f in pose_token_buffer
                  if idx not in keyframe_indices]
        recent = non_kf[-nkf_buffer_size:] if (non_kf and nkf_buffer_size > 0) else []
        seen = set()
        result = []
        for entry in kf_entries + recent:
            if entry[0] not in seen:
                seen.add(entry[0])
                result.append(entry)
        if nkf_buffer_size > 0 and latest_idx not in seen:
            result.append(pose_token_buffer[-1])
        result.sort(key=lambda x: x[0])
        _buffer_indices.clear()
        _buffer_indices.update(idx for idx, _ in result)
        return result

    _injected_loop_entries = []  # temporarily injected entries to remove after model step

    # ── Ref selection: use most recent buffer entries as references ──
    def ref_frame_indices_fn(frame_idx, pose_token_buffer):
        if frame_idx == 0:
            return None
        refs = [idx for idx, _ in pose_token_buffer if idx < frame_idx]

        # Clean up loop entries injected for the PREVIOUS frame
        if _injected_loop_entries:
            injected_ids = {idx for idx, _ in _injected_loop_entries}
            pose_token_buffer[:] = [
                (idx, f) for idx, f in pose_token_buffer if idx not in injected_ids
            ]
            _injected_loop_entries.clear()

        # Inject loop closure frames temporarily into buffer and ref list
        _active_loop_refs.clear()
        if _pending_loop_frames:
            buffer_dict = {idx for idx, _ in pose_token_buffer}
            for loop_idx, score in _pending_loop_frames:
                if loop_idx not in buffer_dict and loop_idx in _kf_feature_archive:
                    entry = _kf_feature_archive[loop_idx]
                    pose_token_buffer.append(entry)
                    _injected_loop_entries.append(entry)
                if loop_idx not in refs:
                    refs.append(loop_idx)
                _active_loop_refs[loop_idx] = score
            _pending_loop_frames.clear()

        if not refs:
            return None
        # Cap to the most-recent `max_ref_frames` refs (loop refs are appended last,
        # so they are preferentially kept). Replaces the model-side max_ref_frames cap.
        if max_ref_frames and max_ref_frames > 0 and len(refs) > max_ref_frames:
            refs = refs[-max_ref_frames:]
        return refs

    from scipy.spatial import cKDTree as _KDTree
    import numpy as _np

    min_conf_kf = params.get('min_conf_keyframe', 1.2)
    overlap_thr = params.get('keyframe_overlap_thr', 0.1)
    percentile = params.get('overlap_percentile', 85)
    kf_subsamp = params.get('kf_x_subsamp', 4)
    depth_normalize = params.get('depth_normalize', True)
    quadrant_divider = params.get('quadrant_divider', 2)

    # ── Quadrant-aware KDTree (MUSt3R-style overlap detection) ──
    _n_quadrants = 2 * quadrant_divider ** 2
    _quadrant_pts_list = [[] for _ in range(_n_quadrants)]  # list of arrays, lazy concat
    _quadrant_pts_count = [0] * _n_quadrants
    _quadrant_trees = [None] * _n_quadrants
    _quadrant_dirty = [False] * _n_quadrants  # True = new pts added since last tree build
    _frame_pts = {}  # frame_idx -> {quadrant_id: np.ndarray} — per-frame point ownership
    _overlap_frame_set = set()  # frame indices currently in the overlap tree

    def _get_quadrant_id(rays, eps=1e-5):
        rays = rays / _np.linalg.norm(rays, axis=-1, keepdims=True).clip(eps)
        thetas = (_np.arccos(rays[:, -1]) / _np.pi).clip(eps, 1 - eps)
        phis = (_np.arctan2(rays[:, 1], rays[:, 0]) / _np.pi).clip(-1 + eps, 1 - eps)
        theta_idx = _np.floor(thetas * quadrant_divider).astype(int)
        phis_idx = _np.floor(phis * quadrant_divider).astype(int) + quadrant_divider
        return (theta_idx + phis_idx * quadrant_divider).astype(int)

    def _add_to_overlap_tree(pts_world_np, cam_center_np, frame_idx=None):
        rays = pts_world_np - cam_center_np[None]
        quad_ids = _get_quadrant_id(rays)
        per_quad = {}
        for q in _np.unique(quad_ids):
            mask = quad_ids == q
            pts_q = pts_world_np[mask]
            _quadrant_pts_list[q].append(pts_q)
            _quadrant_pts_count[q] += pts_q.shape[0]
            _quadrant_dirty[q] = True
            per_quad[q] = pts_q
        if frame_idx is not None:
            _frame_pts[frame_idx] = per_quad
            _overlap_frame_set.add(frame_idx)

    def _remove_from_overlap_tree(frame_idx):
        if frame_idx not in _frame_pts:
            return
        per_quad = _frame_pts.pop(frame_idx)
        _overlap_frame_set.discard(frame_idx)
        # Rebuild affected quadrants from remaining frames
        for q in per_quad:
            # Collect all pts from remaining frames for this quadrant
            new_pts = []
            for fid in _overlap_frame_set:
                fq = _frame_pts.get(fid, {})
                if q in fq:
                    new_pts.append(fq[q])
            _quadrant_pts_list[q] = new_pts if new_pts else []
            _quadrant_pts_count[q] = sum(p.shape[0] for p in new_pts)
            _quadrant_dirty[q] = True

    def _sync_overlap_tree_with_buffer():
        """Remove points from frames no longer in buffer."""
        to_remove = _overlap_frame_set - _buffer_indices
        # Keep init frames (they are always useful for overlap reference)
        to_remove -= set(range(num_init_frames))
        for fid in to_remove:
            _remove_from_overlap_tree(fid)

    def _rebuild_dirty_trees():
        for q in range(_n_quadrants):
            if _quadrant_dirty[q]:
                if _quadrant_pts_list[q]:
                    all_pts = _np.concatenate(_quadrant_pts_list[q])
                    _quadrant_pts_list[q] = [all_pts]
                    _quadrant_trees[q] = _KDTree(all_pts, balanced_tree=False, compact_nodes=False)
                else:
                    _quadrant_trees[q] = None
                _quadrant_dirty[q] = False

    def _query_overlap_tree(pts_world_np, cam_center_np):
        _rebuild_dirty_trees()
        rays = pts_world_np - cam_center_np[None]
        quad_ids = _get_quadrant_id(rays)
        dists = _np.full(pts_world_np.shape[0], _np.inf)
        # Sort by quadrant once, then slice — avoids repeated boolean mask scans
        order = _np.argsort(quad_ids, kind='mergesort')
        sorted_ids = quad_ids[order]
        splits = _np.searchsorted(sorted_ids, _np.arange(_n_quadrants + 1))
        for q in range(_n_quadrants):
            lo, hi = splits[q], splits[q + 1]
            if lo == hi:
                continue
            tree = _quadrant_trees[q]
            if tree is not None:
                d, _ = tree.query(pts_world_np[order[lo:hi]], k=1, workers=4)
                dists[order[lo:hi]] = d
        return dists

    _pose_history = {}       # frame_idx -> (4, 4) c2w tensor (CPU)
    _chain_history = {}      # pure chain history (never modified by PGO)

    def _run_window_lm(window_indices):
        """Sliding window LM: optimize only the poses in window_indices.
        Anchor the oldest pose, use constraints where both endpoints are in window."""
        if not _HAS_GTSAM or len(window_indices) < 2:
            return
        window = sorted(window_indices)
        window_set = set(window)
        edges = []
        for entry in _constraint_buffer:
            j, ref_idx = entry[0], entry[1]
            if j in window_set and ref_idx in window_set:
                edges.append(entry)
        if not edges:
            return
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        for idx in window:
            key = gtsam.symbol('x', idx)
            if idx in _pose_history:
                initial.insert(key, _torch_c2w_to_gtsam_pose3(_pose_history[idx]))
            else:
                initial.insert(key, gtsam.Pose3())
        anchor = window[0]
        prior_sigmas = np.array([1e-6] * 6)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        graph.addPriorPose3(gtsam.symbol('x', anchor),
                            initial.atPose3(gtsam.symbol('x', anchor)),
                            prior_noise)
        for entry in edges:
            j, ref_idx, rel_pose, ss = entry[:4]
            forced_gap = entry[4] if len(entry) > 4 else None
            is_loop = entry[5] if len(entry) > 5 else False
            T_rel = rel_pose.squeeze(0) if rel_pose.dim() == 3 else rel_pose
            gap = forced_gap if forced_gap is not None else abs(j - ref_idx)
            measurement = _make_between_measurement(T_rel)
            noise_fn = _local_loop_noise if is_loop else _local_gap_noise
            noise = noise_fn(gap, sigma_scale=ss)
            graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('x', ref_idx), gtsam.symbol('x', j),
                measurement, noise))
        try:
            lm_params = gtsam.LevenbergMarquardtParams()
            lm_params.setMaxIterations(20)
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
            estimate = optimizer.optimize()
        except Exception:
            return
        max_rot_change = 0.0
        for idx in window:
            key = gtsam.symbol('x', idx)
            if estimate.exists(key):
                new_pose = _gtsam_pose3_to_torch(estimate.atPose3(key), batch=False)
                if idx in _pose_history:
                    R_old = _pose_history[idx][:3, :3]
                    R_new = new_pose[:3, :3]
                    cos_a = ((R_old.T @ R_new).trace() - 1) / 2
                    rot_change = cos_a.clamp(-1, 1).acos().item() * 180 / 3.14159265
                    max_rot_change = max(max_rot_change, rot_change)
                _pose_history[idx] = new_pose
        if window[-1] % 200 == 0:
            print(f"  [WindowLM] newest={window[-1]}: max_rot_change={max_rot_change:.2f}deg, {len(edges)} edges")

    def _accumulate_c2w(frame_idx, result):
        """Chain accumulation with iSAM2 PGO refinement.

        Maintains two independent pose histories:
        - _chain_history: pure chain accumulation (never touched by PGO)
        - _pose_history: PGO-assisted chain (used for subsequent chain + PGO)

        Returns:
            (chain_c2w, pgo_c2w): pure chain pose and PGO-refined pose.
        """
        if frame_idx == 0:
            c2w = torch.eye(4, dtype=torch.float32)
            _chain_history[0] = c2w.clone()
            _pose_history[0] = c2w
            if kf_pgo:
                _pose_list.append(c2w.unsqueeze(0))
            if _isam2_pgo is not None:
                _isam2_pgo.add_pose(0, c2w, is_anchor=True)
            return c2w, c2w

        rel_poses = result.get('relative_poses')
        ref_indices = result.get('ref_frame_indices')

        # Chain accumulation + collect constraints
        c2w = None       # PGO-assisted chain (uses _pose_history)
        chain_c2w = None # pure chain (uses _chain_history)
        if rel_poses is not None and ref_indices is not None:
            K = rel_poses.shape[1]
            for k, ref_idx in enumerate(ref_indices):
                if k >= K:
                    break
                T_rel = rel_poses[0, k].cpu().float()

                # Loop edges: constraint only, no chain accumulation
                is_loop_edge = ref_idx in _active_loop_refs
                if is_loop_edge:
                    loop_score = _active_loop_refs[ref_idx]
                    _has_loop_edges[0] = True
                    real_gap = abs(frame_idx - ref_idx)
                    # Use fixed sigma_scale for loop edges (same magnitude as
                    # sequential edges). The model's relative pose prediction
                    # quality for long-range pairs is not better than for
                    # sequential ones, so loop edges should not be tighter.
                    ss = loop_sigma_scale
                    forced_gap = 1
                    # Debug: print predicted T_rel for loop edge
                    trans = T_rel[:3, 3].norm().item()
                    R = T_rel[:3, :3]
                    cos_a = ((R.trace() - 1) / 2).clamp(-1, 1)
                    rot_deg = cos_a.acos().item() * 180 / 3.14159265
                    print(f"  Loop edge T_rel: frame {frame_idx} <-> {ref_idx}, "
                          f"trans={trans:.2f}m, rot={rot_deg:.1f}deg, ss={ss:.4f}")
                    # Hard rejection only for extreme outliers (likely model failures)
                    if rot_deg > loop_max_rot_deg or trans > loop_max_trans:
                        print(f"  ** REJECTED loop edge: rot={rot_deg:.1f}>{loop_max_rot_deg} or trans={trans:.2f}>{loop_max_trans}")
                        continue
                    if kf_pgo:
                        _constraint_buffer.append(
                            (frame_idx, ref_idx, T_rel.unsqueeze(0), ss, forced_gap, True)
                        )
                    if _isam2_pgo is not None:
                        _isam2_pgo.add_constraint(frame_idx, ref_idx, T_rel, sigma_scale=ss, forced_gap=forced_gap, is_loop=True)
                    continue

                # Normal sequential edges
                is_j_kf = frame_idx in keyframe_indices
                is_ref_kf = ref_idx in keyframe_indices
                ss = 1.0 if (is_j_kf or is_ref_kf) else nkf_sigma_scale
                if pgo_position_scale > 0:
                    ss *= (1.0 + frame_idx / pgo_position_scale)
                pgo_edge_ok = (pgo_max_edges <= 0 or k < pgo_max_edges)
                forced_gap = None
                if kf_pgo and pgo_edge_ok:
                    _constraint_buffer.append(
                        (frame_idx, ref_idx, T_rel.unsqueeze(0), ss, forced_gap)
                    )
                if _isam2_pgo is not None and pgo_edge_ok:
                    _isam2_pgo.add_constraint(frame_idx, ref_idx, T_rel, sigma_scale=ss, forced_gap=forced_gap)
                T_rel_inv = _se3_inverse(T_rel)
                if c2w is None and ref_idx in _pose_history:
                    c2w = _reorthogonalize_c2w(_pose_history[ref_idx] @ T_rel_inv)
                if chain_c2w is None and ref_idx in _chain_history:
                    chain_c2w = _reorthogonalize_c2w(_chain_history[ref_idx] @ T_rel_inv)

        # Fallback for PGO-assisted chain
        # Prefer previous frame's pose (stays in relative chain coordinate system)
        # over camera_pose (CUT3R absolute pose, different coordinate system)
        if c2w is None:
            if (frame_idx - 1) in _pose_history:
                c2w = _pose_history[frame_idx - 1].clone()
                # No refs = overlap frame after reset (same image as previous frame).
                # Add tight identity constraint so iSAM2 graph stays strongly connected.
                if _isam2_pgo is not None:
                    identity = torch.eye(4, dtype=torch.float32)
                    _isam2_pgo.add_constraint(
                        frame_idx, frame_idx - 1, identity,
                        sigma_scale=0.001, forced_gap=1, is_loop=True)  # no Huber, very tight
            else:
                c2w = torch.eye(4, dtype=torch.float32)

        # Fallback for pure chain
        if chain_c2w is None:
            if (frame_idx - 1) in _chain_history:
                chain_c2w = _chain_history[frame_idx - 1].clone()
            else:
                chain_c2w = c2w.clone()

        _chain_history[frame_idx] = chain_c2w.clone()
        _pose_history[frame_idx] = c2w

        # Incremental iSAM2 PGO
        if kf_pgo and len(_constraint_buffer) > 0:
            _pose_list.append(c2w.unsqueeze(0))

            if _isam2_pgo is not None and not _isam2_pgo.is_broken:
                _isam2_pgo.add_pose(frame_idx, c2w)
                _isam2_pgo.optimize()
                if not _isam2_pgo.is_broken:
                    active = _buffer_indices | {frame_idx}
                    for idx in active:
                        opt = _isam2_pgo.get_pose(idx)
                        if opt is not None:
                            _pose_history[idx] = opt.squeeze(0)
                            if idx < len(_pose_list):
                                _pose_list[idx] = opt
        elif kf_pgo:
            _pose_list.append(c2w.unsqueeze(0))

        return chain_c2w, _pose_history[frame_idx]

    def _compute_overlap_score(pts_world_np, depths_np, cam_center_np):
        has_any_pts = any(len(pl) > 0 for pl in _quadrant_pts_list)
        if not has_any_pts:
            return float('inf')
        dists = _query_overlap_tree(pts_world_np, cam_center_np)
        if depth_normalize:
            dists = dists / (_np.abs(depths_np) + 1e-9)
        dists[_np.isposinf(dists)] = _np.finfo(dists.dtype).max
        # O(n) partial sort instead of O(n log n) full percentile
        n = len(dists)
        k = int(_np.ceil(n * percentile / 100.0)) - 1
        k = max(0, min(k, n - 1))
        return float(_np.partition(dists, k)[k])

    def _extract_world_pts(result, c2w, subsamp):
        pts3d_local = result.get('pts3d_in_self_view')
        conf = result.get('conf_self')
        if pts3d_local is None or conf is None:
            return None, None
        # Subsample and mask on GPU before transfer
        pts_gpu = pts3d_local[0]
        c_gpu = conf[0]
        if subsamp:
            pts_gpu = pts_gpu[::subsamp, ::subsamp]
            c_gpu = c_gpu[::subsamp, ::subsamp]
        msk = c_gpu > min_conf_kf
        if msk.sum() == 0:
            return None, None
        pts_masked = pts_gpu[msk].float()
        depths = pts_masked[:, 2].cpu().numpy()
        # World transform on GPU: R @ pts^T + t (avoids homogeneous coord allocation)
        c2w_dev = c2w.to(pts_masked.device) if c2w.device != pts_masked.device else c2w
        R = c2w_dev[:3, :3]
        t = c2w_dev[:3, 3]
        pts_world = (R @ pts_masked.T).T + t
        return pts_world.cpu().numpy(), depths

    _has_loop_edges = [False]

    def on_frame_processed(frame_idx, result):
        chain_c2w, pgo_c2w = _accumulate_c2w(frame_idx, result)
        result['chain_c2w'] = chain_c2w
        result['online_pgo_c2w'] = pgo_c2w
        c2w = pgo_c2w
        pts_world, depths = _extract_world_pts(result, c2w, kf_subsamp)
        cam_center = c2w[:3, 3].numpy()

        if frame_idx < num_init_frames:
            if pts_world is not None:
                _add_to_overlap_tree(pts_world, cam_center, frame_idx=frame_idx)
            return

        if pts_world is None:
            return

        # Sync overlap tree with current buffer (remove old frames' points)
        _sync_overlap_tree_with_buffer()

        overlap_score = _compute_overlap_score(pts_world, depths, cam_center)
        conf = result.get('conf_self')
        median_conf = conf[0].median().item() if conf is not None else 0.0
        is_kf = (overlap_score > overlap_thr) and (median_conf > min_conf_kf)

        if is_kf:
            keyframe_indices.add(frame_idx)
            _add_to_overlap_tree(pts_world, cam_center, frame_idx=frame_idx)

        # Online loop closure detection (only KFs query and add to index)
        if (is_kf and loop_detector is not None and loop_image_paths is not None
                and frame_idx < len(loop_image_paths)):
            img_tensor = result.get('_img_tensor')
            loops = loop_detector.add_and_query(
                frame_idx, loop_image_paths[frame_idx],
                add_to_index=True, img_tensor=img_tensor)
            for loop_idx, score in loops:
                if loop_idx in _kf_feature_archive:
                    _pending_loop_frames.append((loop_idx, score))
                    print(f"  Loop candidate: frame {frame_idx} <-> {loop_idx} "
                          f"(score={score:.3f})")

        # Reset loop flag (iSAM2 handles loop edges incrementally)
        if _has_loop_edges[0]:
            _has_loop_edges[0] = False

    def _finalize_predictions(predictions):
        """Write final poses to all predictions.

        - online_pgo_c2w: refresh from final _pose_history
        - kf_pgo_c2w: iSAM2 global estimate (reads ALL nodes, not just buffer)
        """
        for i in range(len(predictions)):
            if i in _pose_history:
                predictions[i]['online_pgo_c2w'] = _pose_history[i].clone()

        if not kf_pgo or len(_pose_list) == 0:
            return
        _do_finalize = params.get('do_finalize', True)
        if _do_finalize and _isam2_pgo is not None and not _isam2_pgo.is_broken:
            # Run extra iSAM2 iterations so loop edge corrections fully propagate.
            _isam2_pgo.finalize(extra_iterations=50)
            # Read ALL frame poses from iSAM2 (not just buffer frames).
            # During inference, only buffer frames are read back after each
            # optimize() call. Old frames that were corrected by loop edges
            # still have stale values in _pose_history.
            all_indices = list(range(min(len(predictions), len(_pose_list))))
            optimized = _isam2_pgo.get_poses(all_indices)
            for i in all_indices:
                if i in optimized:
                    predictions[i]['kf_pgo_c2w'] = optimized[i].squeeze(0)
                else:
                    predictions[i]['kf_pgo_c2w'] = _pose_list[i].squeeze(0)
        else:
            for i in range(min(len(predictions), len(_pose_list))):
                predictions[i]['kf_pgo_c2w'] = _pose_list[i].squeeze(0)

    on_frame_processed.finalize = _finalize_predictions

    return ref_frame_indices_fn, on_frame_processed, keyframe_indices, kf_only_buffer_pruning



# =====================================================================
# GTSAM helpers
# =====================================================================

def _se3_inverse(T):
    """Compute SE(3) inverse using R^T instead of generic matrix inverse.
    This preserves orthogonality of the rotation part.
    Input/output: (4,4) tensor."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _reorthogonalize_c2w(T):
    """Re-orthogonalize rotation part of a 4x4 SE(3) matrix via SVD.
    Prevents numerical drift from accumulated matrix multiplications."""
    R = T[:3, :3]
    U, _, Vh = torch.linalg.svd(R)
    R_ortho = U @ Vh
    # Ensure det = +1 (proper rotation)
    if torch.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vh
    T_out = T.clone()
    T_out[:3, :3] = R_ortho
    return T_out


def _torch_c2w_to_gtsam_pose3(T):
    """Convert a (4,4) or (1,4,4) c2w torch tensor to gtsam.Pose3."""
    if T.dim() == 3:
        T = T.squeeze(0)
    M = T.cpu().double().numpy()
    R = gtsam.Rot3(M[:3, :3])
    t = gtsam.Point3(M[0, 3], M[1, 3], M[2, 3])
    return gtsam.Pose3(R, t)


def _gtsam_pose3_to_torch(pose3, batch=True):
    """Convert gtsam.Pose3 to a torch float32 tensor (4,4) or (1,4,4)."""
    M = pose3.matrix()  # 4x4 numpy
    T = torch.from_numpy(M).float()
    if batch:
        T = T.unsqueeze(0)
    return T


def _make_between_measurement(T_rel):
    """Model outputs T_rel = inv(c2w_curr) @ c2w_ref.
    GTSAM BetweenFactorPose3(ref, curr) expects inv(c2w_ref) @ c2w_curr = inv(T_rel).
    Input: (4,4) or (1,4,4) torch tensor. Returns gtsam.Pose3."""
    if T_rel.dim() == 3:
        T_rel = T_rel.squeeze(0)
    T_rel_inv = _se3_inverse(T_rel.float()).cpu().double().numpy()
    R = gtsam.Rot3(T_rel_inv[:3, :3])
    t = gtsam.Point3(T_rel_inv[0, 3], T_rel_inv[1, 3], T_rel_inv[2, 3])
    return gtsam.Pose3(R, t)


# Base sigmas for gap-dependent noise
_BASE_SIGMA_ROT = 0.5     # ~28.6 deg
_BASE_SIGMA_TRANS = 0.5   # 50 cm
_PGO_MODE = 'huber'
_ROT_GAP_POWER = 0.5
_TRANS_GAP_POWER = 0.5


def _make_robust_kernel(mode='huber'):
    """Create a GTSAM robust kernel by name."""
    if mode == 'dcs':
        return gtsam.noiseModel.mEstimator.DCS.Create(1.0)
    elif mode == 'cauchy':
        return gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
    elif mode == 'tukey':
        return gtsam.noiseModel.mEstimator.Tukey.Create(4.685)
    elif mode == 'huber':
        return gtsam.noiseModel.mEstimator.Huber.Create(1.345)
    else:
        return gtsam.noiseModel.mEstimator.Huber.Create(1.345)


def _make_loop_noise(frame_gap, sigma_scale=1.0, base_sigma_rot=_BASE_SIGMA_ROT, base_sigma_trans=_BASE_SIGMA_TRANS):
    """Build noise model for loop edges with Huber robust kernel."""
    gap = max(abs(frame_gap), 1)
    rot_scale = (gap ** _ROT_GAP_POWER) * sigma_scale
    trans_scale = (gap ** _TRANS_GAP_POWER) * sigma_scale
    sr = base_sigma_rot * rot_scale
    st = base_sigma_trans * trans_scale
    sigmas = np.array([sr] * 3 + [st] * 3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), noise)
    return noise


def _make_gap_noise(frame_gap, sigma_scale=1.0, base_sigma_rot=_BASE_SIGMA_ROT, base_sigma_trans=_BASE_SIGMA_TRANS):
    """Build gap-dependent noise model with Huber kernel for BetweenFactorPose3."""
    gap = max(abs(frame_gap), 1)
    rot_scale = (gap ** _ROT_GAP_POWER) * sigma_scale
    trans_scale = (gap ** _TRANS_GAP_POWER) * sigma_scale
    sr = base_sigma_rot * rot_scale
    st = base_sigma_trans * trans_scale
    sigmas = np.array([sr] * 3 + [st] * 3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    noise = gtsam.noiseModel.Robust.Create(_make_robust_kernel(_PGO_MODE), noise)
    return noise


class _ISAM2PGO:
    """iSAM2 incremental PGO wrapper.

    Constraints referencing a not-yet-added frame are buffered and flushed
    when that frame's initial value is inserted via add_pose().
    """

    def __init__(self, noise_fns=None):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.1)
        params.relinearizeSkip = 1
        self._isam2 = gtsam.ISAM2(params)
        prior_sigmas = np.array([1e-6] * 6)
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        self._added_keys = set()
        self._current_estimate = None
        self._pending_factors = []
        self._broken = False  # True after first numerical failure
        if noise_fns is not None:
            self._loop_noise_fn, self._gap_noise_fn = noise_fns
        else:
            self._loop_noise_fn, self._gap_noise_fn = _make_loop_noise, _make_gap_noise

    @property
    def is_broken(self):
        return self._broken

    def _key(self, frame_idx):
        return gtsam.symbol('x', frame_idx)

    def _safe_update(self, graph, values=None):
        """Wrap iSAM2 update with numerical error handling.
        On first failure, permanently disables this PGO instance."""
        if self._broken:
            return False
        try:
            if values is not None:
                self._isam2.update(graph, values)
            else:
                self._isam2.update(graph, gtsam.Values())
            return True
        except Exception as e:
            if "Indeterminant" in str(type(e).__name__) or "Indeterminant" in str(e):
                print(f"[PGO] iSAM2 numerical error — disabling PGO for this sequence: {type(e).__name__}")
                self._broken = True
                return False
            raise

    def add_pose(self, frame_idx, c2w_tensor, is_anchor=False):
        """Add a pose node with initial value, flush pending constraints."""
        if self._broken:
            return
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        key = self._key(frame_idx)
        pose = _torch_c2w_to_gtsam_pose3(c2w_tensor)
        values.insert(key, pose)
        if is_anchor:
            graph.addPriorPose3(key, pose, self._prior_noise)
        self._added_keys.add(frame_idx)

        # Flush pending constraints that now have both keys available
        still_pending = []
        n_flushed = 0
        for entry in self._pending_factors:
            ri, fi, meas, gap, ss = entry[:5]
            is_loop = entry[5] if len(entry) > 5 else False
            if ri in self._added_keys and fi in self._added_keys:
                noise_fn = self._loop_noise_fn if is_loop else self._gap_noise_fn
                noise = noise_fn(gap, sigma_scale=ss)
                graph.add(gtsam.BetweenFactorPose3(
                    self._key(ri), self._key(fi), meas, noise))
                n_flushed += 1
            else:
                still_pending.append(entry)
        self._pending_factors = still_pending

        # If non-anchor pose has no factors, add a weak prior
        if not is_anchor and n_flushed == 0 and graph.nrFactors() == 0:
            weak_sigmas = np.array([0.5] * 3 + [2.0] * 3)
            weak_noise = gtsam.noiseModel.Diagonal.Sigmas(weak_sigmas)
            graph.addPriorPose3(key, pose, weak_noise)

        self._safe_update(graph, values)

    def add_constraint(self, frame_idx, ref_idx, T_rel, sigma_scale=1.0, forced_gap=None, is_loop=False):
        """Add a BetweenFactorPose3 constraint. Buffers if keys not yet added."""
        if self._broken:
            return
        measurement = _make_between_measurement(T_rel)
        gap = forced_gap if forced_gap is not None else abs(frame_idx - ref_idx)
        noise_fn = self._loop_noise_fn if is_loop else self._gap_noise_fn
        if ref_idx in self._added_keys and frame_idx in self._added_keys:
            graph = gtsam.NonlinearFactorGraph()
            noise = noise_fn(gap, sigma_scale=sigma_scale)
            graph.add(gtsam.BetweenFactorPose3(
                self._key(ref_idx), self._key(frame_idx), measurement, noise))
            self._safe_update(graph)
        else:
            self._pending_factors.append(
                (ref_idx, frame_idx, measurement, gap, sigma_scale, is_loop))

    def optimize(self, extra_iterations=2):
        """Run additional iSAM2 update iterations for convergence."""
        if self._broken:
            return self._current_estimate
        try:
            for _ in range(extra_iterations):
                self._isam2.update()
            self._current_estimate = self._isam2.calculateEstimate()
        except Exception as e:
            if "Indeterminant" in str(type(e).__name__) or "Indeterminant" in str(e):
                print(f"[PGO] iSAM2 optimize error — disabling PGO: {type(e).__name__}")
                self._broken = True
            else:
                raise
        return self._current_estimate

    def finalize(self, extra_iterations=50):
        """Run many more iSAM2 iterations at the end for global convergence.

        After all loop edges are added, the incremental updates may not have
        fully propagated corrections through the entire graph.  Running extra
        iterations here lets iSAM2 converge before we read out final poses.
        """
        if self._broken or self._current_estimate is None:
            return self._current_estimate
        try:
            for _ in range(extra_iterations):
                self._isam2.update()
            self._current_estimate = self._isam2.calculateEstimate()
        except Exception as e:
            if "Indeterminant" in str(type(e).__name__) or "Indeterminant" in str(e):
                print(f"[PGO] iSAM2 finalize error: {type(e).__name__}")
                self._broken = True
            else:
                raise
        return self._current_estimate

    def get_pose(self, frame_idx):
        """Get optimized pose for a single frame as (1,4,4) torch tensor."""
        if self._current_estimate is None:
            return None
        key = self._key(frame_idx)
        if not self._current_estimate.exists(key):
            return None
        return _gtsam_pose3_to_torch(self._current_estimate.atPose3(key), batch=True)

    def get_poses(self, frame_indices):
        """Get optimized poses for multiple frames. Returns dict {idx: (1,4,4) tensor}."""
        if self._current_estimate is None:
            return {}
        result = {}
        for idx in frame_indices:
            key = self._key(idx)
            if self._current_estimate.exists(key):
                result[idx] = _gtsam_pose3_to_torch(
                    self._current_estimate.atPose3(key), batch=True)
        return result


def accumulate_poses(predictions, views=None, use_relative_pose=None, skip_pgo=False, **kwargs):
    """Read accumulated c2w poses from predictions.

    Priority: kf_pgo_c2w > online_pgo_c2w > chain_c2w > camera_pose.
    Flags:
      skip_pgo=True        → force pure chain_c2w (no PGO at all)
      use_online_pgo=True  → force online_pgo_c2w (sliding window PGO)
      default              → kf_pgo_c2w (final iSAM2 global solution)

    Returns:
        list of (B, 4, 4) tensors, one per frame (c2w poses)
    """
    from dust3r.utils.camera import pose_encoding_to_camera
    use_online_pgo = kwargs.get('use_online_pgo', False)

    if len(predictions) == 0:
        return []

    has_online_pgo = 'online_pgo_c2w' in predictions[0]
    has_final_pgo = 'kf_pgo_c2w' in predictions[0]
    has_chain = 'chain_c2w' in predictions[0]

    # Auto-detect: only use relative pose if the model actually produced relative_poses
    # (chain_c2w/kf_pgo_c2w may be injected by callbacks even for models without relative_poses)
    has_relative_poses = 'relative_poses' in predictions[0]
    if use_relative_pose is None:
        use_relative_pose = has_relative_poses and (has_online_pgo or has_final_pgo or has_chain)
    elif use_relative_pose and not (has_online_pgo or has_final_pgo or has_chain):
        print("Warning: use_relative_pose=True but no c2w found. Falling back.")
        use_relative_pose = False

    if use_relative_pose:
        B = predictions[0]["pts3d_in_self_view"].shape[0]
        # Select pose key based on flags
        if skip_pgo:
            pose_key = 'chain_c2w'
            label = "pure chain"
        elif use_online_pgo and has_online_pgo:
            pose_key = 'online_pgo_c2w'
            label = "online PGO (sliding window)"
        elif has_final_pgo:
            pose_key = 'kf_pgo_c2w'
            label = "global PGO"
        elif has_online_pgo:
            pose_key = 'online_pgo_c2w'
            label = "online PGO (partial)"
        else:
            pose_key = 'chain_c2w'
            label = "chain"
        pr_poses = []
        for pred in predictions:
            c2w = pred[pose_key]
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0).expand(B, -1, -1)
            pr_poses.append(c2w.clone())
        print(f"Using {label} accumulation ({len(pr_poses)} frames)")
    else:
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in predictions
        ]
        print(f"Using direct camera_pose ({len(pr_poses)} frames)")

    return pr_poses
