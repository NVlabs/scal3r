# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pose-graph optimization utilities for Scal3R-CUT3R.

Provides:
    _ISAM2PGO             — iSAM2 incremental PGO wrapper
    make_kf_only_callbacks — keyframe selection + PGO callback factory
    accumulate_poses       — read accumulated c2w poses from predictions
"""

import numpy as np
import torch

try:
    import gtsam
    _HAS_GTSAM = True
except ImportError:
    _HAS_GTSAM = False


# =====================================================================
# Base noise parameters
# =====================================================================

_BASE_SIGMA_ROT = 0.5     # ~28.6 deg
_BASE_SIGMA_TRANS = 0.5   # 50 cm
_PGO_MODE = 'huber'
_ROT_GAP_POWER = 0.5
_TRANS_GAP_POWER = 0.5


# =====================================================================
# SE(3) math helpers
# =====================================================================

def _se3_inverse(T):
    """Compute SE(3) inverse using R^T instead of generic matrix inverse.
    Input/output: (4,4) tensor."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _reorthogonalize_c2w(T):
    """Re-orthogonalize rotation part of a 4x4 SE(3) matrix via SVD."""
    R = T[:3, :3]
    U, _, Vh = torch.linalg.svd(R)
    R_ortho = U @ Vh
    if torch.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vh
    T_out = T.clone()
    T_out[:3, :3] = R_ortho
    return T_out


# =====================================================================
# GTSAM conversion helpers
# =====================================================================

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
    M = pose3.matrix()
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


# =====================================================================
# Noise models
# =====================================================================

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


# =====================================================================
# iSAM2 incremental PGO
# =====================================================================

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
        self._broken = False
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
        """Wrap iSAM2 update with numerical error handling."""
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
        """Run many more iSAM2 iterations at the end for global convergence."""
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
    from scipy.spatial import cKDTree as _KDTree
    import numpy as _np

    num_init_frames = params.get('num_init_frames', 2)
    kf_window = params.get('kf_window', 4)
    nkf_buffer_size = params.get('nkf_buffer_size', 0)
    max_ref_frames = params.get('max_ref_frames', 4)
    keyframe_indices = set()
    for i in range(num_init_frames):
        keyframe_indices.add(i)

    kf_pgo = params.get('kf_pgo', True)
    nkf_sigma_scale = params.get('nkf_sigma_scale', 1.0)
    pgo_position_scale = params.get('pgo_position_scale', 0)
    pgo_max_edges = params.get('pgo_max_edges', 0)

    base_sigma_rot = params.get('pgo_sigma_rot') or _BASE_SIGMA_ROT
    base_sigma_trans = params.get('pgo_sigma_trans') or _BASE_SIGMA_TRANS

    # ── Loop closure ──
    loop_detector = params.get('loop_detector', None)
    loop_image_paths = params.get('loop_image_paths', None)
    _kf_feature_archive = {}
    _pending_loop_frames = []
    _active_loop_refs = {}
    loop_sigma_scale = params.get('loop_sigma_scale', 1.0)
    loop_max_rot_deg = params.get('loop_max_rot_deg', 45.0)
    loop_max_trans = params.get('loop_max_trans', 20.0)

    def _local_loop_noise(frame_gap, sigma_scale=1.0):
        return _make_loop_noise(frame_gap, sigma_scale, base_sigma_rot, base_sigma_trans)

    def _local_gap_noise(frame_gap, sigma_scale=1.0):
        return _make_gap_noise(frame_gap, sigma_scale, base_sigma_rot, base_sigma_trans)

    _buffer_indices = set()
    _isam2_pgo = _ISAM2PGO(noise_fns=(_local_loop_noise, _local_gap_noise)) if (_HAS_GTSAM and kf_pgo) else None
    _constraint_buffer = []
    _pose_list = []

    # ── Buffer pruning: keep most recent N keyframes + recent NKFs ──
    def kf_only_buffer_pruning(pose_token_buffer, _kf_indices, **kwargs):
        for idx, feat in pose_token_buffer:
            if idx in keyframe_indices and idx not in _kf_feature_archive:
                _kf_feature_archive[idx] = (idx, feat)

        if not pose_token_buffer:
            return []
        kf_entries = [(idx, f) for idx, f in pose_token_buffer if idx in keyframe_indices]
        kf_entries = kf_entries[-kf_window:]
        non_kf = [(idx, f) for idx, f in pose_token_buffer if idx not in keyframe_indices]
        recent = non_kf[-nkf_buffer_size:] if (non_kf and nkf_buffer_size > 0) else []
        seen = set()
        result = []
        for entry in kf_entries + recent:
            if entry[0] not in seen:
                seen.add(entry[0])
                result.append(entry)
        latest_idx = pose_token_buffer[-1][0]
        if nkf_buffer_size > 0 and latest_idx not in seen:
            result.append(pose_token_buffer[-1])
        result.sort(key=lambda x: x[0])
        _buffer_indices.clear()
        _buffer_indices.update(idx for idx, _ in result)
        return result

    _injected_loop_entries = []

    # ── Ref selection: use most recent buffer entries as references ──
    def ref_frame_indices_fn(frame_idx, pose_token_buffer):
        if frame_idx == 0:
            return None
        refs = [idx for idx, _ in pose_token_buffer if idx < frame_idx]

        if _injected_loop_entries:
            injected_ids = {idx for idx, _ in _injected_loop_entries}
            pose_token_buffer[:] = [
                (idx, f) for idx, f in pose_token_buffer if idx not in injected_ids
            ]
            _injected_loop_entries.clear()

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
        if max_ref_frames and max_ref_frames > 0 and len(refs) > max_ref_frames:
            refs = refs[-max_ref_frames:]
        return refs

    min_conf_kf = params.get('min_conf_keyframe', 1.2)
    overlap_thr = params.get('keyframe_overlap_thr', 0.1)
    percentile = params.get('overlap_percentile', 85)
    kf_subsamp = params.get('kf_x_subsamp', 4)
    depth_normalize = params.get('depth_normalize', True)
    quadrant_divider = params.get('quadrant_divider', 2)

    # ── Quadrant-aware KDTree (MUSt3R-style overlap detection) ──
    _n_quadrants = 2 * quadrant_divider ** 2
    _quadrant_pts_list = [[] for _ in range(_n_quadrants)]
    _quadrant_pts_count = [0] * _n_quadrants
    _quadrant_trees = [None] * _n_quadrants
    _quadrant_dirty = [False] * _n_quadrants
    _frame_pts = {}
    _overlap_frame_set = set()

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
        for q in per_quad:
            new_pts = []
            for fid in _overlap_frame_set:
                fq = _frame_pts.get(fid, {})
                if q in fq:
                    new_pts.append(fq[q])
            _quadrant_pts_list[q] = new_pts if new_pts else []
            _quadrant_pts_count[q] = sum(p.shape[0] for p in new_pts)
            _quadrant_dirty[q] = True

    def _sync_overlap_tree_with_buffer():
        to_remove = _overlap_frame_set - _buffer_indices
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

    _pose_history = {}
    _chain_history = {}

    def _run_window_lm(window_indices):
        """Sliding window LM: optimize only the poses in window_indices."""
        if not _HAS_GTSAM or len(window_indices) < 2:
            return
        window = sorted(window_indices)
        window_set = set(window)
        edges = [e for e in _constraint_buffer if e[0] in window_set and e[1] in window_set]
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
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
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
        """Chain accumulation with iSAM2 PGO refinement."""
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

        c2w = None
        chain_c2w = None
        if rel_poses is not None and ref_indices is not None:
            K = rel_poses.shape[1]
            for k, ref_idx in enumerate(ref_indices):
                if k >= K:
                    break
                T_rel = rel_poses[0, k].cpu().float()

                is_loop_edge = ref_idx in _active_loop_refs
                if is_loop_edge:
                    _has_loop_edges[0] = True
                    real_gap = abs(frame_idx - ref_idx)
                    ss = loop_sigma_scale
                    forced_gap = 1
                    trans = T_rel[:3, 3].norm().item()
                    R = T_rel[:3, :3]
                    cos_a = ((R.trace() - 1) / 2).clamp(-1, 1)
                    rot_deg = cos_a.acos().item() * 180 / 3.14159265
                    print(f"  Loop edge T_rel: frame {frame_idx} <-> {ref_idx}, "
                          f"trans={trans:.2f}m, rot={rot_deg:.1f}deg, ss={ss:.4f}")
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

        if c2w is None:
            if (frame_idx - 1) in _pose_history:
                c2w = _pose_history[frame_idx - 1].clone()
                if _isam2_pgo is not None:
                    identity = torch.eye(4, dtype=torch.float32)
                    _isam2_pgo.add_constraint(
                        frame_idx, frame_idx - 1, identity,
                        sigma_scale=0.001, forced_gap=1, is_loop=True)
            else:
                c2w = torch.eye(4, dtype=torch.float32)

        if chain_c2w is None:
            if (frame_idx - 1) in _chain_history:
                chain_c2w = _chain_history[frame_idx - 1].clone()
            else:
                chain_c2w = c2w.clone()

        _chain_history[frame_idx] = chain_c2w.clone()
        _pose_history[frame_idx] = c2w

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
        n = len(dists)
        k = int(_np.ceil(n * percentile / 100.0)) - 1
        k = max(0, min(k, n - 1))
        return float(_np.partition(dists, k)[k])

    def _extract_world_pts(result, c2w, subsamp):
        pts3d_local = result.get('pts3d_in_self_view')
        conf = result.get('conf_self')
        if pts3d_local is None or conf is None:
            return None, None
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

        _sync_overlap_tree_with_buffer()

        overlap_score = _compute_overlap_score(pts_world, depths, cam_center)
        conf = result.get('conf_self')
        median_conf = conf[0].median().item() if conf is not None else 0.0
        is_kf = (overlap_score > overlap_thr) and (median_conf > min_conf_kf)

        if is_kf:
            keyframe_indices.add(frame_idx)
            _add_to_overlap_tree(pts_world, cam_center, frame_idx=frame_idx)

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

        if _has_loop_edges[0]:
            _has_loop_edges[0] = False

    def _finalize_predictions(predictions):
        """Write final poses to all predictions."""
        for i in range(len(predictions)):
            if i in _pose_history:
                predictions[i]['online_pgo_c2w'] = _pose_history[i].clone()

        if not kf_pgo or len(_pose_list) == 0:
            return
        _do_finalize = params.get('do_finalize', True)
        if _do_finalize and _isam2_pgo is not None and not _isam2_pgo.is_broken:
            _isam2_pgo.finalize(extra_iterations=50)
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
# Pose accumulation
# =====================================================================

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
    has_relative_poses = 'relative_poses' in predictions[0]

    if use_relative_pose is None:
        use_relative_pose = has_relative_poses and (has_online_pgo or has_final_pgo or has_chain)
    elif use_relative_pose and not (has_online_pgo or has_final_pgo or has_chain):
        print("Warning: use_relative_pose=True but no c2w found. Falling back.")
        use_relative_pose = False

    if use_relative_pose:
        B = predictions[0]["pts3d_in_self_view"].shape[0]
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
