# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pose Graph Optimization utilities for STream3R.

Ported from CUT3R propose/src/dust3r/inference.py with matching function names.
"""

import numpy as np
import torch

try:
    import gtsam
    _HAS_GTSAM = True
    print(f"[PGO] gtsam imported OK, _HAS_GTSAM={_HAS_GTSAM}")
except ImportError as e:
    _HAS_GTSAM = False
    print(f"[PGO] gtsam import failed: {e}")

# Base sigmas for gap-dependent noise
_BASE_SIGMA_ROT = 0.5     # ~28.6 deg
_BASE_SIGMA_TRANS = 0.5   # 50 cm
_PGO_MODE = 'huber'
_ROT_GAP_POWER = 0.5
_TRANS_GAP_POWER = 0.5


def _se3_inverse(T):
    """Compute SE(3) inverse using R^T instead of generic matrix inverse.
    Preserves orthogonality. Input/output: (4,4) tensor."""
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
    GTSAM BetweenFactorPose3(ref, curr) expects inv(T_rel).
    Input: (4,4) or (1,4,4) torch tensor. Returns gtsam.Pose3."""
    if T_rel.dim() == 3:
        T_rel = T_rel.squeeze(0)
    T_rel_inv = _se3_inverse(T_rel.float()).cpu().double().numpy()
    R = gtsam.Rot3(T_rel_inv[:3, :3])
    t = gtsam.Point3(T_rel_inv[0, 3], T_rel_inv[1, 3], T_rel_inv[2, 3])
    return gtsam.Pose3(R, t)


def _make_robust_kernel(mode='huber'):
    """Create a GTSAM robust kernel by name."""
    kernels = {
        'dcs': lambda: gtsam.noiseModel.mEstimator.DCS.Create(1.0),
        'cauchy': lambda: gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
        'tukey': lambda: gtsam.noiseModel.mEstimator.Tukey.Create(4.685),
        'huber': lambda: gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    }
    return kernels.get(mode, kernels['huber'])()


def _make_gap_noise(frame_gap, sigma_scale=1.0, base_sigma_rot=_BASE_SIGMA_ROT, base_sigma_trans=_BASE_SIGMA_TRANS):
    """Build gap-dependent noise model with robust kernel for BetweenFactorPose3."""
    gap = max(abs(frame_gap), 1)
    rot_scale = (gap ** _ROT_GAP_POWER) * sigma_scale
    trans_scale = (gap ** _TRANS_GAP_POWER) * sigma_scale
    sr = base_sigma_rot * rot_scale
    st = base_sigma_trans * trans_scale
    sigmas = np.array([sr] * 3 + [st] * 3)
    noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    noise = gtsam.noiseModel.Robust.Create(_make_robust_kernel(_PGO_MODE), noise)
    return noise


def _make_loop_noise(frame_gap, sigma_scale=1.0, base_sigma_rot=_BASE_SIGMA_ROT, base_sigma_trans=_BASE_SIGMA_TRANS):
    """Build noise model for loop edges — NO robust kernel."""
    gap = max(abs(frame_gap), 1)
    rot_scale = (gap ** _ROT_GAP_POWER) * sigma_scale
    trans_scale = (gap ** _TRANS_GAP_POWER) * sigma_scale
    sr = base_sigma_rot * rot_scale
    st = base_sigma_trans * trans_scale
    sigmas = np.array([sr] * 3 + [st] * 3)
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)  # no robust kernel


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
                print(f"[PGO] iSAM2 numerical error — disabling PGO: {type(e).__name__}")
                self._broken = True
                return False
            raise

    def add_pose(self, frame_idx, c2w_tensor, is_anchor=False):
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

    def add_identity_constraint(self, frame_idx, ref_idx, sigma_scale=0.001):
        """Add tight identity constraint between two frames (no robust kernel).

        Used for reset/overlap frames that have no relative pose predictions,
        keeping the PGO graph strongly connected across reset boundaries.
        """
        if self._broken:
            return
        if ref_idx not in self._added_keys or frame_idx not in self._added_keys:
            return
        identity = gtsam.Pose3()  # identity transform
        sr = _BASE_SIGMA_ROT * sigma_scale
        st = _BASE_SIGMA_TRANS * sigma_scale
        sigmas = np.array([sr] * 3 + [st] * 3)
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)  # no robust kernel
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.BetweenFactorPose3(
            self._key(ref_idx), self._key(frame_idx), identity, noise))
        self._safe_update(graph)

    def add_global_pose_prior(self, frame_idx, c2w_tensor, sigma):
        """Add a PriorFactorPose3 from global pose (pose_enc) prediction."""
        if self._broken or frame_idx not in self._added_keys:
            return
        graph = gtsam.NonlinearFactorGraph()
        key = self._key(frame_idx)
        pose = _torch_c2w_to_gtsam_pose3(c2w_tensor)
        sigmas = np.array([sigma] * 3 + [sigma] * 3)
        noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        graph.addPriorPose3(key, pose, noise)
        self._safe_update(graph)

    def optimize(self, extra_iterations=2):
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

    def get_pose(self, frame_idx):
        if self._current_estimate is None:
            return None
        key = self._key(frame_idx)
        if not self._current_estimate.exists(key):
            return None
        return _gtsam_pose3_to_torch(self._current_estimate.atPose3(key), batch=True)

    def get_poses(self, frame_indices):
        if self._current_estimate is None:
            return {}
        result = {}
        for idx in frame_indices:
            key = self._key(idx)
            if self._current_estimate.exists(key):
                result[idx] = _gtsam_pose3_to_torch(
                    self._current_estimate.atPose3(key), batch=True)
        return result


def make_kf_only_callbacks(**params):
    """Sliding-window keyframe buffer with 3D overlap score and optional PGO.

    Ported from CUT3R propose/src/dust3r/inference.py with matching function names.

    Returns:
        (ref_frame_indices_fn, on_frame_processed, keyframe_indices, buffer_pruning_fn)
    """
    import numpy as _np

    num_init_frames = params.get('num_init_frames', 5)
    kf_window = params.get('kf_window', 4)
    nkf_buffer_size = params.get('nkf_buffer_size', 0)
    keyframe_indices = set()
    for i in range(num_init_frames):
        keyframe_indices.add(i)

    kf_pgo = params.get('kf_pgo', True)
    nkf_sigma_scale = params.get('nkf_sigma_scale', 1.0)
    pgo_position_scale = params.get('pgo_position_scale', 0)
    pgo_max_edges = params.get('pgo_max_edges', 0)

    # Local PGO sigma (do NOT mutate module-level globals)
    base_sigma_rot = params.get('pgo_sigma_rot') or _BASE_SIGMA_ROT
    base_sigma_trans = params.get('pgo_sigma_trans') or _BASE_SIGMA_TRANS

    # Local noise functions bound to local sigma values
    def _local_loop_noise(frame_gap, sigma_scale=1.0):
        return _make_loop_noise(frame_gap, sigma_scale, base_sigma_rot, base_sigma_trans)

    def _local_gap_noise(frame_gap, sigma_scale=1.0):
        return _make_gap_noise(frame_gap, sigma_scale, base_sigma_rot, base_sigma_trans)

    # ── Loop closure ──
    loop_detector = params.get('loop_detector', None)
    loop_image_paths = params.get('loop_image_paths', None)
    _kf_feature_archive = {}   # frame_idx → (frame_idx, feat) — survives buffer pruning
    _pending_loop_frames = []  # (loop_frame_idx, score) to inject into next ref selection
    _active_loop_refs = {}     # frame_idx → score, for geometric verification in _accumulate_c2w
    loop_max_translation = params.get('loop_max_translation', 20.0)
    loop_sigma_scale = params.get('loop_sigma_scale', 1.0)
    _injected_loop_entries = []  # temporarily injected entries to remove after model step
    _has_loop_edges = [False]

    _buffer_indices = set()
    # Try to create iSAM2 PGO; gracefully degrade if gtsam C++ backend fails
    _isam2_pgo = None
    if _HAS_GTSAM and kf_pgo:
        try:
            _isam2_pgo = _ISAM2PGO(noise_fns=(_local_loop_noise, _local_gap_noise))
            print("[PGO] iSAM2 PGO enabled")
        except Exception as e:
            print(f"[PGO] iSAM2 init failed ({type(e).__name__}: {e}), using chain only")
            _isam2_pgo = None
    _pose_list = []
    _pose_history = {}
    _chain_history = {}
    _constraint_buffer = []  # [(frame_idx, ref_idx, T_rel_4x4, sigma_scale, forced_gap, is_loop)]
    _global_pose_history = {}  # frame_idx → global pose c2w from pose_enc

    # Global pose init config
    use_global_pose_init = params.get('use_global_pose_init', False)
    global_pose_prior_sigma = params.get('global_pose_prior_sigma', None)

    # Buffer pruning
    def kf_only_buffer_pruning(pose_token_buffer, _kf_indices, **kwargs):
        # Archive keyframe features before pruning (for loop closure re-injection)
        for idx, feat in pose_token_buffer:
            if idx in keyframe_indices and idx not in _kf_feature_archive:
                _kf_feature_archive[idx] = (idx, feat)

        if not pose_token_buffer:
            return []
        latest_idx = pose_token_buffer[-1][0]
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
        if nkf_buffer_size > 0 and latest_idx not in seen:
            result.append(pose_token_buffer[-1])
        result.sort(key=lambda x: x[0])
        _buffer_indices.clear()
        _buffer_indices.update(idx for idx, _ in result)
        return result

    # Ref selection
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
        return refs

    # ── Overlap detection (optimized: cKDTree + dirty flags + lazy concat + sorted query) ──
    from scipy.spatial import cKDTree as _KDTree

    min_conf_kf = params.get('min_conf_keyframe', 1.2)
    overlap_thr = params.get('keyframe_overlap_thr', 0.1)
    percentile = params.get('overlap_percentile', 85)
    kf_subsamp = params.get('kf_x_subsamp', 4)
    depth_normalize = params.get('depth_normalize', True)
    quadrant_divider = params.get('quadrant_divider', 2)

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

    _query_timers = {'rebuild': [], 'quadrant_id': [], 'sort': [], 'tree_query': [],
                     'n_query_pts': [], 'n_tree_pts': []}

    def _query_overlap_tree(pts_world_np, cam_center_np):
        _tA = _time.perf_counter()
        _rebuild_dirty_trees()
        _tB = _time.perf_counter()
        rays = pts_world_np - cam_center_np[None]
        quad_ids = _get_quadrant_id(rays)
        _tC = _time.perf_counter()
        dists = _np.full(pts_world_np.shape[0], _np.inf)
        # Sort by quadrant once, then slice — avoids repeated boolean mask scans
        order = _np.argsort(quad_ids, kind='mergesort')
        sorted_ids = quad_ids[order]
        splits = _np.searchsorted(sorted_ids, _np.arange(_n_quadrants + 1))
        _tD = _time.perf_counter()
        total_tree_pts = 0
        for q in range(_n_quadrants):
            lo, hi = splits[q], splits[q + 1]
            if lo == hi:
                continue
            tree = _quadrant_trees[q]
            if tree is not None:
                d, _ = tree.query(pts_world_np[order[lo:hi]], k=1, workers=4)
                dists[order[lo:hi]] = d
                total_tree_pts += tree.n
        _tE = _time.perf_counter()
        _query_timers['rebuild'].append(_tB - _tA)
        _query_timers['quadrant_id'].append(_tC - _tB)
        _query_timers['sort'].append(_tD - _tC)
        _query_timers['tree_query'].append(_tE - _tD)
        _query_timers['n_query_pts'].append(len(pts_world_np))
        _query_timers['n_tree_pts'].append(total_tree_pts)
        return dists

    def _accumulate_c2w(frame_idx, result):
        # Extract global pose from pose_enc (if available)
        global_c2w = result.get('global_pose_c2w')
        if global_c2w is not None:
            global_c2w = global_c2w.cpu().float()
            if global_c2w.dim() == 3:
                global_c2w = global_c2w.squeeze(0)
            _global_pose_history[frame_idx] = global_c2w.clone()

        if frame_idx == 0:
            if use_global_pose_init and global_c2w is not None:
                c2w = global_c2w.clone()
            else:
                c2w = torch.eye(4, dtype=torch.float32)
            _chain_history[0] = c2w.clone()
            _pose_history[0] = c2w
            _pose_list.append(c2w.unsqueeze(0))
            if _isam2_pgo is not None:
                _isam2_pgo.add_pose(0, c2w, is_anchor=True)
            return c2w, c2w

        rel_poses = result.get('relative_poses')
        ref_indices = result.get('ref_frame_indices')

        c2w = None
        chain_c2w = None

        if rel_poses is not None and ref_indices is not None:
            # rel_poses: {'rel_trans': [B,S,K,3], 'rel_rot': [B,S,K,3,3]}
            rel_trans = rel_poses.get('rel_trans')
            rel_rot = rel_poses.get('rel_rot')
            if rel_trans is not None and rel_rot is not None:
                K = rel_trans.shape[2] if rel_trans.dim() == 4 else rel_trans.shape[1]
                ref_list = ref_indices if isinstance(ref_indices, list) else []

                for k in range(min(K, len(ref_list))):
                    ref_idx = ref_list[k]
                    # Build 4x4 relative pose
                    if rel_trans.dim() == 4:
                        t_k = rel_trans[0, -1, k].cpu().float()
                        R_k = rel_rot[0, -1, k].cpu().float()
                    else:
                        t_k = rel_trans[0, k].cpu().float()
                        R_k = rel_rot[0, k].cpu().float()

                    T_rel = torch.eye(4, dtype=torch.float32)
                    T_rel[:3, :3] = R_k
                    T_rel[:3, 3] = t_k

                    # Loop edges: constraint only, no chain accumulation
                    is_loop_edge = ref_idx in _active_loop_refs
                    if is_loop_edge:
                        _has_loop_edges[0] = True
                        real_gap = abs(frame_idx - ref_idx)
                        ss = loop_sigma_scale / max(real_gap / 100.0, 1.0) ** 0.5
                        forced_gap = 1
                        _constraint_buffer.append(
                            (frame_idx, ref_idx, T_rel.clone(), ss, forced_gap, True))
                        if _isam2_pgo is not None:
                            _isam2_pgo.add_constraint(
                                frame_idx, ref_idx, T_rel,
                                sigma_scale=ss, forced_gap=forced_gap, is_loop=True)
                            _isam2_pgo.optimize()
                        continue

                    # Normal sequential edges
                    is_j_kf = frame_idx in keyframe_indices
                    is_ref_kf = ref_idx in keyframe_indices
                    ss = 1.0 if (is_j_kf or is_ref_kf) else nkf_sigma_scale
                    if pgo_position_scale > 0:
                        ss *= (1.0 + frame_idx / pgo_position_scale)
                    pgo_edge_ok = (pgo_max_edges <= 0 or k < pgo_max_edges)

                    if pgo_edge_ok:
                        _constraint_buffer.append(
                            (frame_idx, ref_idx, T_rel.clone(), ss, None, False))
                        if _isam2_pgo is not None:
                            _isam2_pgo.add_constraint(frame_idx, ref_idx, T_rel, sigma_scale=ss)

                    T_rel_inv = _se3_inverse(T_rel)
                    if c2w is None and ref_idx in _pose_history:
                        c2w = _reorthogonalize_c2w(_pose_history[ref_idx] @ T_rel_inv)
                    if chain_c2w is None and ref_idx in _chain_history:
                        chain_c2w = _reorthogonalize_c2w(_chain_history[ref_idx] @ T_rel_inv)

        # Use global pose as PGO init (overrides chain-accumulated c2w)
        if use_global_pose_init and global_c2w is not None:
            c2w = global_c2w.clone()

        # No valid relative pose → fallback to previous frame's pose
        # Add tight identity constraint to keep PGO graph connected (CUT3R inference.py:744-750)
        _needs_identity_constraint = False
        if c2w is None:
            c2w = _pose_history.get(frame_idx - 1, torch.eye(4, dtype=torch.float32)).clone()
            if (frame_idx - 1) in _pose_history:
                _needs_identity_constraint = True
        if chain_c2w is None:
            chain_c2w = _chain_history.get(frame_idx - 1, c2w.clone()).clone()

        _chain_history[frame_idx] = chain_c2w.clone()
        _pose_history[frame_idx] = c2w
        _pose_list.append(c2w.unsqueeze(0))

        if _isam2_pgo is not None and not _isam2_pgo.is_broken:
            _isam2_pgo.add_pose(frame_idx, c2w)
            # Tight identity constraint for frames with no refs (e.g., overlap after reset)
            if _needs_identity_constraint:
                _isam2_pgo.add_identity_constraint(frame_idx, frame_idx - 1)
                # Also record in constraint_buffer so batch LM can use it
                identity = torch.eye(4, dtype=torch.float32)
                _constraint_buffer.append(
                    (frame_idx, frame_idx - 1, identity, 0.001, 1, True))  # is_loop=True → no robust kernel
            # Add global pose as unary prior if sigma is configured
            if global_pose_prior_sigma is not None and global_c2w is not None:
                _isam2_pgo.add_global_pose_prior(frame_idx, global_c2w, global_pose_prior_sigma)
            _isam2_pgo.optimize()
            # Write back only buffer + current frame poses (CUT3R inference.py:771-778)
            # Only updating active poses keeps old poses stable and avoids drift propagation
            if not _isam2_pgo.is_broken:
                active = _buffer_indices | {frame_idx}
                for idx in active:
                    opt = _isam2_pgo.get_pose(idx)
                    if opt is not None:
                        _pose_history[idx] = opt.squeeze(0)
                        if idx < len(_pose_list):
                            _pose_list[idx] = opt

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
        # pts3d_local: [B, 1, H, W, 4], conf: [B, 1, H, W] — squeeze B and S dims to [H, W, ...]
        pts_gpu = pts3d_local[0]
        while pts_gpu.dim() > 3:
            pts_gpu = pts_gpu.squeeze(0)
        c_gpu = conf[0]
        while c_gpu.dim() > 2:
            c_gpu = c_gpu.squeeze(0)
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

    _debug_counter = [0]
    _total_frames = [0]
    # Fine-grained timing for KF selection breakdown
    import time as _time
    _kf_timers = {'accumulate': [], 'extract_pts': [], 'sync_tree': [],
                  'overlap_query': [], 'add_tree': []}

    def on_frame_processed(frame_idx, result):
        _t0 = _time.perf_counter()
        chain_c2w, pgo_c2w = _accumulate_c2w(frame_idx, result)
        _t1 = _time.perf_counter()
        result['chain_c2w'] = chain_c2w
        result['online_pgo_c2w'] = pgo_c2w
        c2w = pgo_c2w

        _t2 = _time.perf_counter()
        pts_world, depths = _extract_world_pts(result, c2w, kf_subsamp)
        _t3 = _time.perf_counter()
        cam_center = c2w[:3, 3].numpy()
        _total_frames[0] = frame_idx + 1

        _kf_timers['accumulate'].append(_t1 - _t0)
        _kf_timers['extract_pts'].append(_t3 - _t2)

        if frame_idx < num_init_frames:
            is_kf = True  # init frames are always KF
            if pts_world is not None:
                _ta = _time.perf_counter()
                _add_to_overlap_tree(pts_world, cam_center, frame_idx=frame_idx)
                _kf_timers['add_tree'].append(_time.perf_counter() - _ta)
        elif pts_world is None:
            is_kf = False
        else:
            # Sync overlap tree with current buffer (remove old frames' points)
            _ts = _time.perf_counter()
            _sync_overlap_tree_with_buffer()
            _kf_timers['sync_tree'].append(_time.perf_counter() - _ts)

            _tq = _time.perf_counter()
            overlap_score = _compute_overlap_score(pts_world, depths, cam_center)
            _kf_timers['overlap_query'].append(_time.perf_counter() - _tq)
            conf = result.get('conf_self')
            median_conf = conf[0].median().item() if conf is not None else 0.0
            is_kf = (overlap_score > overlap_thr) and (median_conf > min_conf_kf)

            if is_kf:
                keyframe_indices.add(frame_idx)
                _ta = _time.perf_counter()
                _add_to_overlap_tree(pts_world, cam_center, frame_idx=frame_idx)
                _kf_timers['add_tree'].append(_time.perf_counter() - _ta)

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

        # Reset loop flag
        if _has_loop_edges[0]:
            _has_loop_edges[0] = False

    def _print_kf_stats():
        pass
    on_frame_processed.print_kf_stats = _print_kf_stats

    # Global LM PGO (matches CUT3R _run_global_pgo)
    def _run_global_pgo():
        """KF-only global PGO (Levenberg-Marquardt): optimize keyframe poses,
        recover non-KF poses by relative offset to nearest optimized KF.
        Also includes loop closure edges."""
        if not _HAS_GTSAM or len(_constraint_buffer) < 2:
            return None
        import bisect

        # Include ALL edges for full global optimization (not just KF↔KF)
        # This ensures loop correction propagates through the entire graph
        kf_graph_indices = set()
        processed_edges = []
        n_loop = 0
        for entry in _constraint_buffer:
            j, ref_idx, T_rel = entry[0], entry[1], entry[2]
            ss = entry[3]
            forced_gap = entry[4] if len(entry) > 4 else None
            is_loop = entry[5] if len(entry) > 5 else False

            gap = forced_gap if forced_gap is not None else abs(j - ref_idx)
            processed_edges.append((j, ref_idx, T_rel, gap, ss, is_loop))
            kf_graph_indices.add(j)
            kf_graph_indices.add(ref_idx)
            if is_loop:
                n_loop += 1

        kf_graph_indices = sorted(kf_graph_indices)
        if len(processed_edges) < 2 or len(kf_graph_indices) < 2:
            return None

        # Build factor graph
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        for idx in kf_graph_indices:
            key = gtsam.symbol('x', idx)
            # Prefer global pose as init over chain-accumulated
            if use_global_pose_init and idx in _global_pose_history:
                initial.insert(key, _torch_c2w_to_gtsam_pose3(_global_pose_history[idx]))
            elif idx in _pose_history:
                initial.insert(key, _torch_c2w_to_gtsam_pose3(_pose_history[idx]))
            else:
                initial.insert(key, gtsam.Pose3())

        # Anchor first KF
        anchor = kf_graph_indices[0]
        prior_sigmas = np.array([1e-6] * 6)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        graph.addPriorPose3(gtsam.symbol('x', anchor),
                            initial.atPose3(gtsam.symbol('x', anchor)),
                            prior_noise)

        # Add global pose priors for all KFs (must match online iSAM2)
        if use_global_pose_init and global_pose_prior_sigma is not None:
            gp_sigmas = np.array([global_pose_prior_sigma] * 6)
            gp_noise = gtsam.noiseModel.Diagonal.Sigmas(gp_sigmas)
            for idx in kf_graph_indices:
                if idx != anchor and idx in _global_pose_history:
                    key = gtsam.symbol('x', idx)
                    gp_pose = _torch_c2w_to_gtsam_pose3(_global_pose_history[idx])
                    graph.addPriorPose3(key, gp_pose, gp_noise)

        for edge in processed_edges:
            kf_j, kf_ref, T_rel, gap, ss = edge[:5]
            edge_is_loop = edge[5] if len(edge) > 5 else False
            measurement = _make_between_measurement(T_rel)
            noise = _local_loop_noise(gap, sigma_scale=ss) if edge_is_loop else _local_gap_noise(gap, sigma_scale=ss)
            graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol('x', kf_ref), gtsam.symbol('x', kf_j),
                measurement, noise))

        # Solve with Levenberg-Marquardt
        try:
            lm_params = gtsam.LevenbergMarquardtParams()
            lm_params.setMaxIterations(200)
            lm_params.setRelativeErrorTol(1e-8)
            lm_params.setAbsoluteErrorTol(1e-8)
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
            estimate = optimizer.optimize()
        except Exception as e:
            print(f"  Global LM failed: {type(e).__name__}: {e}")
            return None

        # Extract optimized KF poses
        kf_optimized = {}
        for idx in kf_graph_indices:
            key = gtsam.symbol('x', idx)
            if estimate.exists(key):
                kf_optimized[idx] = _gtsam_pose3_to_torch(estimate.atPose3(key), batch=False)

        # Recover ALL frames: NKF via relative offset to nearest optimized KF
        all_frame_indices = sorted(_pose_history.keys())
        opt_kf_sorted = sorted(kf_optimized.keys())
        result = {}
        for i in all_frame_indices:
            if i in kf_optimized:
                result[i] = kf_optimized[i].unsqueeze(0)
            else:
                pos = bisect.bisect_right(opt_kf_sorted, i) - 1
                nearest_kf = opt_kf_sorted[max(pos, 0)]
                chain_kf = _pose_history[nearest_kf]
                chain_nkf = _pose_history[i]
                T_rel = _se3_inverse(chain_kf) @ chain_nkf
                opt_nkf = kf_optimized[nearest_kf] @ T_rel
                result[i] = opt_nkf.unsqueeze(0)

        print(f"  Global LM PGO: {len(processed_edges)} edges ({n_loop} loop), "
              f"{len(kf_graph_indices)} graph nodes, {len(result)} total")
        return result

    # Attach finalize method
    def _finalize_predictions(predictions):
        n = min(len(predictions), len(_pose_list))

        # online_pgo_c2w: snapshot from online chain
        for i in range(n):
            if i in _pose_history:
                predictions[i]['online_pgo_c2w'] = _pose_history[i].clone()

        if not kf_pgo or len(_pose_list) == 0:
            for i in range(n):
                if i in _pose_history:
                    predictions[i]['kf_pgo_c2w'] = _pose_history[i].clone()
            return

        # If loop edges exist, use batch LM for full global correction
        # (iSAM2 incremental can't propagate loop corrections through long chains)
        has_any_loop = any(len(entry) > 5 and entry[5] for entry in _constraint_buffer)
        if has_any_loop:
            print(f"  [Finalize] Loop edges detected, running batch LM for global correction")
            optimized = _run_global_pgo()
            if optimized is not None:
                for i in range(n):
                    if i in optimized:
                        predictions[i]['kf_pgo_c2w'] = optimized[i].squeeze(0)
                    elif i in _pose_history:
                        predictions[i]['kf_pgo_c2w'] = _pose_history[i].clone()
                return

        # kf_pgo_c2w: use iSAM2 final poses directly (matches CUT3R finalize)
        if _isam2_pgo is not None and not _isam2_pgo.is_broken:
            all_indices = list(range(n))
            optimized = _isam2_pgo.get_poses(all_indices)
            for i in all_indices:
                if i in optimized:
                    predictions[i]['kf_pgo_c2w'] = optimized[i].squeeze(0)
                else:
                    predictions[i]['kf_pgo_c2w'] = _pose_list[i].squeeze(0)
        else:
            # Fallback: batch LM if iSAM2 broken
            optimized = _run_global_pgo()
            if optimized is not None:
                for i in range(n):
                    if i in optimized:
                        predictions[i]['kf_pgo_c2w'] = optimized[i].squeeze(0)
                    elif i in _pose_history:
                        predictions[i]['kf_pgo_c2w'] = _pose_history[i].clone()
            else:
                for i in range(n):
                    if i in _pose_history:
                        predictions[i]['kf_pgo_c2w'] = _pose_history[i].clone()

    on_frame_processed.finalize = _finalize_predictions

    return ref_frame_indices_fn, on_frame_processed, keyframe_indices, kf_only_buffer_pruning
