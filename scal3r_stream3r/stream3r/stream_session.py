# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.pose_enc import pose_encoding_to_extri_intri
from stream3r.dust3r.utils.geometry import inv


class StreamSession:
    """
    A causal streaming inference session with KV cache management for STream3R.
    Supports CUT3R-style pose_token_buffer for multi-reference relative pose.
    """
    def __init__(self, model: STream3R, mode: str, use_pgo: bool = False, pgo_config: dict = None):
        self.model = model
        self.mode = mode
        self.aggregator_kv_cache_depth = model.aggregator.depth
        self.camera_head_kv_cache_depth = model.camera_head.trunk_depth
        self.camera_head_iterations = 4

        if self.mode not in ["causal", "window"]:
            raise ValueError(f"Unsupported attention mode when using kv_cache: {self.mode}")

        # Keyframe-only cache: only keyframes update KV cache (non-keyframes still forward)
        pgo_config = pgo_config or {}
        self.kf_only_cache = pgo_config.get('kf_only_cache', False)
        self.cache_window_size = pgo_config.get('kf_window', 5)
        self._num_init_frames = pgo_config.get('num_init_frames', 1)
        self._use_global_pose_init = pgo_config.get('use_global_pose_init', False)
        self._img_size_hw = None  # set on first forward for pose_enc → c2w conversion

        # PGO integration (optional)
        self.use_pgo = use_pgo
        self.pgo_callbacks = None
        if use_pgo:
            try:
                from stream3r.utils.pgo import make_kf_only_callbacks
                pgo_config = pgo_config or {}
                (self.ref_frame_indices_fn,
                 self.on_frame_processed,
                 self.keyframe_indices,
                 self.buffer_pruning_fn) = make_kf_only_callbacks(**pgo_config)
                self.pgo_callbacks = True
            except ImportError:
                import warnings
                warnings.warn("PGO utilities not available, falling back to chain accumulation")
                self.use_pgo = False

        self.clear()

    def _clear_predictions(self):
        self.predictions = dict()

    def _update_predictions(self, predictions):
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in predictions:
                self.predictions[k] = torch.cat(
                    [self.predictions.get(k, torch.empty(0, device=predictions[k].device)), predictions[k]],
                    dim=1
                )

        # Handle rel_pose as list (K varies per frame with dynamic buffer size)
        if "rel_pose" in predictions:
            if "rel_pose" not in self.predictions:
                self.predictions["rel_pose"] = []
            self.predictions["rel_pose"].append(predictions["rel_pose"])

    def _clear_cache(self):
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = [[[None, None] for _ in range(self.camera_head_kv_cache_depth)] for _ in range(self.camera_head_iterations)]
        # CUT3R-style pose_token_buffer: List[(frame_idx, [B, C])]
        self.pose_token_buffer = []
        self.frame_count = 0
        # Per-frame PGO results: List[dict] indexed by frame_idx
        self.pgo_results = []

    def _get_pose_buffer_entry(self, frame_idx, global_img_feat):
        """CUT3R model.py:866 — img_feat mode."""
        return (frame_idx, global_img_feat.squeeze(1).detach())

    def _update_cache(self, aggregator_kv_cache_list, camera_head_kv_cache_list):
        if self.mode == "causal":
            self.aggregator_kv_cache_list = aggregator_kv_cache_list
            self.camera_head_kv_cache_list = camera_head_kv_cache_list
        elif self.mode == "window":
            window_size = self.cache_window_size
            for k in range(2):
                for i in range(self.aggregator_kv_cache_depth):
                    h, w = self.predictions["depth"].shape[2], self.predictions["depth"].shape[3]
                    # KV cache does not contain rel_pose tokens (VQT strips them), so P excludes them
                    P = h * w // self.model.aggregator.patch_size // self.model.aggregator.patch_size + self.model.aggregator.patch_start_idx
                    anchor_token = aggregator_kv_cache_list[i][k][:, :, :P]
                    window_tokens = aggregator_kv_cache_list[i][k][:, :, max(P, aggregator_kv_cache_list[i][k].size(2)-window_size*P):]
                    self.aggregator_kv_cache_list[i][k] = torch.cat(
                        [anchor_token, window_tokens], dim=2
                    )
                for i in range(self.camera_head_iterations):
                    for j in range(self.camera_head_kv_cache_depth):
                        anchor_token = camera_head_kv_cache_list[i][j][k][:, :, :1]
                        window_tokens = camera_head_kv_cache_list[i][j][k][:, :, max(1, camera_head_kv_cache_list[i][j][k].size(2)-window_size):]
                        self.camera_head_kv_cache_list[i][j][k] = torch.cat(
                            [anchor_token, window_tokens], dim=2
                        )
        else:
            raise ValueError(f"Unsupported attention mode when using kv_cache: {self.mode}")

    def _get_cache(self):
        return (self.aggregator_kv_cache_list, self.camera_head_kv_cache_list)

    def get_all_predictions(self):
        return self.predictions

    def get_last_prediction(self):
        last_predictions = dict()
        for k in ["pose_enc", "world_points", "world_points_conf", "depth", "depth_conf", "images"]:
            if k in self.predictions:
                last_predictions[k] = self.predictions[k][:, -1:]

        if "rel_pose" in self.predictions and self.predictions["rel_pose"]:
            last_predictions["rel_pose"] = self.predictions["rel_pose"][-1]
        return last_predictions

    def get_pgo_poses(self):
        """Get accumulated c2w poses from PGO callbacks.

        Priority (matches CUT3R accumulate_poses):
          kf_pgo_c2w > online_pgo_c2w > chain_c2w

        Returns:
            list of [B, 4, 4] c2w tensors, one per frame. None if PGO not enabled.
        """
        if not self.use_pgo or not self.pgo_results:
            print(f"[get_pgo_poses] No PGO results: use_pgo={self.use_pgo}, n_results={len(self.pgo_results)}")
            return None

        # Finalize PGO (runs final iSAM2 optimization, updates all poses)
        if hasattr(self.on_frame_processed, 'finalize'):
            self.on_frame_processed.finalize(self.pgo_results)

        poses = []
        for result in self.pgo_results:
            if 'kf_pgo_c2w' in result:
                c2w = result['kf_pgo_c2w']
            elif 'online_pgo_c2w' in result:
                c2w = result['online_pgo_c2w']
            elif 'chain_c2w' in result:
                c2w = result['chain_c2w']
            else:
                c2w = torch.eye(4)
            if c2w.dim() == 2:
                c2w = c2w.unsqueeze(0)
            poses.append(c2w)
        return poses

    def reset_streaming_state(self):
        """Reset KV caches and pose_token_buffer without clearing predictions or PGO results.

        Used by reset_interval to break long sequences into segments while keeping
        the global trajectory intact. frame_count continues incrementing.
        Both KV cache and pose_token_buffer are cleared (matching CUT3R), so the
        overlap frame has no references and gets an identity constraint in PGO
        to maintain graph connectivity.
        """
        self.aggregator_kv_cache_list = [[None, None] for _ in range(self.aggregator_kv_cache_depth)]
        self.camera_head_kv_cache_list = [[[None, None] for _ in range(self.camera_head_kv_cache_depth)] for _ in range(self.camera_head_iterations)]
        self.pose_token_buffer = []

    def clear(self):
        self._clear_predictions()
        self._clear_cache()

    def _snapshot_cache(self):
        """Deep copy KV cache tensors to prevent in-place mutation by model forward."""
        agg_snap = [[t.clone() if t is not None else None for t in pair]
                    for pair in self.aggregator_kv_cache_list]
        cam_snap = [[[t.clone() if t is not None else None for t in pair]
                     for pair in layer] for layer in self.camera_head_kv_cache_list]
        return agg_snap, cam_snap

    def _restore_cache(self, agg_snap, cam_snap):
        """Restore KV cache from snapshot (discard in-place mutations from model forward)."""
        self.aggregator_kv_cache_list = agg_snap
        self.camera_head_kv_cache_list = cam_snap

    def forward_stream(self, images):
        # For kf_only_cache: snapshot cache before forward, because the model mutates
        # kv_cache_list in-place (aggregator._process_global_attention writes back to the list).
        # If the current frame turns out to be a non-keyframe, we restore the snapshot.
        cache_snapshot = None
        if self.kf_only_cache:
            cache_snapshot = self._snapshot_cache()

        aggregator_kv_cache_list, camera_head_kv_cache_list = self._get_cache()

        # Loop closure: inject pending loop frames into buffer before model forward
        if self.use_pgo and hasattr(self, 'ref_frame_indices_fn') and self.ref_frame_indices_fn is not None:
            self.ref_frame_indices_fn(self.frame_count, self.pose_token_buffer)

        # Cap reference count to max_ref_frames here (buffer-management layer),
        # so the model's assemble stays cap-agnostic (mirrors CUT3R).
        capped_buffer = self.pose_token_buffer[-self.model.max_ref_frames:]
        outputs = self.model(
            images=images,
            mode=self.mode,
            aggregator_kv_cache_list=aggregator_kv_cache_list,
            camera_head_kv_cache_list=camera_head_kv_cache_list,
            pose_token_buffer=capped_buffer,
        )

        self._update_predictions(outputs)

        # Update pose_token_buffer (CUT3R: pose_token_buffer.append(_get_pose_buffer_entry(...)))
        rel_pose_info = outputs.get('_rel_pose_info', {})
        if 'camera_token' in rel_pose_info:
            # camera_token is [B, S, 2C], take last frame as the reference feature
            self.pose_token_buffer.append(
                self._get_pose_buffer_entry(self.frame_count, rel_pose_info['camera_token'][:, -1:])
            )

        # PGO: on_frame_processed callback (CUT3R inference.py:677)
        # Must run BEFORE cache update so keyframe_indices is populated for kf_only_cache
        if self.use_pgo and hasattr(self, 'on_frame_processed') and self.on_frame_processed is not None:
            rel_pose = outputs.get("rel_pose")
            ref_indices = rel_pose_info.get('ref_indices', [])
            # Convert ref_indices for streaming: list of frame indices
            if isinstance(ref_indices, list) and len(ref_indices) > 0:
                ref_list = ref_indices
            elif hasattr(ref_indices, 'shape'):
                # Training tensor format [S, K] — not used in streaming, but handle gracefully
                ref_list = []
            else:
                ref_list = []
            result = {
                'ref_frame_indices': ref_list,
            }
            if rel_pose is not None:
                result['relative_poses'] = rel_pose
            if 'world_points' in outputs:
                result['pts3d_in_self_view'] = outputs['world_points'][:, -1:]
            if 'world_points_conf' in outputs:
                result['conf_self'] = outputs['world_points_conf'][:, -1:]
            # Pass image tensor for loop closure (avoids disk I/O)
            if 'images' in outputs:
                result['_img_tensor'] = outputs['images'][:, -1:]
            # Compute global pose c2w from pose_enc for PGO init
            if self._use_global_pose_init and 'pose_enc' in self.predictions:
                if self._img_size_hw is None and 'images' in self.predictions:
                    self._img_size_hw = self.predictions['images'].shape[-2:]
                if self._img_size_hw is not None:
                    pose_enc_i = self.predictions['pose_enc'][:, -1:]  # [B, 1, 9]
                    extr_i, _ = pose_encoding_to_extri_intri(pose_enc_i, self._img_size_hw)
                    extr_44 = torch.cat([extr_i[0, 0], torch.tensor([[0, 0, 0, 1]], device=extr_i.device)], dim=0)
                    result['global_pose_c2w'] = inv(extr_44)
            self.on_frame_processed(self.frame_count, result)
            # Store per-frame PGO results (chain_c2w, online_pgo_c2w written by callback)
            self.pgo_results.append(result)

        # Update KV cache: conditionally skip non-keyframes when kf_only_cache is enabled
        if self.kf_only_cache:
            is_keyframe = (self.frame_count < self._num_init_frames) or \
                          (hasattr(self, 'keyframe_indices') and self.frame_count in self.keyframe_indices)
            if is_keyframe:
                self._update_cache(
                    outputs["aggregator_kv_cache_list"],
                    outputs["camera_head_kv_cache_list"],
                )
            else:
                # Restore pre-forward snapshot: discard non-keyframe K/V that was
                # written in-place by the model's _process_global_attention
                self._restore_cache(*cache_snapshot)
        else:
            self._update_cache(
                outputs["aggregator_kv_cache_list"],
                outputs["camera_head_kv_cache_list"],
            )

        # Buffer pruning (CUT3R: kf_only_buffer_pruning)
        if self.use_pgo and hasattr(self, 'buffer_pruning_fn') and self.buffer_pruning_fn is not None:
            self.pose_token_buffer = self.buffer_pruning_fn(
                self.pose_token_buffer, self.keyframe_indices)

        self.frame_count += 1

        return self.get_all_predictions()
