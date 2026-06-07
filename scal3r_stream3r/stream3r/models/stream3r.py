# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from stream3r.dust3r.utils.misc import freeze_all_params
from stream3r.models.components.aggregator.streamaggregator import STreamAggregator
from stream3r.models.components.heads.camera_head import CameraHead
from stream3r.models.components.heads.dpt_head import DPTHead


class STream3R(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        freeze="none",
        # CUT3R-style relative pose prompt parameters
        use_rel_pose_prompt: bool = False,
        num_rel_pose_tokens: int = 4,  # learnable base token count (fixed at training)
        max_ref_frames: int = 4,       # buffer window size (can override at inference)
        use_align_scale: bool = False,  # CUT3R-style pts3d scale alignment for rel_pose loss
    ):
        super().__init__()

        self.use_rel_pose_prompt = use_rel_pose_prompt
        self.patch_size = patch_size
        self.use_align_scale = use_align_scale
        # Buffer-window cap (reference-count limit). Lives here, not in the aggregator's
        # attention/assemble logic: it is applied by the buffer-management layer
        # (per-frame training loop + StreamSession at inference), mirroring CUT3R.
        self.max_ref_frames = max_ref_frames

        self.aggregator = STreamAggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_rel_pose_prompt=use_rel_pose_prompt,
            num_rel_pose_tokens=num_rel_pose_tokens,
        )
        self.camera_head = CameraHead(
            dim_in=2 * embed_dim,
            use_rel_pose_prompt=use_rel_pose_prompt,
        )
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        # Optimization flags for rel_pose_prompt training (set by set_freeze)
        self._skip_point_head = False
        self._no_grad_depth_head = False
        self._no_grad_point_head = False

        self.set_freeze(freeze)

    def set_freeze(self, freeze):
        self.freeze = freeze

        if freeze == "rel_pose_prompt":
            # Freeze all parameters first
            for param in self.parameters():
                param.requires_grad = False

            # Unfreeze rel_pose_token in aggregator
            if hasattr(self.aggregator, 'rel_pose_token') and self.aggregator.rel_pose_token is not None:
                self.aggregator.rel_pose_token.requires_grad = True

            # Unfreeze prev_pose_proj in aggregator
            if hasattr(self.aggregator, 'prev_pose_proj'):
                for p in self.aggregator.prev_pose_proj.parameters():
                    p.requires_grad = True

            # Unfreeze rel_pose_decoder in camera_head
            if hasattr(self.camera_head, 'rel_pose_decoder') and self.camera_head.rel_pose_decoder is not None:
                for param in self.camera_head.rel_pose_decoder.parameters():
                    param.requires_grad = True

            # Optimization A: skip/no_grad frozen heads during training
            if self.use_align_scale:
                # Need point_head output for pts3d scale alignment, run under no_grad
                self._skip_point_head = False
                self._no_grad_point_head = True
            else:
                self._skip_point_head = True
                self._no_grad_point_head = False
            self._no_grad_depth_head = True
            self.camera_head._no_grad_trunk = True

            # Optimization D: disable gradient checkpointing (avoid recomputation)
            self.aggregator.use_checkpoint = False
        else:
            to_be_frozen = {
                "none": [],
                "encoder": [self.aggregator.patch_embed],
            }
            freeze_all_params(to_be_frozen[freeze])

    def _forward_per_frame_training(self, images: torch.Tensor, mode: str = "causal"):
        """Per-frame loop training for relative pose prompt (camera-token references).

        Mirrors StreamSession.forward_stream() but accumulates all frame
        predictions into batch tensors for loss computation.  Camera tokens
        from previous frames (detached) are stored in pose_token_buffer and
        used as conditioning for subsequent frames' rel_pose assembly.
        """
        B, S, C_in, H, W = images.shape
        device = images.device
        K_max = self.max_ref_frames

        # Initialize KV caches (same structure as StreamSession._clear_cache)
        agg_kv = [[None, None] for _ in range(self.aggregator.depth)]
        cam_kv = [[[None, None] for _ in range(self.camera_head.trunk_depth)]
                  for _ in range(4)]  # 4 CameraHead iterations
        pose_token_buffer = []

        # Per-frame accumulators
        all_pose_enc = []
        all_depth = []
        all_depth_conf = []
        all_rel_trans = []
        all_rel_rot = []
        all_ref_indices = []
        all_valid_mask = []
        all_rel_pose_info = {}  # keep last frame's rel_pose_info for _rel_pose_info
        all_pts3d = [] if self._no_grad_point_head else None
        all_pts3d_conf = [] if self._no_grad_point_head else None

        for frame_idx in range(S):
            frame_img = images[:, frame_idx:frame_idx + 1]  # [B, 1, 3, H, W]

            # --- Aggregator forward with KV cache ---
            agg_out = self.aggregator(
                frame_img, mode=mode,
                kv_cache_list=agg_kv,
                pose_token_buffer=pose_token_buffer,
            )
            agg_tokens_list, patch_start_idx, n_qs, agg_kv, rel_pose_info = agg_out
            all_rel_pose_info = rel_pose_info

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                # --- CameraHead forward with KV cache ---
                if self.use_rel_pose_prompt and n_qs > 0:
                    pose_enc_list, rel_pose_dict, cam_kv = self.camera_head(
                        agg_tokens_list, mode=mode,
                        kv_cache_list=cam_kv,
                        num_rel_pose_tokens=n_qs,
                    )
                    # Pad rel_pose to K_max for uniform stacking
                    rel_trans = rel_pose_dict['rel_trans']  # [B, 1, K_cur, 3]
                    rel_rot = rel_pose_dict['rel_rot']      # [B, 1, K_cur, 3, 3]
                    K_cur = rel_trans.shape[2]
                    if K_cur < K_max:
                        pad_K = K_max - K_cur
                        rel_trans = torch.cat([
                            rel_trans,
                            torch.zeros(B, 1, pad_K, 3, device=device, dtype=rel_trans.dtype),
                        ], dim=2)
                        rel_rot_pad = torch.eye(3, device=device, dtype=rel_rot.dtype)
                        rel_rot_pad = rel_rot_pad.reshape(1, 1, 1, 3, 3).expand(B, 1, pad_K, 3, 3).clone()
                        rel_rot = torch.cat([rel_rot, rel_rot_pad], dim=2)
                    all_rel_trans.append(rel_trans)
                    all_rel_rot.append(rel_rot)
                else:
                    pose_enc_list, cam_kv = self.camera_head(
                        agg_tokens_list, mode=mode,
                        kv_cache_list=cam_kv,
                    )
                    # Frame with no refs: zero rel_pose with identity rotation
                    all_rel_trans.append(torch.zeros(B, 1, K_max, 3, device=device))
                    rot_eye = torch.eye(3, device=device).reshape(1, 1, 1, 3, 3).expand(B, 1, K_max, 3, 3).clone()
                    all_rel_rot.append(rot_eye)

                all_pose_enc.append(pose_enc_list[-1])  # [B, 1, 9]

                # --- Depth head under no_grad (for scale factors) ---
                with torch.no_grad():
                    depth, depth_conf = self.depth_head(
                        agg_tokens_list, images=frame_img,
                        patch_start_idx=patch_start_idx,
                        num_rel_pose_tokens=n_qs,
                    )
                all_depth.append(depth)
                all_depth_conf.append(depth_conf)

                # --- Point head under no_grad (for align_scale) ---
                if self._no_grad_point_head and self.point_head is not None:
                    with torch.no_grad():
                        pts3d, pts3d_conf = self.point_head(
                            agg_tokens_list, images=frame_img,
                            patch_start_idx=patch_start_idx,
                            num_rel_pose_tokens=n_qs,
                        )
                    all_pts3d.append(pts3d)
                    all_pts3d_conf.append(pts3d_conf)

            # --- Build ref_indices and valid_mask for this frame ---
            ref_indices_frame = rel_pose_info.get('ref_indices', [])
            ri_tensor = torch.zeros(1, K_max, dtype=torch.long, device=device)
            vm_tensor = torch.zeros(B, 1, K_max, dtype=torch.bool, device=device)
            if isinstance(ref_indices_frame, list):
                for k_idx, ref_frame in enumerate(ref_indices_frame):
                    if k_idx < K_max:
                        ri_tensor[0, k_idx] = ref_frame
                        vm_tensor[:, 0, k_idx] = True
            all_ref_indices.append(ri_tensor)
            all_valid_mask.append(vm_tensor)

            # --- Store camera token in buffer for next frame ---
            cam_tok = agg_tokens_list[-1][:, 0, 0, :]  # [B, 2C]
            pose_token_buffer.append((frame_idx, cam_tok.detach()))
            # Cap buffer to K_max most-recent refs (cap lives in the buffer-management
            # layer, not the aggregator assemble; behaviorally a no-op when S <= K_max+1)
            pose_token_buffer = pose_token_buffer[-K_max:]

        # --- Assemble batch predictions ---
        predictions = {}
        predictions["pose_enc"] = torch.cat(all_pose_enc, dim=1)          # [B, S, 9]
        predictions["pose_enc_list"] = [predictions["pose_enc"]]
        predictions["depth"] = torch.cat(all_depth, dim=1)
        predictions["depth_conf"] = torch.cat(all_depth_conf, dim=1)

        if all_pts3d is not None and len(all_pts3d) > 0:
            predictions["world_points"] = torch.cat(all_pts3d, dim=1)
            predictions["world_points_conf"] = torch.cat(all_pts3d_conf, dim=1)

        predictions["rel_pose"] = {
            'rel_trans': torch.cat(all_rel_trans, dim=1),    # [B, S, K_max, 3]
            'rel_rot': torch.cat(all_rel_rot, dim=1),        # [B, S, K_max, 3, 3]
            'ref_indices': torch.cat(all_ref_indices, dim=0), # [S, K_max]
            'valid_mask': torch.cat(all_valid_mask, dim=1),   # [B, S, K_max]
        }

        # Pass rel_pose_info for loss scale computation
        if all_rel_pose_info:
            predictions['_rel_pose_info'] = all_rel_pose_info

        return predictions

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "causal",
        aggregator_kv_cache_list: List[List[torch.Tensor]] = None,
        camera_head_kv_cache_list: List[List[List[torch.Tensor]]] = None,
        pose_token_buffer: Optional[list] = None,
    ):
        """
        Forward pass of the STream3R model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
            mode (str): Global attention mode, could be either "causal", "window", "full"
            aggregator_kv_cache_list: KV cache for aggregator global attention layers
            camera_head_kv_cache_list: KV cache for camera head attention layers
            pose_token_buffer: List[(frame_idx, Tensor)] for streaming rel_pose reference features

        Returns:
            dict: predictions including pose_enc, depth, world_points, rel_pose, etc.
        """
        if self.training:
            images = torch.stack([view["img"] for view in images], dim=1)
            images = (images + 1.) / 2.

        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # Per-frame loop for camera_token mode (both training and batch inference)
        # camera_token ref features come from aggregator output (2C dim), which requires
        # sequential per-frame processing to feed previous frames' camera tokens as ref features.
        # Streaming inference (aggregator_kv_cache_list != None) handles this via pose_token_buffer.
        if self.use_rel_pose_prompt and aggregator_kv_cache_list is None:
            return self._forward_per_frame_training(images, mode)

        # Aggregator forward
        if aggregator_kv_cache_list is not None:
            aggregated_tokens_list, patch_start_idx, num_rel_pose_tokens, aggregator_kv_cache_list, rel_pose_info = \
                self.aggregator(images, mode=mode, kv_cache_list=aggregator_kv_cache_list,
                                pose_token_buffer=pose_token_buffer)
        else:
            aggregated_tokens_list, patch_start_idx, num_rel_pose_tokens, rel_pose_info = \
                self.aggregator(images, mode=mode)

        predictions = {}

        # Pass camera token through rel_pose_info for streaming inference buffer
        if self.use_rel_pose_prompt:
            cam_tok = aggregated_tokens_list[-1][:, :, 0, :]  # [B, S, 2C]
            rel_pose_info['camera_token'] = cam_tok

        # Pass through rel_pose_info for loss computation
        if rel_pose_info:
            predictions['_rel_pose_info'] = rel_pose_info

        with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):
            if self.camera_head is not None:
                if camera_head_kv_cache_list is not None:
                    camera_head_output = self.camera_head(
                        aggregated_tokens_list,
                        mode=mode,
                        kv_cache_list=camera_head_kv_cache_list,
                        num_rel_pose_tokens=num_rel_pose_tokens,
                    )
                    # rel_pose_dict is only returned when num_rel_pose_tokens > 0
                    if self.use_rel_pose_prompt and num_rel_pose_tokens > 0:
                        pose_enc_list, rel_pose_dict, camera_head_kv_cache_list = camera_head_output
                        # Merge aggregator info into rel_pose_dict
                        rel_pose_dict['ref_indices'] = rel_pose_info.get('ref_indices')
                        rel_pose_dict['valid_mask'] = rel_pose_info.get('valid_mask')
                        predictions["rel_pose"] = rel_pose_dict
                    else:
                        pose_enc_list, camera_head_kv_cache_list = camera_head_output
                else:
                    camera_head_output = self.camera_head(
                        aggregated_tokens_list,
                        mode=mode,
                        num_rel_pose_tokens=num_rel_pose_tokens,
                    )
                    if self.use_rel_pose_prompt and num_rel_pose_tokens > 0:
                        pose_enc_list, rel_pose_dict = camera_head_output
                        rel_pose_dict['ref_indices'] = rel_pose_info.get('ref_indices')
                        rel_pose_dict['valid_mask'] = rel_pose_info.get('valid_mask')
                        predictions["rel_pose"] = rel_pose_dict
                    else:
                        pose_enc_list = camera_head_output

                predictions["pose_enc"] = pose_enc_list[-1]
                if self.training:
                    predictions["pose_enc_list"] = pose_enc_list

            if self.point_head is not None and not (self.training and self._skip_point_head):
                if self.training and self._no_grad_point_head:
                    with torch.no_grad():
                        pts3d, pts3d_conf = self.point_head(
                            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                            num_rel_pose_tokens=num_rel_pose_tokens
                        )
                else:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                        num_rel_pose_tokens=num_rel_pose_tokens
                    )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.depth_head is not None:
                if self.training and self._no_grad_depth_head:
                    with torch.no_grad():
                        depth, depth_conf = self.depth_head(
                            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                            num_rel_pose_tokens=num_rel_pose_tokens
                        )
                else:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                        num_rel_pose_tokens=num_rel_pose_tokens
                    )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

        if aggregator_kv_cache_list is not None:
            predictions["aggregator_kv_cache_list"] = aggregator_kv_cache_list

        if camera_head_kv_cache_list is not None:
            predictions["camera_head_kv_cache_list"] = camera_head_kv_cache_list

        if not self.training:
            predictions["images"] = images

        return predictions
