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
        use_rel_pose_prompt: bool = False,
        num_rel_pose_tokens: int = 4,  # learnable base token count (fixed at training)
        max_ref_frames: int = 4,       # buffer window size (can override at inference)
        use_align_scale: bool = False,  # CUT3R-style pts3d scale alignment for rel_pose loss
    ):
        super().__init__()

        self.use_rel_pose_prompt = use_rel_pose_prompt
        self.patch_size = patch_size
        self.use_align_scale = use_align_scale
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
        else:
            to_be_frozen = {
                "none": [],
                "encoder": [self.aggregator.patch_embed],
            }
            freeze_all_params(to_be_frozen[freeze])

    def _forward_per_frame_training(self, images: torch.Tensor, mode: str = "causal"):
        """Per-frame loop training for relative pose prompt (camera-token references).

        Mirrors StreamSession.forward_stream(): drives the streaming forward()
        path frame-by-frame (KV cache + pose_token_buffer) and accumulates each
        frame's predictions into batch tensors for loss computation.  Reusing
        forward() keeps head no_grad / rel_pose decoding identical to inference.
        """
        B, S = images.shape[:2]
        device = images.device
        K_max = self.max_ref_frames

        # KV caches + reference buffer (same structure as StreamSession._clear_cache)
        agg_kv = [[None, None] for _ in range(self.aggregator.depth)]
        cam_kv = [[[None, None] for _ in range(self.camera_head.trunk_depth)]
                  for _ in range(4)]  # 4 CameraHead iterations
        pose_token_buffer = []

        keys = ["pose_enc", "depth", "depth_conf", "rel_trans", "rel_rot",
                "ref_indices", "valid_mask"]
        if self.use_align_scale:
            keys += ["world_points", "world_points_conf"]
        acc = {k: [] for k in keys}

        for frame_idx in range(S):
            preds = self.forward(
                images[:, frame_idx:frame_idx + 1], mode,
                aggregator_kv_cache_list=agg_kv,
                camera_head_kv_cache_list=cam_kv,
                # Cap references to the K_max most recent (buffer-management layer)
                pose_token_buffer=pose_token_buffer[-K_max:],
            )
            agg_kv = preds["aggregator_kv_cache_list"]
            cam_kv = preds["camera_head_kv_cache_list"]

            for k in keys:
                if k in preds:
                    acc[k].append(preds[k])

            # Pad this frame's rel_pose to K_max (identity rot / zero trans for
            # missing refs, e.g. the first frame which has no "rel_pose" key).
            rel_trans = torch.zeros(B, 1, K_max, 3, device=device)
            rel_rot = torch.eye(3, device=device).reshape(1, 1, 1, 3, 3).expand(B, 1, K_max, 3, 3).clone()
            ri = torch.zeros(1, K_max, dtype=torch.long, device=device)
            vm = torch.zeros(B, 1, K_max, dtype=torch.bool, device=device)
            if "rel_pose" in preds:
                rp = preds["rel_pose"]
                K_cur = rp["rel_trans"].shape[2]
                rel_trans[:, :, :K_cur] = rp["rel_trans"]
                rel_rot[:, :, :K_cur] = rp["rel_rot"]
                for k_idx, ref_frame in enumerate(rp["ref_indices"][:K_max]):
                    ri[0, k_idx] = ref_frame
                    vm[:, 0, k_idx] = True
            acc["rel_trans"].append(rel_trans)
            acc["rel_rot"].append(rel_rot)
            acc["ref_indices"].append(ri)
            acc["valid_mask"].append(vm)

            # Store this frame's camera token as reference for subsequent frames
            cam_tok = preds["_rel_pose_info"]["camera_token"][:, -1:]  # [B, 1, 2C]
            pose_token_buffer.append((frame_idx, cam_tok.squeeze(1).detach()))

        # --- Assemble batch predictions (concat over the S sequence dim) ---
        predictions = {
            "pose_enc": torch.cat(acc["pose_enc"], dim=1),        # [B, S, 9]
            "depth": torch.cat(acc["depth"], dim=1),
            "depth_conf": torch.cat(acc["depth_conf"], dim=1),
            "rel_pose": {
                "rel_trans": torch.cat(acc["rel_trans"], dim=1),   # [B, S, K_max, 3]
                "rel_rot": torch.cat(acc["rel_rot"], dim=1),       # [B, S, K_max, 3, 3]
                "ref_indices": torch.cat(acc["ref_indices"], dim=0),  # [S, K_max]
                "valid_mask": torch.cat(acc["valid_mask"], dim=1),    # [B, S, K_max]
            },
        }
        predictions["pose_enc_list"] = [predictions["pose_enc"]]
        if self.use_align_scale:
            predictions["world_points"] = torch.cat(acc["world_points"], dim=1)
            predictions["world_points_conf"] = torch.cat(acc["world_points_conf"], dim=1)

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
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            mode (str): Global attention mode, could be either "causal", "window", "full"
            aggregator_kv_cache_list (List[List[torch.Tensor]]): List of cached key-value pairs for
                each global attention layer of the aggregator
            camera_head_kv_cache_list (List[List[List[torch.Tensor]]]): List of cached key-value pairs for 
                each iterations and each attention layer of the camera head

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """
        if self.training and not torch.is_tensor(images):
            images = torch.stack([view["img"] for view in images], dim=1)
            images = (images + 1.) / 2.

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

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

        if self.use_rel_pose_prompt:
            cam_tok = aggregated_tokens_list[-1][:, :, 0, :]  # [B, S, 2C]
            rel_pose_info['camera_token'] = cam_tok

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

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    num_rel_pose_tokens=num_rel_pose_tokens
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.depth_head is not None:
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