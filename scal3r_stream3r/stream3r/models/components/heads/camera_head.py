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

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stream3r.models.components.layers import Mlp
from stream3r.models.components.layers.block import Block
from stream3r.models.components.heads.head_act import activate_pose


class RelativePoseDecoder(nn.Module):
    """CUT3R-style per-token MLP decoder -> SE(3).

    Each rel_pose token independently predicts a relative pose (3D translation + 3x3 rotation)
    via a simple MLP(dim_in -> dim_in*4 -> 9). No attention pooling or FiLM modulation.
    """

    def __init__(self, dim_in: int = 2048, mlp_ratio: int = 4):
        super().__init__()
        self.mlp = Mlp(dim_in, int(dim_in * mlp_ratio), 9, drop=0)

    def orthogonalize_rotation(self, R: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt: (B, 2, 3) -> (B, 3, 3)."""
        x = F.normalize(R[:, 0], dim=-1)
        z = F.normalize(torch.cross(x, R[:, 1], dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack([x, y, z], dim=1)

    def forward(self, rel_pose_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rel_pose_tokens: [B, S, K, C] - K prompt tokens per frame

        Returns:
            rel_trans: [B, S, K, 3] - relative translation per token
            rel_rot: [B, S, K, 3, 3] - relative rotation per token
        """
        B, S, K, C = rel_pose_tokens.shape
        pred = self.mlp(rel_pose_tokens.reshape(B * S * K, C).float()).reshape(B, S, K, 9)
        rel_trans = pred[..., :3]
        rot_6d = pred[..., 3:9].reshape(B * S * K, 2, 3)
        rel_rot = self.orthogonalize_rotation(rot_6d).reshape(B, S, K, 3, 3)
        return rel_trans, rel_rot


class CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",  # Field of view activations: ensures FOV values are positive.
        # Relative pose prompt parameters
        use_rel_pose_prompt: bool = False,
        num_rel_pose_tokens: int = 8,
        rel_pose_global_only: bool = False,
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        # CUT3R-style per-token MLP decoder
        self.use_rel_pose_prompt = use_rel_pose_prompt
        self.rel_pose_global_only = rel_pose_global_only
        if use_rel_pose_prompt:
            decoder_dim = dim_in // 2 if rel_pose_global_only else dim_in
            self.rel_pose_decoder = RelativePoseDecoder(dim_in=decoder_dim)
        else:
            self.rel_pose_decoder = None

        self._no_grad_trunk = False

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def _create_attn_mask(self, S: int, mode: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        N = S
        mask = torch.zeros((N, N), dtype=dtype, device=device)
        
        if mode == "causal":
            for i in range(S):
                curr_view_start = i
                curr_view_end = (i + 1)
                mask[curr_view_start:curr_view_end, curr_view_end:] = float('-inf')        
        elif mode == "window":
            window_size = 5
            for i in range(S):
                curr_view_start = i
                curr_view_end = (i + 1)
                mask[curr_view_start:curr_view_end, 1:] = float('-inf')
                start_view = max(1, i - window_size + 1)
                mask[curr_view_start:curr_view_end, start_view:(i+1)] = 0
        elif mode == "full":
            mask = None
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

        return mask

    def forward(
        self,
        aggregated_tokens_list: list,
        num_iterations: int = 4,
        mode: str = "causal",
        kv_cache_list: List[List[List[torch.Tensor]]] = None,
        num_rel_pose_tokens: int = 0,
    ):
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.
            mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[List[torch.Tensor]]]): List of cached key-value pairs for
                each iterations and each attention layer of the camera head
            num_rel_pose_tokens (int): Number of rel_pose tokens at the end of token sequence

        Returns:
            If use_rel_pose_prompt:
                tuple: (pred_pose_enc_list, rel_pose_dict, [kv_cache_list])
            Else:
                list or tuple: predicted camera encodings, optionally with kv_cache_list.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]  # [B, S, P, C]

        # Extract the camera tokens (always at position 0)
        pose_tokens = tokens[:, :, 0]  # [B, S, C]

        # Extract rel_pose_tokens if enabled (at the end of token sequence)
        rel_pose_tokens = None
        if self.use_rel_pose_prompt and num_rel_pose_tokens > 0:
            if self.rel_pose_global_only:
                # Only use global-path features (last half of concat dim)
                C_half = tokens.shape[-1] // 2
                rel_pose_tokens = tokens[:, :, -num_rel_pose_tokens:, C_half:]  # [B, S, K, C/2]
            else:
                rel_pose_tokens = tokens[:, :, -num_rel_pose_tokens:]  # [B, S, K, C]

        pose_tokens = self.token_norm(pose_tokens)

        B, S, C = pose_tokens.shape
        attn_mask = None
        if kv_cache_list is None:
            attn_mask = self._create_attn_mask(S, mode, pose_tokens.dtype, pose_tokens.device)

        if self.training and self._no_grad_trunk:
            with torch.no_grad():
                pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations, attn_mask, kv_cache_list)
        else:
            pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations, attn_mask, kv_cache_list)

        # CUT3R-style per-token MLP decode (no prev_pose_token needed)
        rel_pose_dict = None
        if self.rel_pose_decoder is not None and rel_pose_tokens is not None:
            rel_trans, rel_rot = self.rel_pose_decoder(rel_pose_tokens)
            rel_pose_dict = {
                'rel_trans': rel_trans,   # [B, S, K, 3]
                'rel_rot': rel_rot,       # [B, S, K, 3, 3]
            }

        # Handle return based on kv_cache and rel_pose
        if kv_cache_list is not None:
            pred_pose_enc_list, kv_cache_list = pred_pose_enc_list
            if rel_pose_dict is not None:
                return pred_pose_enc_list, rel_pose_dict, kv_cache_list
            return pred_pose_enc_list, kv_cache_list

        if rel_pose_dict is not None:
            return pred_pose_enc_list, rel_pose_dict
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int, attn_mask: torch.Tensor, kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = None) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, S, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for iter in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for i in range(self.trunk_depth):
                if kv_cache_list is not None:
                    pose_tokens_modulated, kv_cache_list[iter][i] = self.trunk[i](pose_tokens_modulated, attn_mask=attn_mask, kv_cache=kv_cache_list[iter][i])
                else:
                    pose_tokens_modulated = self.trunk[i](pose_tokens_modulated, attn_mask=attn_mask)

            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                fl_act=self.fl_act,
            )
            pred_pose_enc_list.append(activated_pose)

        if kv_cache_list is not None:
            return pred_pose_enc_list, kv_cache_list
        else:
            return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
