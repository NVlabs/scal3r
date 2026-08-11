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

import logging
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from torch.utils.checkpoint import checkpoint

from stream3r.models.components.layers import PatchEmbed
from stream3r.models.components.layers.block import Block
from stream3r.models.components.layers.mlp import Mlp
from stream3r.models.components.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from stream3r.models.components.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class STreamAggregator(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_rel_pose_prompt: bool = False,
        num_rel_pose_tokens: int = 4,  # number of learnable base tokens (fixed at training)
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        self.use_rel_pose_prompt = use_rel_pose_prompt
        if use_rel_pose_prompt:
            self.rel_pose_token = nn.Parameter(torch.randn(1, num_rel_pose_tokens, embed_dim) * 0.02)
            self.prev_pose_proj = Mlp(embed_dim * 2, embed_dim * 4, embed_dim, act_layer=nn.GELU, drop=0)
        else:
            self.rel_pose_token = None

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def _assemble_rel_pose_tokens_streaming(self, B, device, dtype, pose_token_buffer):
        """Streaming assembly: build rel_pose tokens from the reference buffer.

        Uses the FULL buffer it is given (reversed so k=0 = most recent). The
        reference-count cap is an inference/training buffer-management concern
        applied by the caller (StreamSession / per-frame training loop / PGO
        callback), not by the model — mirroring CUT3R's design.

        Args:
            pose_token_buffer: List[(frame_idx, [B, C])] — ref feat history (already capped)

        Returns:
            assembled: [B, n_actual, C] or None if no refs available
            ref_indices: List[int] — frame indices of selected refs
            valid_mask: [B, 1, n_actual] — all True (no padding)
            n_actual: int — number of actual valid refs (= token count)
        """
        C = self.rel_pose_token.shape[-1]
        n_base = self.rel_pose_token.shape[1]

        selected = pose_token_buffer[::-1]
        n_actual = len(selected)

        if n_actual == 0:
            return None, [], None, 0

        ref_indices = [frame_idx for frame_idx, _ in selected]
        ref_feats = [feat for _, feat in selected]

        if n_actual <= n_base:
            base = self.rel_pose_token[:, :n_actual, :].expand(B, -1, -1).clone()
        else:
            base_part = self.rel_pose_token.expand(B, -1, -1).clone()
            extra = self.rel_pose_token[:, -1:, :].expand(B, n_actual - n_base, -1).clone()
            base = torch.cat([base_part, extra], dim=1)

        proj_dtype = self.prev_pose_proj.fc1.weight.dtype
        projected = torch.stack([self.prev_pose_proj(f.to(proj_dtype)) for f in ref_feats], dim=1)  # [B, n_actual, C]

        assembled = base + projected  # [B, n_actual, C]

        valid_mask = torch.ones(B, 1, n_actual, dtype=torch.bool, device=device)

        return assembled, ref_indices, valid_mask, n_actual

    def _create_attn_mask(self, S: int, P: int, mode: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        N = S * P
        mask = torch.zeros((N, N), dtype=dtype, device=device)
        
        if mode == "causal":
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                mask[curr_view_start:curr_view_end, curr_view_end:] = float('-inf')        
        elif mode == "window":
            window_size = 5
            for i in range(S):
                curr_view_start = i * P
                curr_view_end = (i + 1) * P
                mask[curr_view_start:curr_view_end, P:] = float('-inf')
                start_view = max(1, i - window_size + 1)
                mask[curr_view_start:curr_view_end, start_view*P:(i+1)*P] = 0
        elif mode == "full":
            mask = None
        else:
            raise NotImplementedError(f"Unknown attention mode: {mode}")

        return mask

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "causal",
        kv_cache_list: List[List[torch.Tensor]] = None,
        pose_token_buffer: Optional[list] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            mode (str): Global attention mode, could be either "causal", "window" or "full"
            kv_cache_list (List[List[torch.Tensor]]): List of cached key-value pairs for
                each global attention layer of the aggregator

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)

        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        rel_pose_info = {}

        # Expand camera and register tokens to match batch size and sequence length
        is_anchor_exist = kv_cache_list is None or kv_cache_list[0][0] is None
        camera_token = slice_expand_and_flatten(self.camera_token, B, S, is_anchor_exist=is_anchor_exist)
        register_token = slice_expand_and_flatten(self.register_token, B, S, is_anchor_exist=is_anchor_exist)

        n_qs = 0  # number of rel_pose query-suffix tokens (dynamic)
        if self.use_rel_pose_prompt:
            assert kv_cache_list is not None and pose_token_buffer is not None, \
                "use_rel_pose_prompt requires streaming (kv_cache + pose_token_buffer)"
            assembled, ref_indices, valid_mask, n_actual = self._assemble_rel_pose_tokens_streaming(
                B, patch_tokens.device, patch_tokens.dtype, pose_token_buffer
            )
            if assembled is not None:
                rel_pose_tokens = assembled
                n_qs = n_actual
            else:
                rel_pose_tokens = None
                ref_indices = []
                valid_mask = torch.zeros(B, 1, 0, dtype=torch.bool, device=patch_tokens.device)
            rel_pose_info['ref_indices'] = ref_indices
            rel_pose_info['valid_mask'] = valid_mask

            if rel_pose_tokens is not None:
                tokens = torch.cat([camera_token, register_token, patch_tokens, rel_pose_tokens], dim=1)
            else:
                tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        else:
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Add zero positions for rel_pose_tokens (no RoPE) — dynamic count
        if n_qs > 0:
            pos_rel_pose = torch.zeros(B * S, n_qs, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos, pos_rel_pose], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        attn_mask = None
        block_causal_S = 0
        if kv_cache_list is None:
            if mode == "causal":
                # Training causal: per-frame Flash Attention (no mask needed)
                block_causal_S = S
            else:
                # Training window/full: use explicit mask
                attn_mask = self._create_attn_mask(S, P, mode, tokens.dtype, tokens.device)

        # Save original assembled rel_pose tokens for residual connection (CUT3R-style)
        rel_pose_residual = None
        if self.use_rel_pose_prompt and n_qs > 0:
            # tokens shape: [B*S, P, C] — last n_qs positions are rel_pose
            rel_pose_residual = tokens[:, -n_qs:, :].clone()  # [B*S, K, C]
            rel_pose_residual = rel_pose_residual.view(B, S, n_qs, C)

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, n_query_suffix=n_qs
                    )
                elif attn_type == "global":
                    if kv_cache_list is not None:
                        kv_cache = kv_cache_list[global_idx]
                        # Inference: n_query_suffix for K/V stripping before cache
                        tokens, global_idx, global_intermediates, kv_cache = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask,
                            kv_cache=kv_cache, n_query_suffix=n_qs
                        )
                        kv_cache_list[global_idx-1] = kv_cache
                    else:
                        # Training: per-frame Flash Attention for causal, mask for window/full
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, attn_mask=attn_mask,
                            n_query_suffix=n_qs if block_causal_S > 0 else 0,
                            block_causal_S=block_causal_S,
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        if rel_pose_residual is not None and len(output_list) > 0:
            last = output_list[-1].clone()  # [B, S, P, 2C]
            last[:, :, -n_qs:, :C] = last[:, :, -n_qs:, :C] + rel_pose_residual
            last[:, :, -n_qs:, C:] = last[:, :, -n_qs:, C:] + rel_pose_residual
            output_list[-1] = last

        del concat_inter
        del frame_intermediates
        del global_intermediates

        if kv_cache_list is not None:
            return output_list, self.patch_start_idx, n_qs, kv_cache_list, rel_pose_info
        else:
            return output_list, self.patch_start_idx, n_qs, rel_pose_info

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, n_query_suffix=0):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = checkpoint(
                self.frame_blocks[frame_idx],
                tokens,
                pos,
                None,
                None,
                n_query_suffix,
                use_reentrant=False,
                
            )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, attn_mask=None,
                                   kv_cache=None, n_query_suffix=0, block_causal_S=0):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if kv_cache is not None:
                tokens, kv_cache = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    attn_mask,
                    kv_cache,
                    n_query_suffix,
                    block_causal_S,
                    use_reentrant=False
                )
            else:
                tokens = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    attn_mask,
                    None,
                    n_query_suffix,
                    block_causal_S,
                    use_reentrant=False
                )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        if kv_cache is not None:
            return tokens, global_idx, intermediates, kv_cache

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S, is_anchor_exist=False):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    if is_anchor_exist:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    else:
        query = token_tensor[:, 1:, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined