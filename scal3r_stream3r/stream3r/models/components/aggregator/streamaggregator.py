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
from typing import Tuple, List, Optional, Dict
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
        # CUT3R-style relative pose prompt parameters
        use_rel_pose_prompt: bool = False,
        num_rel_pose_tokens: int = 4,  # number of learnable base tokens (fixed at training)
        max_ref_frames: int = 4,       # buffer window size (can override at inference)
        ref_feat_type: str = "img_feat",  # "img_feat" or "camera_token"
        rel_pose_global_only: bool = False,  # only use global-path features for rel_pose decoder
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
        self.use_checkpoint = True

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # CUT3R-style relative pose prompt tokens
        # rel_pose_token: (1, n_base, C) — learnable base tokens (CUT3R: num_prompt_tokens)
        # Training: token slots = max_ref_frames (fixed K for uniform P across frames in batch)
        # Streaming inference: token slots = n_actual (dynamic, determined by buffer content)
        # When n_base < K, last base token is repeated (CUT3R model.py:934-938)
        self.use_rel_pose_prompt = use_rel_pose_prompt
        self.rel_pose_global_only = rel_pose_global_only
        self.ref_feat_type = ref_feat_type
        if use_rel_pose_prompt:
            self.rel_pose_token = nn.Parameter(torch.randn(1, num_rel_pose_tokens, embed_dim) * 0.02)
            self.max_ref_frames = max_ref_frames

            # prev_pose_proj: project reference feature to embed_dim
            # img_feat mode: input is mean-pooled patch features (embed_dim)
            # camera_token mode: input is aggregator camera token
            #   - global_only=False: full concat (embed_dim * 2)
            #   - global_only=True: global half only (embed_dim)
            if ref_feat_type == "camera_token" and not rel_pose_global_only:
                proj_in_dim = embed_dim * 2
            else:
                proj_in_dim = embed_dim
            self.prev_pose_proj = Mlp(proj_in_dim, embed_dim * 4, embed_dim, act_layer=nn.GELU, drop=0)
        else:
            self.rel_pose_token = None
            self.max_ref_frames = 0

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

    def _get_img_level_feat(self, feat):
        """Mean pool over spatial dimension. Matches CUT3R model.py:863."""
        return torch.mean(feat, dim=1, keepdim=True)  # [B*S, P_patch, C] -> [B*S, 1, C]

    def _assemble_rel_pose_tokens(self, B, S, device, dtype, global_img_feats):
        """Assemble multi-reference rel_pose tokens for batch training.

        Matches CUT3R model.py:881 signature style.
        K = max_ref_frames (fixed token slot count for uniform P across frames).
        Number of refs per frame = min(available, K).

        Args:
            B: batch size
            S: sequence length
            device, dtype: tensor device/dtype
            global_img_feats: [B, S, C] — mean-pooled patch features (detached)

        Returns:
            assembled: [B*S, K, C] — assembled tokens
            ref_indices: [S, K] — per-frame reference frame indices
            valid_mask: [B, S, K] — which refs are valid
        """
        K = self.max_ref_frames  # training: fixed K for uniform P across frames
        C = global_img_feats.shape[-1]
        n_base = self.rel_pose_token.shape[1]

        # Base tokens: handle n_base != K (CUT3R model.py:930-938)
        if K <= n_base:
            base = self.rel_pose_token[:, :K, :].unsqueeze(1).expand(B, S, K, C).clone()
        else:
            base_part = self.rel_pose_token.unsqueeze(1).expand(B, S, n_base, C)
            extra = self.rel_pose_token[:, -1:, :].unsqueeze(1).expand(B, S, K - n_base, C)
            base = torch.cat([base_part, extra], dim=2).clone()

        # Build ref_indices: frame i, ref k -> frame (i - k - 1)
        # Convention: k=0 = most recent reference (matches CUT3R)
        ref_indices = torch.zeros(S, K, dtype=torch.long, device=device)
        valid_mask = torch.zeros(B, S, K, dtype=torch.bool, device=device)
        ref_feats = torch.zeros(B, S, K, C, device=device, dtype=dtype)

        for k in range(K):
            idx = torch.arange(S, device=device) - k - 1
            valid = idx >= 0
            ref_indices[:, k] = idx.clamp(min=0)
            valid_mask[:, :, k] = valid.unsqueeze(0).expand(B, -1)
            # Gather reference features
            idx_clamped = idx.clamp(min=0)
            ref_feats[:, :, k] = global_img_feats[:, idx_clamped]

        ref_feats = ref_feats * valid_mask.unsqueeze(-1).to(ref_feats.dtype)

        # Project reference features (CUT3R: prev_pose_proj)
        # Cast to match prev_pose_proj weight dtype (may be bf16 under DeepSpeed)
        proj_dtype = self.prev_pose_proj.fc1.weight.dtype
        projected = self.prev_pose_proj(ref_feats.reshape(-1, C).to(proj_dtype)).reshape(B, S, K, C)
        projected = projected * valid_mask.unsqueeze(-1).to(projected.dtype)

        # CUT3R inject_mode == "once": rel_pose_feat = base + projected_refs
        assembled = base + projected

        return assembled.reshape(B * S, K, C), ref_indices, valid_mask

    def _assemble_rel_pose_tokens_streaming(self, B, device, dtype, pose_token_buffer,
                                              max_ref_frames_override=None):
        """Streaming inference: assemble from buffer with dynamic token count.

        Takes the most recent max_ref_frames entries from the buffer (after pruning),
        reversed so k=0 = most recent reference. Matches CUT3R model.py:942-950.

        Args:
            pose_token_buffer: List[(frame_idx, [B, C])] — ref feat history (already pruned)
            max_ref_frames_override: override buffer window size at inference time

        Returns:
            assembled: [B, n_actual, C] or None if no refs available
            ref_indices: List[int] — frame indices of selected refs
            valid_mask: [B, 1, n_actual] — all True (no padding)
            n_actual: int — number of actual valid refs (= token count)
        """
        max_refs = max_ref_frames_override if max_ref_frames_override is not None else self.max_ref_frames
        C = self.rel_pose_token.shape[-1]
        n_base = self.rel_pose_token.shape[1]

        # Take most recent max_refs entries, then reverse so k=0 = most recent
        selected = pose_token_buffer[-max_refs:][::-1]
        n_actual = len(selected)

        if n_actual == 0:
            return None, [], None, 0

        ref_indices = [frame_idx for frame_idx, _ in selected]
        ref_feats = [feat for _, feat in selected]

        # Base tokens: expand to n_actual, repeat last if n_actual > n_base
        # (matches CUT3R model.py:930-938)
        if n_actual <= n_base:
            base = self.rel_pose_token[:, :n_actual, :].expand(B, -1, -1).clone()
        else:
            base_part = self.rel_pose_token.expand(B, -1, -1).clone()
            extra = self.rel_pose_token[:, -1:, :].expand(B, n_actual - n_base, -1).clone()
            base = torch.cat([base_part, extra], dim=1)

        # Project + add (CUT3R inject_mode="once")
        proj_dtype = self.prev_pose_proj.fc1.weight.dtype
        projected = torch.stack([self.prev_pose_proj(f.to(proj_dtype)) for f in ref_feats], dim=1)  # [B, n_actual, C]

        assembled = base + projected  # [B, n_actual, C]

        valid_mask = torch.ones(B, 1, n_actual, dtype=torch.bool, device=device)

        return assembled, ref_indices, valid_mask, n_actual

    def _create_attn_mask(self, S: int, P: int, mode: str, dtype: torch.dtype, device: torch.device,
                          num_rel_pose_tokens: int = 0) -> torch.Tensor:
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

        # VQT: all queries cannot attend to rel_pose key positions in global blocks
        # rel_pose tokens are at the end of each frame's token sequence
        if num_rel_pose_tokens > 0 and mask is not None:
            for i in range(S):
                rel_start = i * P + P - num_rel_pose_tokens
                rel_end = i * P + P
                mask[:, rel_start:rel_end] = float('-inf')

        return mask

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "causal",
        kv_cache_list: List[List[torch.Tensor]] = None,
        pose_token_buffer: Optional[list] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)

        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch, C = patch_tokens.shape

        # CUT3R-style: mean-pool patch features for reference injection (detached!)
        rel_pose_info = {}
        if self.use_rel_pose_prompt:
            global_img_feat = self._get_img_level_feat(patch_tokens).squeeze(1).detach()  # [B*S, C]
            global_img_feat_bs = global_img_feat.view(B, S, C)  # [B, S, C]
            rel_pose_info['global_img_feat'] = global_img_feat_bs

        # Expand camera and register tokens to match batch size and sequence length
        is_anchor_exist = kv_cache_list is None or kv_cache_list[0][0] is None
        camera_token = slice_expand_and_flatten(self.camera_token, B, S, is_anchor_exist=is_anchor_exist)
        register_token = slice_expand_and_flatten(self.register_token, B, S, is_anchor_exist=is_anchor_exist)

        # CUT3R-style: assemble rel_pose_tokens with reference injection
        n_qs = 0  # number of rel_pose query-suffix tokens (dynamic)
        if self.use_rel_pose_prompt:
            if kv_cache_list is not None and pose_token_buffer is not None:
                # Streaming inference: dynamic token count
                assembled, ref_indices, valid_mask, n_actual = self._assemble_rel_pose_tokens_streaming(
                    B, patch_tokens.device, patch_tokens.dtype, pose_token_buffer
                )
                if assembled is not None:
                    # assembled is [B, n_actual, C] where S=1 for streaming
                    rel_pose_tokens = assembled
                    n_qs = n_actual
                else:
                    # No refs available (first frame) — skip rel_pose tokens entirely
                    rel_pose_tokens = None
                    ref_indices = []
                    valid_mask = torch.zeros(B, 1, 0, dtype=torch.bool, device=patch_tokens.device)
                rel_pose_info['ref_indices'] = ref_indices
                rel_pose_info['valid_mask'] = valid_mask
            else:
                # Batch training: fixed K = max_ref_frames for uniform P
                assembled, ref_indices, valid_mask = self._assemble_rel_pose_tokens(
                    B, S, patch_tokens.device, patch_tokens.dtype, global_img_feat_bs
                )
                rel_pose_tokens = assembled  # [B*S, K, C]
                n_qs = self.max_ref_frames
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
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Add zero positions for rel_pose_tokens (no RoPE) — dynamic count
        if n_qs > 0:
            pos_rel_pose = torch.zeros(B * S, n_qs, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos, pos_rel_pose], dim=1)

        _, P, C = tokens.shape

        attn_mask = None
        block_causal_S = 0
        if kv_cache_list is None:
            if mode == "causal":
                # Training causal: per-frame Flash Attention (no mask needed)
                block_causal_S = S
            else:
                # Training window/full: use explicit mask
                attn_mask = self._create_attn_mask(S, P, mode, tokens.dtype, tokens.device,
                                                    num_rel_pose_tokens=n_qs)

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
                    # Frame blocks: n_query_suffix for K/V truncation (VQT)
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
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        # Residual: add original assembled rel_pose tokens to the last output.
        # C here is the per-path dim (1024), output_list has concat dim (2C=2048).
        if rel_pose_residual is not None and len(output_list) > 0:
            last = output_list[-1].clone()  # [B, S, P, 2C]
            if self.rel_pose_global_only:
                # Only add residual to global half
                last[:, :, -n_qs:, C:] = last[:, :, -n_qs:, C:] + rel_pose_residual
            else:
                # Add residual to both frame half and global half
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
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if self.use_checkpoint:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    None,  # attn_mask
                    None,  # kv_cache
                    n_query_suffix,
                    use_reentrant=False
                )
            else:
                tokens = self.frame_blocks[frame_idx](
                    tokens, pos, None, None, n_query_suffix,
                )
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, attn_mask=None,
                                   kv_cache=None, n_query_suffix=0, block_causal_S=0):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if kv_cache is not None:
                if self.use_checkpoint:
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
                    tokens, kv_cache = self.global_blocks[global_idx](
                        tokens, pos, attn_mask, kv_cache, n_query_suffix, block_causal_S,
                    )
            else:
                if self.use_checkpoint:
                    tokens = checkpoint(
                        self.global_blocks[global_idx],
                        tokens,
                        pos,
                        attn_mask,
                        None,  # kv_cache
                        n_query_suffix,
                        block_causal_S,
                        use_reentrant=False
                    )
                else:
                    tokens = self.global_blocks[global_idx](
                        tokens, pos, attn_mask, None, n_query_suffix, block_causal_S,
                    )
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        if kv_cache is not None:
            return tokens, global_idx, intermediates, kv_cache

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S, is_anchor_exist=False):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing.

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """
    if is_anchor_exist:
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    else:
        query = token_tensor[:, 1:, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
