# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None, n_query_suffix=0,
                block_causal_S=0) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Block-causal per-frame Flash Attention (training, causal mode)
        if block_causal_S > 0:
            x = self._block_causal_attn(q, k, v, block_causal_S, n_query_suffix)
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        # VQT-style asymmetric attention: exclude trailing query-only tokens from K/V
        # (matches CUT3R blocks.py:123-126)
        if n_query_suffix > 0:
            k_for_attn = k[:, :, :-n_query_suffix, :]
            v_for_attn = v[:, :, :-n_query_suffix, :]
        else:
            k_for_attn = k
            v_for_attn = v

        # KV cache concat (uses stripped K/V)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            if k_cache is not None and v_cache is not None:
                k_for_attn = torch.cat([k_cache, k_for_attn], dim=2)
                v_for_attn = torch.cat([v_cache, v_for_attn], dim=2)
            kv_cache = [k_for_attn, v_for_attn]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k_for_attn.transpose(-2, -1)

            if attn_mask is not None:
                attn = attn + attn_mask

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v_for_attn

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        x = self.proj_drop(x)

        if kv_cache is not None:
            return x, kv_cache

        return x

    def _block_causal_attn(self, q, k, v, S, n_query_suffix):
        """Block-causal attention via per-frame Flash Attention.

        Frame i attends to frames 0..i (causal across frames, full within frame).
        VQT: rel_pose tokens stripped from K/V per frame via n_query_suffix.
        Each per-frame call has no mask → dispatches to Flash Attention.
        """
        B, H, N, D = q.shape
        P = N // S
        P_kv = P - n_query_suffix

        q_frames = q.view(B, H, S, P, D)
        k_frames = k.view(B, H, S, P, D)
        v_frames = v.view(B, H, S, P, D)

        if n_query_suffix > 0:
            k_frames = k_frames[:, :, :, :P_kv, :].contiguous()
            v_frames = v_frames[:, :, :, :P_kv, :].contiguous()

        outputs = []
        for i in range(S):
            q_i = q_frames[:, :, i]                                    # [B, H, P, D]
            k_i = k_frames[:, :, :i + 1].reshape(B, H, (i + 1) * P_kv, D)
            v_i = v_frames[:, :, :i + 1].reshape(B, H, (i + 1) * P_kv, D)
            out_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            outputs.append(out_i)

        return torch.stack(outputs, dim=2).reshape(B, H, N, D)


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
