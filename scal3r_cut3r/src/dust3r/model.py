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
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    fix_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger
from einops import rearrange

printer = get_logger(__name__, log_level="INFO")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        num_prompt_tokens=8,
        use_mask_attention=True,
        model_update_type="cut3r",  # "cut3r" or "ttt3r"
        max_ref_frames=4,  # Maximum number of reference frames in sliding window (0=unlimited)
        prev_pose_proj_type="linear",  # "linear" | "mlp"
        prev_pose_input_type="decoder_output",  # "decoder_output" | "img_feat" | "12d_pose"
        inject_mode="once",  # "once" | "per_layer_pre_attn" | "per_layer_film"
        film_hidden_ratio=0.25,
        rel_pose_head_type="mlp",  # "mlp" | "transformer"
        rel_pose_trunk_depth=2,  # Transformer trunk depth (only for transformer head)
        rel_pose_num_iterations=4,  # Iterative refinement steps (only for transformer head)
        rel_pose_loss_gamma=0.8,  # Gamma decay for per-iteration loss weighting
        use_ref_condition=True,  # False = ablation: skip prev_pose_proj injection
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.num_prompt_tokens = num_prompt_tokens
        self.use_mask_attention = use_mask_attention
        self.model_update_type = model_update_type
        self.max_ref_frames = max_ref_frames
        self.prev_pose_proj_type = prev_pose_proj_type
        self.prev_pose_input_type = prev_pose_input_type
        self.inject_mode = inject_mode
        self.film_hidden_ratio = film_hidden_ratio
        self.rel_pose_head_type = rel_pose_head_type
        self.rel_pose_trunk_depth = rel_pose_trunk_depth
        self.rel_pose_num_iterations = rel_pose_num_iterations
        self.rel_pose_loss_gamma = rel_pose_loss_gamma
        self.use_ref_condition = use_ref_condition
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _, _, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _, _, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        # Restore config after super().__init__ overwrites it with CrocoConfig
        self.config = config
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.relative_pose_token = nn.Parameter(
            torch.randn(1, config.num_prompt_tokens, self.dec_embed_dim) * 0.02, requires_grad=True
        )
        self.prev_pose_input_type = getattr(config, 'prev_pose_input_type', 'decoder_output')
        if self.prev_pose_input_type == '12d_pose':
            # Similar to pow3r: Mlp(12 -> 4*dec_dim -> dec_dim)
            self.prev_pose_proj = Mlp(
                in_features=12,
                hidden_features=self.dec_embed_dim * 4,
                out_features=self.dec_embed_dim,
                act_layer=nn.GELU, drop=0.0)
        elif self.prev_pose_input_type == 'img_feat':
            # global_img_feat_i comes from encoder (enc_embed_dim), not decoder
            self.prev_pose_proj = Mlp(
                in_features=self.enc_embed_dim,
                hidden_features=self.dec_embed_dim * 4,
                out_features=self.dec_embed_dim,
                act_layer=nn.GELU, drop=0.0)
        elif self.prev_pose_input_type == 'img_feat_12d_pose':
            # Concat img_feat (enc_embed_dim) + 12D GT pose = enc_embed_dim + 12
            self.prev_pose_proj = Mlp(
                in_features=self.enc_embed_dim + 12,
                hidden_features=self.dec_embed_dim * 4,
                out_features=self.dec_embed_dim,
                act_layer=nn.GELU, drop=0.0)
        elif getattr(config, 'prev_pose_proj_type', 'linear') == 'mlp':
            self.prev_pose_proj = Mlp(
                in_features=self.dec_embed_dim,
                hidden_features=self.dec_embed_dim * 4,
                out_features=self.dec_embed_dim,
                act_layer=nn.GELU, drop=0.0)
        else:
            self.prev_pose_proj = nn.Linear(self.dec_embed_dim, self.dec_embed_dim)
        self.num_prompt_tokens = config.num_prompt_tokens
        self.use_mask_attention = config.use_mask_attention
        self.max_ref_frames = getattr(config, 'max_ref_frames', 4)
        self.use_ref_condition = getattr(config, 'use_ref_condition', True)
        self.inject_mode = getattr(config, 'inject_mode', 'once')

        # FiLM conditioning modules (trainable, NOT in frozen list)
        if self.inject_mode == 'per_layer_film':
            film_hidden_ratio = getattr(config, 'film_hidden_ratio', 0.25)
            film_hidden_dim = int(self.dec_embed_dim * film_hidden_ratio)
            self.film_gamma_layers = nn.ModuleList([
                Mlp(in_features=self.dec_embed_dim, hidden_features=film_hidden_dim,
                    out_features=self.dec_embed_dim, act_layer=nn.GELU, drop=0.0)
                for _ in range(self.dec_depth)
            ])
            self.film_beta_layers = nn.ModuleList([
                Mlp(in_features=self.dec_embed_dim, hidden_features=film_hidden_dim,
                    out_features=self.dec_embed_dim, act_layer=nn.GELU, drop=0.0)
                for _ in range(self.dec_depth)
            ])
            # Default Kaiming init — no zero-init so FiLM starts with non-trivial modulation

        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            model = load_model(pretrained_model_name_or_path, device="cpu")
            return model
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
            "encoder_and_decoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
                self.downstream_head.dpt_self,
                self.downstream_head.final_transform,
                self.downstream_head.dpt_cross,
                self.downstream_head.dpt_rgb,
                self.downstream_head.pose_head,
            ],
        }
        if freeze == "encoder_and_decoder_and_head":
            # fix_all_params(to_be_frozen["encoder_and_decoder_and_head"]) # will not be updated
            freeze_all_params(to_be_frozen["encoder_and_decoder_and_head"]) # requires_grad = False
        else:
            freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, f_rel_pose, pos_rel_pose, return_attn=False, inject_per_layer=None, n_inject_suffix=0):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img)

        # VQT-style asymmetric attention: instead of boolean masks,
        # we exclude rel_pose tokens from K/V so they are query-only.
        n_query_suffix = 0
        n_cross_kv_suffix = 0

        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            if f_rel_pose is not None and pos_rel_pose is not None:
                n_rel_pose = f_rel_pose.shape[1]

                # Concatenate tokens: [pose, img, rel_pose]
                f_img = torch.cat([f_pose, f_img, f_rel_pose], dim=1)
                pos_img = torch.cat([pos_pose, pos_img, pos_rel_pose], dim=1)

                if self.use_mask_attention:
                    # VQT-style asymmetric attention (replaces boolean masks)
                    n_query_suffix = n_rel_pose
                    n_cross_kv_suffix = n_rel_pose
            else:
                f_img = torch.cat([f_pose, f_img], dim=1)
                pos_img = torch.cat([pos_pose, pos_img], dim=1)

        final_output.append((f_state, f_img))
        attention_maps = []  # Store attention maps for TTT3R

        # Precompute per-layer FiLM signals
        film_signals = None
        if self.inject_mode == 'per_layer_film' and inject_per_layer is not None:
            film_signals = []
            for i in range(len(self.dec_blocks)):
                gamma = self.film_gamma_layers[i](inject_per_layer)
                beta = self.film_beta_layers[i](inject_per_layer)
                film_signals.append((gamma, beta))

        for layer_idx, (blk_state, blk_img) in enumerate(zip(self.dec_blocks_state, self.dec_blocks)):
            layer_inject = film_signals[layer_idx] if film_signals is not None else inject_per_layer
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # Pass parameters as positional arguments for gradient checkpointing compatibility
                f_state, _, self_attn_state, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn,  # return_attn
                    0,  # n_query_suffix: state self-attn has no suffix to exclude
                    n_cross_kv_suffix,  # n_cross_kv_suffix_exclude: exclude rel_pose from state's cross-attn K/V
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, self_attn_img, cross_attn_img = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn,  # return_attn
                    n_query_suffix,  # n_query_suffix: exclude rel_pose from img self-attn K/V
                    0,  # n_cross_kv_suffix_exclude: img cross-attends to state, no exclusion needed
                    layer_inject,  # inject_before_attn
                    n_inject_suffix,  # n_inject_suffix
                    use_reentrant=not self.fixed_input_length,
                )
                attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
            else:
                f_state, _, self_attn_state, cross_attn_state = blk_state(
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn=return_attn,
                    n_query_suffix=0,  # state self-attn has no suffix to exclude
                    n_cross_kv_suffix_exclude=n_cross_kv_suffix,  # exclude rel_pose from state's cross-attn K/V
                )
                f_img, _, self_attn_img, cross_attn_img = blk_img(
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn=return_attn,
                    n_query_suffix=n_query_suffix,  # exclude rel_pose from img self-attn K/V
                    n_cross_kv_suffix_exclude=0,  # img cross-attends to state, no exclusion needed
                    inject_before_attn=layer_inject,
                    n_inject_suffix=n_inject_suffix,
                )
                attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
            final_output.append((f_state, f_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        rel_pose_feat,
        rel_pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
        inject_per_layer=None,
        n_inject_suffix=0,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, rel_pose_feat, rel_pose_pos,
            return_attn=return_attn, inject_per_layer=inject_per_layer, n_inject_suffix=n_inject_suffix,
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _get_pose_buffer_entry(self, frame_idx, out_pose_feat_i, global_img_feat_i=None, gt_c2w=None, pred_c2w=None):
        """Return (frame_idx, feature) tuple for pose_token_buffer based on prev_pose_input_type."""
        if self.prev_pose_input_type == 'img_feat':
            return (frame_idx, global_img_feat_i.squeeze(1).detach())
        elif self.prev_pose_input_type == '12d_pose':
            c2w = gt_c2w if gt_c2w is not None else pred_c2w
            if c2w is None:
                B = out_pose_feat_i.shape[0]
                c2w = torch.eye(4, device=out_pose_feat_i.device).unsqueeze(0).expand(B, -1, -1)
            pose_12d = c2w[:, :3].reshape(c2w.shape[0], -1)  # (B, 12)
            return (frame_idx, pose_12d.detach())
        elif self.prev_pose_input_type == 'img_feat_12d_pose':
            # Concat img_feat (B, enc_dim) + 12D pose (B, 12)
            img_feat = global_img_feat_i.squeeze(1).detach()  # (B, enc_dim)
            c2w = gt_c2w if gt_c2w is not None else pred_c2w
            if c2w is None:
                B = img_feat.shape[0]
                c2w = torch.eye(4, device=img_feat.device).unsqueeze(0).expand(B, -1, -1)
            pose_12d = c2w[:, :3].reshape(c2w.shape[0], -1).to(img_feat.device)  # (B, 12)
            return (frame_idx, torch.cat([img_feat, pose_12d.detach()], dim=-1))  # (B, enc_dim+12)
        else:  # decoder_output
            return (frame_idx, out_pose_feat_i.squeeze(1).detach())

    def _assemble_rel_pose_tokens(self, B, device, dtype, pose_token_buffer, ref_frame_indices=None):
        """Assemble relative pose tokens from buffer with dynamic token count.

        Args:
            B: batch size
            device: torch device
            dtype: torch dtype for positions
            pose_token_buffer: List[Tuple[int, Tensor]] — (frame_idx, feat) pairs
            ref_frame_indices: Optional[List[int]] — specify reference frame indices;
                None uses automatic sliding window (most recent entries from buffer)

        Returns:
            rel_pose_feat: (B, n_actual, D) or None if n_actual == 0
            rel_pose_pos:  (B, n_actual, 2) or None if n_actual == 0
            n_actual: int — number of actual tokens assembled
            ref_indices: List[int] — frame indices of the assembled references
        """
        if ref_frame_indices is None:
            # Automatic sliding window with max_ref_frames cap
            if self.max_ref_frames > 0:
                selected = pose_token_buffer[-self.max_ref_frames:][::-1]
            else:
                selected = pose_token_buffer[::-1]
            n_actual = len(selected)
            if n_actual == 0:
                return None, None, 0, [], None
            ref_indices = [frame_idx for frame_idx, _ in selected]
            ref_feats = [feat for _, feat in selected]
        else:
            # Explicit reference frame selection: look up from buffer
            buffer_dict = {frame_idx: feat for frame_idx, feat in pose_token_buffer}
            ref_indices = []
            ref_feats = []
            for idx in ref_frame_indices:
                if idx in buffer_dict:
                    ref_indices.append(idx)
                    ref_feats.append(buffer_dict[idx])
            # Apply max_ref_frames cap (same as automatic path)
            if self.max_ref_frames > 0 and len(ref_indices) > self.max_ref_frames:
                ref_indices = ref_indices[-self.max_ref_frames:]
                ref_feats = ref_feats[-self.max_ref_frames:]
            n_actual = len(ref_indices)
            if n_actual == 0:
                return None, None, 0, [], None
            # Reverse to match training convention: k=0 = most recent reference
            ref_indices = ref_indices[::-1]
            ref_feats = ref_feats[::-1]

        # Per-reference learnable base tokens + projected reference features
        n_base = self.relative_pose_token.shape[1]  # number of distinct base tokens
        if n_actual <= n_base:
            # Each reference k uses its own base token k
            rel_pose_feat = self.relative_pose_token[:, :n_actual, :].expand(B, -1, -1).clone()
        else:
            # Use all base tokens, then repeat the last one for extras
            base_part = self.relative_pose_token.expand(B, -1, -1).clone()
            extra = self.relative_pose_token[:, -1:, :].expand(B, n_actual - n_base, -1).clone()
            rel_pose_feat = torch.cat([base_part, extra], dim=1)

        if self.use_ref_condition:
            # Project reference features and inject into tokens
            projected_refs = torch.stack([self.prev_pose_proj(ref_feat) for ref_feat in ref_feats], dim=1)

            if self.inject_mode == "once":
                rel_pose_feat = rel_pose_feat + projected_refs
                inject_per_layer = None
            else:
                # per_layer_pre_attn: don't add here, pass to decoder for per-layer injection
                inject_per_layer = projected_refs
        else:
            # No reference conditioning: base tokens go through as-is
            inject_per_layer = None

        rel_pose_pos = -torch.ones(B, n_actual, 2, device=device, dtype=dtype)

        return rel_pose_feat, rel_pose_pos, n_actual, ref_indices, inject_per_layer

    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
        pose_token_buffer=None,
    ):
        if pose_token_buffer is None:
            pose_token_buffer = []
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )

            B = feat_i.shape[0]
            rel_pose_feat_i, rel_pose_pos_i, n_actual, ref_indices, inject_per_layer = \
                self._assemble_rel_pose_tokens(B, feat_i.device, pos_i.dtype, pose_token_buffer)
        else:
            pose_feat_i = None
            pose_pos_i = None
            rel_pose_feat_i = None
            rel_pose_pos_i = None
            n_actual = 0
            ref_indices = []



        new_state_feat, dec, _, _, _, _ = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            rel_pose_feat_i,
            rel_pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
            return_attn=False,
            inject_per_layer=inject_per_layer,
            n_inject_suffix=n_actual,
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )

        if n_actual > 0:
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:-n_actual].float(),
                dec[self.dec_depth * 3 // 4][:, 1:-n_actual].float(),
                dec[self.dec_depth][:, :-n_actual].float(),
            ]
            rel_pose_token = dec[self.dec_depth][:, -n_actual:].float()
            # Residual connection: add original prompt tokens to decoder output
            rel_pose_token = rel_pose_token + rel_pose_feat_i
        else:
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            rel_pose_token = None

        res = self._downstream_head(
            head_input,
            shape_i,
            pos=pos_i,
            rel_pose_token=rel_pose_token,
            n_valid_tokens=n_actual,
            ref_frame_indices=ref_indices,
        )

        # Update pose_token_buffer with (frame_idx, feat) tuple
        gt_c2w = views[i].get('camera_pose', None)  # (B, 4, 4) GT c2w, available during training
        pose_token_buffer.append(self._get_pose_buffer_entry(
            i, out_pose_feat_i, global_img_feat_i=global_img_feat_i, gt_c2w=gt_c2w))

        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )
        mem = new_mem * update_mask + mem * (1 - update_mask)

        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
            # Reset buffer on sequence reset
            if reset_mask.any():
                pose_token_buffer = []

        return res, (state_feat, mem), pose_token_buffer

    def _forward_impl(self, views, ret_state=False):
        # Encoder and state init use only frozen parameters — skip gradient tracking
        # to save memory and compute (same pattern as TBPTT path in inference.py)
        with torch.no_grad():
            shape, feat_ls, pos = self._encode_views(views)
            feat_raw = feat_ls[-1]
            state_feat, state_pos = self._init_state(feat_raw[0], pos[0])
            mem = self.pose_retriever.mem.expand(feat_raw[0].shape[0], -1, -1)
        # Detach all encoder outputs to cut the graph at the encoder-decoder boundary
        feat = [f.detach() for f in feat_raw]
        pos = [p.detach() for p in pos]
        shape = [s.detach() for s in shape]
        state_feat = state_feat.detach()
        state_pos = state_pos.detach()
        mem = mem.detach()
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []

        # Determine if we need attention maps for TTT3R
        model_update_type = getattr(self.config, 'model_update_type', 'cut3r')
        ttt3r_mode = getattr(self.config, 'ttt3r_mode', 'attn')
        use_ttt3r = model_update_type == "ttt3r"

        pose_token_buffer = []  # Sliding window buffer for pose tokens: List[Tuple[int, Tensor]]

        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )

                B = feat_i.shape[0]
                rel_pose_feat_i, rel_pose_pos_i, n_actual, ref_indices, inject_per_layer = \
                    self._assemble_rel_pose_tokens(B, feat_i.device, pos_i.dtype, pose_token_buffer)
            else:
                pose_feat_i = None
                pose_pos_i = None
                rel_pose_feat_i = None
                rel_pose_pos_i = None
                n_actual = 0
                ref_indices = []
    


            # For TTT3R attn mode, we need attention maps only for frames after the first
            # For TTT3R delta mode, we never need attention maps (use Flash Attention path)
            return_attn = use_ttt3r and ttt3r_mode == 'attn' and (i > 0)

            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                rel_pose_feat_i,
                rel_pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=return_attn,
                inject_per_layer=inject_per_layer,
                n_inject_suffix=n_actual,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1

            # Build head input and extract rel_pose_token
            if n_actual > 0:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:-n_actual].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:-n_actual].float(),
                    dec[self.dec_depth][:, :-n_actual].float(),
                ]
                rel_pose_token = dec[self.dec_depth][:, -n_actual:].float()
                # Residual connection: add original prompt tokens to decoder output
                rel_pose_token = rel_pose_token + rel_pose_feat_i
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
                rel_pose_token = None

            res = self._downstream_head(head_input, shape[i], pos=pos_i, rel_pose_token=rel_pose_token, n_valid_tokens=n_actual, ref_frame_indices=ref_indices)

            # Update pose_token_buffer with (frame_idx, feat) tuple
            gt_c2w = views[i].get('camera_pose', None)
            pose_token_buffer.append(self._get_pose_buffer_entry(
                i, out_pose_feat_i, global_img_feat_i=global_img_feat_i, gt_c2w=gt_c2w))

            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # TTT3R adaptive state update
            if i == 0:
                # First frame: always full update
                update_mask_state = update_mask
            else:
                if model_update_type == "cut3r":
                    # CUT3R: always full update
                    update_mask_state = update_mask
                elif model_update_type == "ttt3r":
                    ttt3r_bias = getattr(self.config, 'ttt3r_bias', 0.0)
                    if ttt3r_mode == 'delta':
                        # Delta mode: use state feature change as update signal
                        # No attention maps needed — stays on Flash Attention path
                        ttt3r_scale = getattr(self.config, 'ttt3r_scale', 10.0)
                        cos_sim = torch.nn.functional.cosine_similarity(
                            new_state_feat, state_feat, dim=-1)  # [B, n_state]
                        update_weight = torch.sigmoid((1 - cos_sim) * ttt3r_scale + ttt3r_bias)
                        update_mask_state = update_mask * update_weight[:, :, None]
                    else:
                        # Attn mode: adaptive update based on cross-attention confidence
                        cross_attn_state_list = list(cross_attn_state)
                        cross_attn_state_list = [x for x in cross_attn_state_list if x is not None]
                        if len(cross_attn_state_list) > 0:
                            cross_attn_tensor = torch.stack(cross_attn_state_list, dim=0)
                            cross_attn_tensor = rearrange(cross_attn_tensor, 'l b h nstate nimg -> b nstate nimg (l h)')
                            state_query_img_key = cross_attn_tensor.mean(dim=(-1, -2))  # [B, n_state]
                            update_weight = torch.sigmoid(state_query_img_key + ttt3r_bias)
                            update_mask_state = update_mask * update_weight[:, :, None]
                        else:
                            update_mask_state = update_mask
                else:
                    raise ValueError(f"Invalid model_update_type: {model_update_type}")

            state_feat = new_state_feat * update_mask_state + state_feat * (
                1 - update_mask_state
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state (always full update for mem)
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                if reset_mask.any():
                    pose_token_buffer = []
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem,
        pose_token_buffer=None,
        frame_idx=None,
        ref_frame_indices=None,
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes)

        feat_i = feat_ls[-1]
        pos_i = pos

        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )

            if pose_token_buffer is None:
                pose_token_buffer = []
            B = batch_size
            rel_pose_feat_i, rel_pose_pos_i, n_actual, ref_indices_out, inject_per_layer = \
                self._assemble_rel_pose_tokens(B, feat_i.device, pos_i.dtype, pose_token_buffer, ref_frame_indices)
        else:
            pose_feat_i = None
            pose_pos_i = None
            rel_pose_feat_i = None
            rel_pose_pos_i = None
            n_actual = 0
            ref_indices_out = []
            inject_per_layer = None



        new_state_feat, dec, _, _, _, _ = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            rel_pose_feat_i,
            rel_pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=False,
            inject_per_layer=inject_per_layer,
            n_inject_suffix=n_actual,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1

        # Build head input and extract rel_pose_token
        if n_actual > 0:
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:-n_actual].float(),
                dec[self.dec_depth * 3 // 4][:, 1:-n_actual].float(),
                dec[self.dec_depth][:, :-n_actual].float(),
            ]
            rel_pose_token = dec[self.dec_depth][:, -n_actual:].float()
            # Residual connection: add original prompt tokens to decoder output
            rel_pose_token = rel_pose_token + rel_pose_feat_i
        else:
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            rel_pose_token = None

        res = self._downstream_head(head_input, shape, pos=pos_i, rel_pose_token=rel_pose_token, n_valid_tokens=n_actual, ref_frame_indices=ref_indices_out)
        # Update pose_token_buffer with (frame_idx, feat) tuple
        if pose_token_buffer is None:
            pose_token_buffer = []
        # Use provided frame_idx or auto-increment based on buffer length
        actual_frame_idx = frame_idx if frame_idx is not None else len(pose_token_buffer)
        # For inference: use accumulated_c2w from view dict if available (for 12d_pose type)
        pred_c2w = view.get('accumulated_c2w', None)
        pose_token_buffer.append(self._get_pose_buffer_entry(
            actual_frame_idx, out_pose_feat_i, global_img_feat_i=global_img_feat_i, pred_c2w=pred_c2w))
        return res, view, pose_token_buffer

    def forward_recurrent(self, views, device, ret_state=False, ref_frame_indices_fn=None, keyframe_indices=None, on_frame_processed=None, buffer_pruning_fn=None):
        ress = []
        all_state_args = []
        processed_views = []  # Store processed views with minimal data
        # Initialize state variables
        state_feat = None
        state_pos = None
        init_state_feat = None
        mem = None
        init_mem = None
        pose_token_buffer = []  # Sliding window buffer: List[Tuple[int, Tensor]]
        pose_c2w_history = {}  # For 12d_pose inference: {frame_idx: c2w (B,4,4)}
        # Process views one at a time to save GPU memory
        for i, cpu_view in enumerate(views):
            # Move current view to GPU
            view = {}
            ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"])
            
            for name, value in cpu_view.items():
                if name in ignore_keys:
                    view[name] = value
                elif isinstance(value, tuple) or isinstance(value, list):
                    view[name] = [x.to(device, non_blocking=True) for x in value]
                else:
                    view[name] = value.to(device, non_blocking=True)
            
            # Set device from the current view
            current_device = view["img"].device
            batch_size = view["img"].shape[0]
            
            img_mask = view["img_mask"].reshape(-1, batch_size)
            ray_mask = view["ray_mask"].reshape(-1, batch_size)
            imgs = view["img"].unsqueeze(0)
            ray_maps = view["ray_map"].unsqueeze(0)
            
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=current_device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )
            
            imgs = imgs.view(-1, *imgs.shape[2:])
            ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])
            shapes = shapes.view(-1, 2).to(imgs.device)
            
            img_masks_flat = img_mask.view(-1)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
                
            ray_maps = ray_maps.permute(0, 3, 1, 2)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(selected_ray_maps, selected_shapes_ray)
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError


            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                # Only store initial state
                all_state_args.append(
                    (state_feat.cpu(), state_pos.cpu(), init_state_feat.cpu(), mem.cpu(), init_mem.cpu())
                )

            # Get reset mask for current view
            reset_mask = view.get("reset", torch.tensor(False))
            if isinstance(reset_mask, torch.Tensor):
                reset_mask = reset_mask.item() if reset_mask.numel() == 1 else reset_mask.any().item()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )

                B = feat_i.shape[0]
                requested_refs = ref_frame_indices_fn(i, pose_token_buffer) if ref_frame_indices_fn is not None else None
                rel_pose_feat_i, rel_pose_pos_i, n_actual, ref_indices, inject_per_layer = \
                    self._assemble_rel_pose_tokens(B, feat_i.device, pos_i.dtype, pose_token_buffer, requested_refs)
            else:
                pose_feat_i = None
                pose_pos_i = None
                rel_pose_feat_i = None
                rel_pose_pos_i = None
                n_actual = 0
                ref_indices = []
    


            # For TTT3R, we need attention maps only for frames after the first
            model_update_type = getattr(self.config, 'model_update_type', 'cut3r')
            ttt3r_mode = getattr(self.config, 'ttt3r_mode', 'attn')
            use_ttt3r = model_update_type == "ttt3r"
            return_attn = use_ttt3r and ttt3r_mode == 'attn' and (i > 0)

            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                rel_pose_feat_i,
                rel_pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=return_attn,
                inject_per_layer=inject_per_layer,
                n_inject_suffix=n_actual,
            )

            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            assert len(dec) == self.dec_depth + 1

            # Build head input and extract rel_pose_token
            if n_actual > 0:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:-n_actual].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:-n_actual].float(),
                    dec[self.dec_depth][:, :-n_actual].float(),
                ]
                rel_pose_token = dec[self.dec_depth][:, -n_actual:].float()
                # Residual connection: add original prompt tokens to decoder output
                rel_pose_token = rel_pose_token + rel_pose_feat_i
            else:
                head_input = [
                    dec[0].float(),
                    dec[self.dec_depth * 2 // 4][:, 1:].float(),
                    dec[self.dec_depth * 3 // 4][:, 1:].float(),
                    dec[self.dec_depth].float(),
                ]
                rel_pose_token = None

            res = self._downstream_head(head_input, shape, pos=pos_i, rel_pose_token=rel_pose_token, n_valid_tokens=n_actual, ref_frame_indices=ref_indices)

            # Update pose_c2w_history for 12d_pose / img_feat_12d_pose inference
            if self.prev_pose_input_type in ('12d_pose', 'img_feat_12d_pose'):
                B = feat_i.shape[0]
                rel_poses = res.get('relative_poses')  # (B, N, 4, 4) or None
                if i == 0 or not pose_c2w_history:
                    c2w_i = torch.eye(4, device=feat_i.device).unsqueeze(0).expand(B, -1, -1).clone()
                elif rel_poses is not None and ref_indices:
                    # Use most recent ref with known c2w for accumulation
                    c2w_i = None
                    for k, ref_idx in enumerate(ref_indices):
                        if ref_idx in pose_c2w_history:
                            T_rel = rel_poses[:, k].detach()
                            c2w_i = pose_c2w_history[ref_idx] @ torch.inverse(T_rel)
                            break
                    if c2w_i is None:
                        c2w_i = torch.eye(4, device=feat_i.device).unsqueeze(0).expand(B, -1, -1).clone()
                else:
                    c2w_i = torch.eye(4, device=feat_i.device).unsqueeze(0).expand(B, -1, -1).clone()
                pose_c2w_history[i] = c2w_i.detach()
            else:
                c2w_i = None

            # Update pose_token_buffer with (frame_idx, feat) tuple
            pose_token_buffer.append(self._get_pose_buffer_entry(
                i, out_pose_feat_i, global_img_feat_i=global_img_feat_i, pred_c2w=c2w_i))

            # Callback: let caller decide if this frame is a keyframe (before pruning)
            if on_frame_processed is not None:
                res['_img_tensor'] = view['img']  # pass raw img for loop detection
                on_frame_processed(i, res)
                res.pop('_img_tensor', None)



            if buffer_pruning_fn is not None:
                pose_token_buffer = buffer_pruning_fn(pose_token_buffer, keyframe_indices)

            # Move result to CPU immediately to save GPU memory
            res_cpu = {}
            for key, value in res.items():
                if isinstance(value, torch.Tensor):
                    res_cpu[key] = value.cpu()
                else:
                    res_cpu[key] = value
            ress.append(res_cpu)

            # Create minimal processed view (keep only essential data on CPU)
            processed_view = {
                "img": view["img"].cpu(),
                "idx": view.get("idx", i),
                "instance": view.get("instance", str(i)),
                "reset": view.get("reset", torch.tensor(False).unsqueeze(0)).cpu(),
            }
            processed_views.append(processed_view)

            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = img_mask & update
            else:
                update_mask = img_mask

            update_mask = update_mask[:, None, None].float()

            # TTT3R adaptive state update
            if i == 0:
                # First frame: always full update
                update_mask_state = update_mask
            else:
                if model_update_type == "cut3r":
                    # CUT3R: always full update
                    update_mask_state = update_mask
                elif model_update_type == "ttt3r":
                    ttt3r_bias = getattr(self.config, 'ttt3r_bias', 0.0)
                    if ttt3r_mode == 'delta':
                        ttt3r_scale = getattr(self.config, 'ttt3r_scale', 10.0)
                        cos_sim = torch.nn.functional.cosine_similarity(
                            new_state_feat, state_feat, dim=-1)
                        update_weight = torch.sigmoid((1 - cos_sim) * ttt3r_scale + ttt3r_bias)
                        update_mask_state = update_mask * update_weight[:, :, None]
                    else:
                        cross_attn_state_list = list(cross_attn_state)
                        cross_attn_state_list = [x for x in cross_attn_state_list if x is not None]
                        if len(cross_attn_state_list) > 0:
                            cross_attn_tensor = torch.stack(cross_attn_state_list, dim=0)
                            cross_attn_tensor = rearrange(cross_attn_tensor, 'l b h nstate nimg -> b nstate nimg (l h)')
                            state_query_img_key = cross_attn_tensor.mean(dim=(-1, -2))
                            update_weight = torch.sigmoid(state_query_img_key + ttt3r_bias)
                            update_mask_state = update_mask * update_weight[:, :, None]
                        else:
                            update_mask_state = update_mask
                else:
                    raise ValueError(f"Invalid model_update_type: {model_update_type}")

            # Selective state update: only keyframes write back to state/mem.
            # Non-keyframe frames are read-only (MUSt3R-style).
            # When keyframe_indices is None (no keyframe selection), always update.
            if keyframe_indices is None or i in keyframe_indices:
                state_feat = new_state_feat * update_mask_state + state_feat * (1 - update_mask_state)
                mem = new_mem * update_mask + mem * (1 - update_mask)

            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                if reset_mask.any():
                    pose_token_buffer = []
                    pose_c2w_history.clear()

            # Dynamic KF-based reset: reset encoder state + clear buffer
            # Keep pose_c2w_history so the retained KF can still chain c2w
            if res_cpu.get('_trigger_reset', False):
                state_feat = init_state_feat.clone()
                mem = init_mem.clone()
                current_kf_entry = pose_token_buffer[-1] if pose_token_buffer else None
                pose_token_buffer = [current_kf_entry] if current_kf_entry is not None else []

            # Only append final state if needed
            if ret_state and i == len(views) - 1:
                all_state_args.append(
                    (state_feat.cpu(), state_pos.cpu(), init_state_feat.cpu(), mem.cpu(), init_mem.cpu())
                )

            # Explicit cleanup
            del view, imgs, ray_maps, img_out, ray_out, feat_i, pos_i, res
            del new_state_feat, dec, head_input
            if 'global_img_feat_i' in locals():
                del global_img_feat_i
            if 'pose_feat_i' in locals():
                del pose_feat_i
            if 'out_pose_feat_i' in locals():
                del out_pose_feat_i
            if 'new_mem' in locals():
                del new_mem
                
            # Force GPU memory cleanup
            torch.cuda.empty_cache()
            
            # Print memory usage every 50 frames
            if (i + 1) % 50 == 0:
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    print(f"  After {i+1} frames: {allocated:.2f}GB GPU memory allocated")
            
        if ret_state:
            return ress, processed_views, all_state_args
        return ress, processed_views


if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
