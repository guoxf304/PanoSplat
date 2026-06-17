# aggregator.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

from functools import partial
import logging
import os
import math
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import requests

from panovggt.dinov2.layers import Mlp, PatchEmbed
from panovggt.layers.pos_embed import RoPE2D, PositionGetter
from panovggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from panovggt.layers.block import BlockRope
from panovggt.layers.attention import FlashAttentionRope

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    Aggregator decoder module:
      1. RoPE (Rotary Position Encoding) for relative positional relationships.
      2. Layer-wise additive absolute spherical position encoding for geometric priors.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 5,
        rope_freq: int = 100,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        qk_norm: bool = True,
        init_values: float = 0.01,
        num_dec_blk_not_to_checkpoint: int = 4,
        use_checkpoint: bool = True,
        patch_embed: str = "dinov2_vitl14_reg",
        use_pano_pos: bool = True,
        pos_mlp_hidden: int = 1024,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.dec_embed_dim = embed_dim
        self.depth = depth
        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = False
        self.hook_layer_indices = self._default_hook_indices(depth)

        # 1) DINO patch embedding
        self._build_patch_embed(
            patch_embed=patch_embed,
            img_size=img_size, # 518
            patch_size=patch_size, # 14
            num_register_tokens=num_register_tokens, # 5
            embed_dim=embed_dim, # 1024
        )

        # 2) RoPE (relative position encoding within attention)
        self.rope = RoPE2D(freq=float(rope_freq)) if rope_freq and rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # 3) Absolute spherical position encoding (additive)
        self.use_pano_pos = use_pano_pos # True
        if self.use_pano_pos:
            self.pano_pos_mlp = nn.Sequential(
                nn.Linear(4, pos_mlp_hidden), # 4: sin(theta), cos(theta), sin(phi), cos(phi)
                nn.GELU(),
                nn.Linear(pos_mlp_hidden, self.dec_embed_dim), # 1024
            )
            self.alpha_pos = nn.Parameter(torch.tensor(0.0))

        # 4) Decoder blocks
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=self.dec_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=init_values,
                qk_norm=qk_norm,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            )
            for _ in range(depth) # 24
        ])

        # 5) Register tokens and normalization buffers
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        self.register_buffer(
            "_resnet_mean",
            torch.tensor(_RESNET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.tensor(_RESNET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _default_hook_indices(depth: int) -> List[int]:
        """Decoder block indices (1-based) for multi-scale DPT hooks."""
        if depth >= 36:
            return [8, 18, 27, 35]
        d = max(depth, 4)
        return [
            max(1, d // 4),
            max(1, d // 2),
            max(1, (3 * d) // 4),
            d,
        ]

    # ------------------------------------------------------------------ #
    #                        Patch Embedding                               #
    # ------------------------------------------------------------------ #
    def _build_patch_embed(
        self,
        patch_embed: str,
        img_size: int,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
        interpolate_antialias: bool = True,
        interpolate_offset: float = 0.0,
        block_chunks: int = 0,
        init_values: float = 1.0,
    ):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size,
                in_chans=3, embed_dim=embed_dim,
            )
            self.patch_embed_dim = embed_dim
            self.needs_projection = False
            return

        vit_registry = {
            "dinov2_vitl14_reg": (vit_large, 1024, "dinov2_vitl14"),
            "dinov2_vitb14_reg": (vit_base, 768, "dinov2_vitb14"),
            "dinov2_vits14_reg": (vit_small, 384, "dinov2_vits14"),
            "dinov2_vitg2_reg": (vit_giant2, 1536, "dinov2_vitg14"),
        }
        vit_url_map = {
            "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            "dinov2_vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "dinov2_vitg2_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        }

        vit_fn, vit_dim, hub_name = vit_registry[patch_embed]
        self.patch_embed = vit_fn(
            img_size=518, patch_size=patch_size,
            num_register_tokens=4,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
            block_chunks=block_chunks,
            init_values=init_values,
        )

        # Attempt to load DINOv2 pretrained weights
        self._try_load_dinov2(hub_name, vit_url_map.get(patch_embed), patch_embed)

        self.patch_embed_dim = vit_dim
        self.needs_projection = vit_dim != self.dec_embed_dim
        if self.needs_projection:
            self.patch_embed_projection = nn.Linear(vit_dim, self.dec_embed_dim)

        if hasattr(self.patch_embed, "mask_token"):
            delattr(self.patch_embed, "mask_token")

    def _load_dinov2_into_patch_embed(self, state, model_dict, source: str) -> bool:
        if isinstance(state, dict) and "teacher" in state:
            state = state["teacher"]
        matched = {
            k: v for k, v in state.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        logger.info(f"Matched {len(matched)}/{len(model_dict)} layers from {source}")
        if not matched:
            return False
        model_dict.update(matched)
        self.patch_embed.load_state_dict(model_dict)
        return True

    def _try_load_dinov2(self, hub_name: str, url: Optional[str], patch_embed_key: str):
        """Load DINOv2 weights: local cache first, then torch.hub, then URL download."""
        success = False
        model_dict = self.patch_embed.state_dict()
        weights_dir = Path(os.path.expanduser("~/.cache/panovggt/weights"))
        local_path = weights_dir / f"{patch_embed_key}_pretrain.pth"

        # Method 1: local cache (avoids slow/hung GitHub access via torch.hub)
        if local_path.exists():
            try:
                logger.info(f"Loading DINOv2 weights from local cache: {local_path}")
                state = torch.load(local_path, map_location="cpu")
                success = self._load_dinov2_into_patch_embed(
                    state, model_dict, source=str(local_path)
                )
            except Exception as e:
                logger.warning(f"Local DINOv2 cache load failed: {e}")

        # Method 2: torch.hub
        if not success:
            try:
                logger.info(f"Loading DINOv2 weights for {hub_name} via torch.hub")
                pretrained = torch.hub.load("facebookresearch/dinov2", hub_name)
                success = self._load_dinov2_into_patch_embed(
                    pretrained.state_dict(), model_dict, source="torch.hub"
                )
            except Exception as e:
                logger.warning(f"torch.hub load failed: {e}")

        # Method 3: Direct download
        if not success and url:
            try:
                logger.info(f"Downloading DINOv2 weights from {url}")
                weights_dir.mkdir(parents=True, exist_ok=True)
                if not local_path.exists():
                    r = requests.get(url, allow_redirects=True, timeout=120)
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                state = torch.load(local_path, map_location="cpu")
                success = self._load_dinov2_into_patch_embed(
                    state, model_dict, source=str(local_path)
                )
            except Exception as e:
                logger.warning(f"Direct download failed: {e}")

        if success:
            for p in self.patch_embed.parameters():
                p.requires_grad = True
            logger.info("DINOv2 weights loaded; parameters set to trainable")
        else:
            logger.warning("Could not load DINOv2 pretrained weights; using random init")

    # ------------------------------------------------------------------ #
    #                           Decode                                     #
    # ------------------------------------------------------------------ #
    def _decode(self, hidden: torch.Tensor, B: int, S: int, H: int, W: int):
        BN, hw, C = hidden.shape # B*S,2738,1024
        assert BN == B * S
        Hp, Wp = H // self.patch_size, W // self.patch_size # 37，74
        assert hw == Hp * Wp # 2738

        # Prepend register tokens
        reg = self.register_token.repeat(B, S, 1, 1).reshape(B * S, self.patch_start_idx, C) # (B*S, 5, 1024)
        hidden = torch.cat([reg, hidden], dim=1)  # (B*S, P, C)
        P = hidden.shape[1] # 2738+5=2743

        # --- RoPE position indices ---
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, Hp, Wp, device=hidden.device) # (B*S, 2738, 2)
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2,
                device=hidden.device, dtype=pos.dtype,
            ) # 寄存器的位置是 0
            pos = torch.cat([pos_special, pos], dim=1)

        # --- Absolute spherical position encoding ---
        pos_embed_single_bs = None
        pos_embed_multi_bsp = None
        if self.use_pano_pos:
            device = hidden.device
            ys = torch.arange(Hp, device=device, dtype=torch.float32) + 0.5 # (37,)
            xs = torch.arange(Wp, device=device, dtype=torch.float32) + 0.5 # (74,)
            theta = (ys[:, None] / Hp - 0.5) * math.pi # (37, 74)
            phi = (xs[None, :] / Wp - 0.5) * (2 * math.pi) # (37, 74)
            theta = theta.expand(Hp, Wp)
            phi = phi.expand(Hp, Wp) # (37, 74)

            pos_feats = torch.stack(
                [torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)], # (37*74, 4)
                dim=-1,
            ).reshape(Hp * Wp, 4)

            pos_embed_patch = self.pano_pos_mlp(pos_feats) # (2738, 1024)，通过一个MLP得到位置编码

            # Register tokens get zero positional encoding
            zeros_reg = torch.zeros(self.patch_start_idx, C, device=device, dtype=hidden.dtype) # (5, 1024)
            pos_embed_single = torch.cat([zeros_reg, pos_embed_patch], dim=0)  # (P, C) # (2743, 1024)

            pos_embed_single_bs = pos_embed_single.unsqueeze(0).expand(B * S, -1, -1) # (B*S, 2743, 1024)
            pos_embed_multi_bsp = (
                pos_embed_single.unsqueeze(0).unsqueeze(0)
                .expand(B, S, P, C)
                .reshape(B, S * P, C) # (B, S*P, 1024)
            )

        # --- Decoder loop ---
        last_two = []
        hook_outputs = {}
        for i, blk in enumerate(self.decoder):
            if i % 2 == 0:
                # Single-frame branch
                h_in = hidden.reshape(B * S, P, C) # (B*S, 2743, 1024)
                if self.use_pano_pos:
                    h_in = h_in + self.alpha_pos * pos_embed_single_bs # 绝对球面位置编码乘以一个可训练的参数
                p_in = None if pos is None else pos.reshape(B * S, P, -1) # (B*S, 2743, 2)
            else:
                # Multi-frame branch
                h_in = hidden.reshape(B, S * P, C) # (B, S*P, 1024)
                if self.use_pano_pos:
                    h_in = h_in + self.alpha_pos * pos_embed_multi_bsp # 绝对球面位置编码乘以一个可训练的参数
                p_in = None if pos is None else pos.reshape(B, S * P, -1) # (B, S*P, 2)

            if self.training and self.use_checkpoint and i >= self.num_dec_blk_not_to_checkpoint:
                h_out = checkpoint(blk, h_in, p_in, use_reentrant=False) # 使用梯度检查点技术
            else:
                h_out = blk(h_in, xpos=p_in)

            hidden = h_out.reshape(B * S, P, C) # (B*S, 2743, 1024)

            layer_idx = i + 1
            if layer_idx in self.hook_layer_indices:
                hook_outputs[layer_idx] = hidden

            if layer_idx in [self.depth - 1, self.depth]:
                last_two.append(hidden) # 最后两层特征

        last_two_cat = torch.cat(last_two, dim=-1) if len(last_two) == 2 else last_two[-1] # (B*S, 2743, 2048)

        hook_list = [hook_outputs[idx] for idx in self.hook_layer_indices if idx in hook_outputs]
        if len(hook_list) < len(self.hook_layer_indices):
            hook_list = [hidden for _ in self.hook_layer_indices]

        pos_2d = None if pos is None else pos.reshape(B * S, P, -1) # (B*S, 2743, 2)
        return hook_list, last_two_cat, pos_2d

    # ------------------------------------------------------------------ #
    #                           Forward                                    #
    # ------------------------------------------------------------------ #
    def forward(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int, Optional[torch.Tensor]]:
        """
        Args:
            images: (B, S, 3, H, W) input panoramic images.

        Returns:
            hook_tokens_list: list of (B, S, P, C) multi-scale features.
            output_list: [(B, S, P, 2*C)] final aggregated features.
            patch_start_idx: number of register tokens prepended.
            pos_2d: optional RoPE position embeddings.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize
        images = (images - self._resnet_mean) / self._resnet_std # 归一化

        # Patch embed
        x = images.view(B * S, C_in, H, W) # (B*S, 3, 518, 1036)
        patch_tokens = self.patch_embed(x) # (B*S, 2738, 1024)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        if getattr(self, "needs_projection", False):
            patch_tokens = self.patch_embed_projection(patch_tokens)

        # Decode
        hook_list, hidden_cat, pos_2d = self._decode(patch_tokens, B, S, H, W)

        P = hidden_cat.shape[1] # 2743
        C = self.dec_embed_dim
        C2 = hidden_cat.shape[-1] # 2048
        hook_tokens_list = []
        for h in hook_list:
            if h.shape[-1] != C:
                raise RuntimeError(
                    f"Hook token dim {h.shape[-1]} != dec_embed_dim {C}; "
                    "expected per-layer hooks, not fused decoder tokens."
                )
            hook_tokens_list.append(h.view(B, S, P, C))
        output = hidden_cat.view(B, S, P, C2) # (B, S, 2743, 2048)
        return hook_tokens_list, [output], self.patch_start_idx, pos_2d