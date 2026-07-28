# Copyright (c) OpenMMLab. All rights reserved.
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def vit(cfg):
    """
    Spiral cosine-similarity token pruning ViT.

    Behavior:
      - per-image pruning (no batch-shared mask)
      - compute cosine once per image on spiral order
      - threshold cascade over the same cosine vector
      - drop the 2nd token in each matched spiral pair
      - encoder runs once on batched kept tokens [B, K, D]
      - dropped tokens bypass encoder and only receive final norm
    """
    return ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        token_drop_enable=True,
        token_keep_ratio=0.50,
        token_drop_thresholds=(0.90, 0.85, 0.80, 0.75, 0.70),
    )


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    return torch.cat([cls_token, abs_pos], dim=1) if cls_token is not None else abs_pos


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or (dim // num_heads)
        all_head_dim = head_dim * num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x, **kwargs):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), (Hp, Wp)


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding."""

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                feat = backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = feat.shape[-2:]
                feature_dim = feat.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        return self.proj(self.backbone(x)[-1].flatten(2).transpose(1, 2))


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        use_checkpoint=False,
        frozen_stages=-1,
        ratio=1,
        last_norm=True,
        patch_padding='pad',
        freeze_attn=False,
        freeze_ffn=False,
        token_drop_enable=False,
        token_keep_ratio=1.0,
        token_drop_thresholds=(0.90, 0.85, 0.80, 0.75, 0.70),
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        self.token_drop_enable = bool(token_drop_enable)
        self.token_keep_ratio = float(token_keep_ratio)
        self.token_drop_thresholds = [float(t) for t in token_drop_thresholds]
        self._spiral_cache = {}

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                ratio=ratio,
            )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        print(
            "[ViT][InitConfig] "
            f"TOKEN_DROP_ENABLE={self.token_drop_enable} | "
            f"TOKEN_KEEP_RATIO={self.token_keep_ratio} | "
            f"TOKEN_DROP_THRESHOLDS={self.token_drop_thresholds} | "
            "TOKEN_DROP_MODE=batched_per_image_spiral_cos_once"
        )

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            block = self.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for block in self.blocks:
                block.attn.eval()
                block.norm1.eval()
                for param in block.attn.parameters():
                    param.requires_grad = False
                for param in block.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for block in self.blocks:
                block.mlp.eval()
                block.norm2.eval()
                for param in block.mlp.parameters():
                    param.requires_grad = False
                for param in block.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _run_blocks(self, x):
        for block in self.blocks:
            x = checkpoint.checkpoint(block, x) if self.use_checkpoint else block(x)
        return self.last_norm(x)

    def _get_spiral_indices(self, Hp, Wp, device):
        key = (int(Hp), int(Wp), device.type)
        if key in self._spiral_cache:
            return self._spiral_cache[key]

        top, bottom = 0, Hp - 1
        left, right = 0, Wp - 1
        coords = []
        while top <= bottom and left <= right:
            for c in range(left, right + 1):
                coords.append((top, c))
            top += 1

            for r in range(top, bottom + 1):
                coords.append((r, right))
            right -= 1

            if top <= bottom:
                for c in range(right, left - 1, -1):
                    coords.append((bottom, c))
                bottom -= 1

            if left <= right:
                for r in range(bottom, top - 1, -1):
                    coords.append((r, left))
                left += 1

        idx = torch.tensor([r * Wp + c for r, c in coords], device=device, dtype=torch.long)
        self._spiral_cache[key] = idx
        return idx

    @torch.no_grad()
    def _threshold_sequential_drop_spiral(self, cos_vec, keep_ratio):
        """
        cos_vec: (N-1,) cosine along spiral pairs (i -> i+1)
        keep_ratio: keep fraction in (0,1]
        Return: dropped_spiral (N,) bool mask over spiral positions (0..N-1)
        Rule: scan pairs in order, drop the 2nd token in pair if cos >= thr,
              then lower thr if still not enough.
        """
        N = int(cos_vec.numel()) + 1
        keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
        if keep_ratio >= 1.0 or N <= 1:
            return torch.zeros(N, device=cos_vec.device, dtype=torch.bool)

        target_drop = max(0, min(int(round((1.0 - keep_ratio) * N)), N - 1))
        dropped = torch.zeros(N, device=cos_vec.device, dtype=torch.bool)
        dropped_count = 0

        for thr in self.token_drop_thresholds:
            if dropped_count >= target_drop:
                break
            for i in range(N - 1):
                if dropped_count >= target_drop:
                    break
                if cos_vec[i] >= thr and not dropped[i + 1]:
                    dropped[i + 1] = True
                    dropped_count += 1

        if dropped_count < target_drop:
            for i in range(N - 1):
                if dropped_count >= target_drop:
                    break
                if not dropped[i + 1]:
                    dropped[i + 1] = True
                    dropped_count += 1

        return dropped

    @torch.no_grad()
    def _select_keep_drop_indices_single(self, x_single, Hp, Wp):
        """
        x_single: (N, D) AFTER pos embed addition for one image.
        Returns keep/drop row-major indices for that image.
        """
        N = x_single.shape[0]
        keep_ratio = max(0.0, min(1.0, float(self.token_keep_ratio)))
        if keep_ratio >= 1.0 or N <= 1:
            keep = torch.arange(N, device=x_single.device, dtype=torch.long)
            return keep, keep.new_empty(0)

        spiral = self._get_spiral_indices(Hp, Wp, x_single.device)
        x_spiral = x_single.index_select(0, spiral)
        cos_vec = F.cosine_similarity(x_spiral[:-1], x_spiral[1:], dim=-1)
        dropped_spiral = self._threshold_sequential_drop_spiral(cos_vec, keep_ratio)

        drop = spiral[dropped_spiral]
        keep_mask = torch.ones(N, device=x_single.device, dtype=torch.bool)
        keep_mask[drop] = False
        keep = keep_mask.nonzero(as_tuple=False).flatten()
        return keep, drop.sort().values

    @torch.no_grad()
    def _select_keep_drop_indices_batch(self, x, Hp, Wp):
        """
        x: (B, N, D) AFTER pos embed addition.

        Returns:
            keep_idx: (B, K) row-major kept token indices.
            drop_idx: (B, N-K) row-major dropped token indices.

        The token positions may differ for each image, but because keep_ratio is fixed,
        K is the same for all images. This lets us gather kept tokens as one batch
        tensor and run the ViT encoder once instead of B times.
        """
        keep_list = []
        drop_list = []
        B = x.shape[0]
        for b in range(B):
            keep, drop = self._select_keep_drop_indices_single(x[b], Hp, Wp)
            keep_list.append(keep)
            drop_list.append(drop)

        keep_idx = torch.stack(keep_list, dim=0)

        # drop may be empty only for edge cases. Keep a valid empty tensor shape.
        if drop_list[0].numel() == 0:
            drop_idx = keep_idx.new_empty((B, 0))
        else:
            drop_idx = torch.stack(drop_list, dim=0)

        return keep_idx, drop_idx

    def forward_features(self, x):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # Keep the original HaMeR/WiLoR behavior: no explicit cls token is used,
            # but the cls positional embedding is added as a global offset.
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        if not (self.token_drop_enable and self.token_keep_ratio < 1.0):
            x = self._run_blocks(x)
            return x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        N, D = x.shape[1], x.shape[2]

        # 1) Select per-image keep/drop indices, but do NOT run ViT per image.
        keep_idx, drop_idx = self._select_keep_drop_indices_batch(x, Hp, Wp)

        # 2) Gather all kept tokens into a fixed-shape batch tensor: [B, K, D].
        keep_expand = keep_idx.unsqueeze(-1).expand(-1, -1, D)
        x_keep = torch.gather(x, dim=1, index=keep_expand)

        # 3) Run the transformer blocks ONCE for the whole batch.
        kept_out = self._run_blocks(x_keep)

        # 4) Scatter encoded kept tokens back to their original spatial positions.
        x_full = x.new_empty(B, N, D)
        x_full.scatter_(dim=1, index=keep_expand, src=kept_out)

        # 5) Dropped tokens bypass encoder and receive only the final norm,
        # matching the original SpiralRef behavior.
        if drop_idx.numel() > 0:
            drop_expand = drop_idx.unsqueeze(-1).expand(-1, -1, D)
            x_drop = torch.gather(x, dim=1, index=drop_expand)
            drop_out = self.last_norm(x_drop)
            x_full.scatter_(dim=1, index=drop_expand, src=drop_out)

        return x_full.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

    def forward(self, x):
        return self.forward_features(x)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

