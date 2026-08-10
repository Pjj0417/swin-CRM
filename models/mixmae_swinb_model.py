import math
import os

import torch
import torch.nn as nn


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device,
    )
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(
            x,
            self.drop_prob,
            self.training,
        )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size,
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1]
        )

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = (
            norm_layer(embed_dim)
            if norm_layer is not None
            else nn.Identity()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(
        self,
        input_resolution,
        dim,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(
            4 * dim,
            2 * dim,
            bias=False,
        )
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape

        if l != h * w:
            raise RuntimeError(
                f"PatchMerging expected {h*w} tokens, got {l}."
            )
        if h % 2 or w % 2:
            raise RuntimeError(
                f"PatchMerging requires even H/W, got {h}x{w}."
            )

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat(
            [x0, x1, x2, x3],
            dim=-1,
        )
        x = x.view(
            b,
            -1,
            4 * c,
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, c)
    )
    return windows


def window_reverse(
    windows,
    window_size,
    h,
    w,
):
    b = int(
        windows.shape[0]
        / (h * w / window_size / window_size)
    )
    x = windows.view(
        b,
        h // window_size,
        w // window_size,
        window_size,
        window_size,
        -1,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(b, h, w, -1)
    )
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = (
            window_size,
            window_size,
        )
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        table_size = (
            (2 * window_size - 1)
            * (2 * window_size - 1)
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                table_size,
                num_heads,
            )
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(
                coords_h,
                coords_w,
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(
            coords,
            1,
        )
        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )
        relative_coords = (
            relative_coords
            .permute(1, 2, 0)
            .contiguous()
        )
        relative_coords[:, :, 0] += (
            window_size - 1
        )
        relative_coords[:, :, 1] += (
            window_size - 1
        )
        relative_coords[:, :, 0] *= (
            2 * window_size - 1
        )
        relative_position_index = (
            relative_coords.sum(-1)
        )
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=True,
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )
        self.attn_drop = nn.Dropout(
            attn_drop
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(
            proj_drop
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b_, n, c = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                b_,
                n,
                3,
                self.num_heads,
                c // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        bias = bias.view(
            n,
            n,
            -1,
        )
        bias = (
            bias.permute(2, 0, 1)
            .contiguous()
        )
        attn = attn + bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(b_, n, c)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixMIMBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = (
            int(input_resolution[0]),
            int(input_resolution[1]),
        )
        self.num_heads = num_heads

        self.window_size = int(
            min(
                window_size,
                self.input_resolution[0],
                self.input_resolution[1],
            )
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(
                dim * mlp_ratio
            ),
            drop=drop,
        )

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape

        if l != h * w:
            raise RuntimeError(
                f"Block expected {h*w} tokens, got {l}."
            )

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        windows = window_partition(
            x,
            self.window_size,
        )
        windows = windows.view(
            -1,
            self.window_size
            * self.window_size,
            c,
        )

        windows = self.attn(windows)

        windows = windows.view(
            -1,
            self.window_size,
            self.window_size,
            c,
        )
        x = window_reverse(
            windows,
            self.window_size,
            h,
            w,
        )
        x = x.view(b, h * w, c)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(
            self.mlp(
                self.norm2(x)
            )
        )
        return x


class MixMIMLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rates=None,
        downsample=False,
    ):
        super().__init__()

        if drop_path_rates is None:
            drop_path_rates = [
                0.0
            ] * depth

        self.blocks = nn.ModuleList(
            [
                MixMIMBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path_rate=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            PatchMerging(
                input_resolution,
                dim=dim,
                norm_layer=nn.LayerNorm,
            )
            if downsample
            else None
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MixMAESwinBaseBackbone(nn.Module):
    """
    Swin-B/W14 encoder used by the official MixMAE Base model.

    Architecture:
      img=224, patch=4, embed=128
      depths=[2,2,18,2]
      heads=[4,8,16,32]
      windows=[14,14,14,7]
    """

    def __init__(
        self,
        drop_rate=0.0,
        drop_path_rate=0.10,
    ):
        super().__init__()

        self.embed_dim = 128
        self.depths = [2, 2, 18, 2]
        self.num_heads = [
            4, 8, 16, 32
        ]
        self.window_size = [
            14, 14, 14, 7
        ]
        self.num_features = 1024

        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            norm_layer=nn.LayerNorm,
        )

        self.patch_grid = (
            56,
            56,
        )

        dpr = torch.linspace(
            0,
            float(drop_path_rate),
            sum(self.depths),
        ).tolist()

        self.layers = nn.ModuleList()
        cursor = 0

        for stage_idx in range(4):
            depth = self.depths[
                stage_idx
            ]
            resolution = (
                56 // (2 ** stage_idx),
                56 // (2 ** stage_idx),
            )
            dim = 128 * (
                2 ** stage_idx
            )

            self.layers.append(
                MixMIMLayer(
                    dim=dim,
                    input_resolution=resolution,
                    depth=depth,
                    num_heads=self.num_heads[
                        stage_idx
                    ],
                    window_size=self.window_size[
                        stage_idx
                    ],
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=float(drop_rate),
                    attn_drop=0.0,
                    drop_path_rates=dpr[
                        cursor:
                        cursor + depth
                    ],
                    downsample=(
                        stage_idx < 3
                    ),
                )
            )
            cursor += depth

        self.pos_drop = nn.Dropout(
            p=float(drop_rate)
        )
        self.norm = nn.LayerNorm(
            self.num_features
        )
        self.avgpool = (
            nn.AdaptiveAvgPool1d(1)
        )

        # Official MixMAE checkpoint contains this frozen sin/cos
        # position embedding. It is learned as a tensor in the
        # checkpoint state_dict but does not require gradients.
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embed.num_patches,
                self.embed_dim,
            ),
            requires_grad=False,
        )

    def forward_tokens(self, x):
        x = self.patch_embed(x)
        x = (
            x
            + self.absolute_pos_embed
            .to(dtype=x.dtype)
        )
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    def forward_features(self, x):
        x = self.forward_tokens(x)
        x = self.avgpool(
            x.transpose(1, 2)
        )
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.forward_features(x)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "MixMAE checkpoint must be a dict."
        )

    for key in (
        "model",
        "state_dict",
        "model_ema",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def _clean_pretrain_state_dict(
    state_dict,
):
    """
    Official MixMAE pretrain checkpoint contains encoder + decoder.
    Keep encoder-compatible keys only.
    """
    cleaned = {}

    for key, value in state_dict.items():
        clean = key

        for prefix in (
            "module.",
            "model.",
            "backbone.",
        ):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]

        # Pretraining-only modules / classification heads.
        if (
            clean.startswith("decoder")
            or clean.startswith("mask_token")
            or clean.startswith("head.")
        ):
            continue

        cleaned[clean] = value

    return cleaned


def _coverage(
    state_dict,
    model,
):
    model_state = model.state_dict()
    total_numel = 0
    matched_numel = 0
    matched_keys = set()

    for key, tensor in model_state.items():
        # relative_position_index is deterministic buffer and can
        # legitimately differ across PyTorch/timm versions.
        if key.endswith(
            "relative_position_index"
        ):
            continue

        total_numel += tensor.numel()

        if (
            key in state_dict
            and tuple(
                state_dict[key].shape
            )
            == tuple(tensor.shape)
        ):
            matched_numel += (
                tensor.numel()
            )
            matched_keys.add(key)

    return (
        matched_numel
        / max(1, total_numel),
        matched_keys,
    )


class MixMAESwinBaseClassifier(nn.Module):
    def __init__(
        self,
        pretrained_path,
        num_classes=2,
        drop_rate=0.10,
        drop_path_rate=0.10,
        freeze_backbone=False,
    ):
        super().__init__()

        if not pretrained_path:
            raise ValueError(
                "Pass MixMAE pretrained checkpoint with --pretrained."
            )

        pretrained_path = os.path.expanduser(
            str(pretrained_path)
        )
        if not os.path.isfile(
            pretrained_path
        ):
            raise FileNotFoundError(
                "\nMixMAE pretrained checkpoint not found:\n"
                f"{pretrained_path}\n"
            )

        self.backbone = (
            MixMAESwinBaseBackbone(
                drop_rate=0.0,
                drop_path_rate=drop_path_rate,
            )
        )
        self.num_features = (
            self.backbone.num_features
        )

        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = (
            _clean_pretrain_state_dict(
                _extract_state_dict(
                    checkpoint
                )
            )
        )

        coverage, matched_keys = (
            _coverage(
                state_dict,
                self.backbone,
            )
        )

        print(
            "=" * 70,
            flush=True,
        )
        print(
            "[MixMAE] Loading official Swin-B/W14 MAE-style pretrained encoder",
            flush=True,
        )
        print(
            f"[MixMAE] Checkpoint: {pretrained_path}",
            flush=True,
        )
        print(
            "[MixMAE] Encoder parameter coverage: "
            f"{coverage * 100.0:.2f}%",
            flush=True,
        )

        if coverage < 0.95:
            missing = [
                key
                for key in self.backbone.state_dict()
                if (
                    key not in matched_keys
                    and not key.endswith(
                        "relative_position_index"
                    )
                )
            ][:20]

            raise RuntimeError(
                "MixMAE checkpoint did not match the Swin-B/W14 encoder. "
                f"Coverage={coverage * 100.0:.2f}%. "
                f"First unmatched keys: {missing}"
            )

        incompatible = (
            self.backbone.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        # Buffers such as relative_position_index are allowed to differ.
        missing_filtered = [
            key
            for key in incompatible.missing_keys
            if not key.endswith(
                "relative_position_index"
            )
        ]

        print(
            "[MixMAE] Missing non-buffer keys: "
            f"{len(missing_filtered)}",
            flush=True,
        )
        print(
            "[MixMAE] Unexpected keys: "
            f"{len(incompatible.unexpected_keys)}",
            flush=True,
        )
        print(
            "=" * 70,
            flush=True,
        )

        if freeze_backbone:
            for parameter in (
                self.backbone.parameters()
            ):
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=float(drop_rate)
            ),
            nn.Linear(
                self.num_features,
                int(num_classes),
            ),
        )

        nn.init.normal_(
            self.classifier[-1].weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            self.classifier[-1].bias
        )

    def forward_features(self, images):
        return (
            self.backbone
            .forward_features(images)
        )

    def forward(self, images):
        return self.classifier(
            self.forward_features(images)
        )


def build_mixmae_swinb(
    config=None,
    num_classes=None,
    model_name="mixmae_swin_base_w14_224",
    drop_rate=0.10,
    drop_path_rate=0.10,
    freeze_backbone=False,
    pretrained_path=None,
):
    if (
        config is not None
        and hasattr(config, "MODEL")
    ):
        if num_classes is None:
            num_classes = int(
                config.MODEL.NUM_CLASSES
            )

        if hasattr(
            config.MODEL,
            "DROP_RATE",
        ):
            drop_rate = float(
                config.MODEL.DROP_RATE
            )

        if hasattr(
            config.MODEL,
            "DROP_PATH_RATE",
        ):
            drop_path_rate = float(
                config.MODEL.DROP_PATH_RATE
            )

        if (
            pretrained_path is None
            and hasattr(
                config,
                "PRETRAINED",
            )
            and config.PRETRAINED
        ):
            pretrained_path = str(
                config.PRETRAINED
            )

    if num_classes is None:
        num_classes = 2

    return MixMAESwinBaseClassifier(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
    )
