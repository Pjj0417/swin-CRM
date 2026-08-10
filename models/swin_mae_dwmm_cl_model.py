"""
Swin-B + MAE + DWMM (DW-MSA / DSW-MSA).

The deformable attention core follows the DW-MSA / DSW-MSA implementation
provided by the user:
- local 3x3 offset prediction
- bilinear K/V resampling inside each Swin window
- per-head deformation gate
- depth-wise local positional enhancement (LEPE)
- zero-scaled output refinement
- ordinary/shifted window alternation

This file adds:
1) a Swin-B configuration for fair comparison with the existing Swin-MAE run;
2) MAE-style 75% grouped patch masking and RGB patch reconstruction;
3) a fine-tuning classifier that loads only the MAE-pretrained DWMM encoder.
"""

import copy
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_


Size2T = Union[int, Tuple[int, int]]
NormLayer = Callable[[int], nn.Module]

SWIN_MAE_DWMM_MODEL_NAME = "swin_base_patch4_window7_224_dwmm"

SWIN_B_EMBED_DIM = 128
SWIN_B_DEPTHS = (2, 2, 18, 2)
SWIN_B_NUM_HEADS = (4, 8, 16, 32)
SWIN_B_WINDOW_SIZE = 7

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(
    x: torch.Tensor,
    window_size: Size2T,
) -> torch.Tensor:
    window_h, window_w = to_2tuple(window_size)
    batch_size, height, width, channels = x.shape

    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            f"Feature map size ({height}, {width}) must be divisible by "
            f"window size ({window_h}, {window_w})."
        )

    x = x.view(
        batch_size,
        height // window_h,
        window_h,
        width // window_w,
        window_w,
        channels,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_h, window_w, channels)
    )
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: Size2T,
    height: int,
    width: int,
) -> torch.Tensor:
    window_h, window_w = to_2tuple(window_size)

    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            f"Output size ({height}, {width}) must be divisible by "
            f"window size ({window_h}, {window_w})."
        )

    windows_per_image = (
        (height // window_h)
        * (width // window_w)
    )

    if windows.shape[0] % windows_per_image != 0:
        raise ValueError(
            "The number of windows is incompatible with the requested output size."
        )

    batch_size = windows.shape[0] // windows_per_image

    x = windows.view(
        batch_size,
        height // window_h,
        width // window_w,
        window_h,
        window_w,
        -1,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch_size, height, width, -1)
    )
    return x


class DeformableWindowAttention(nn.Module):
    """
    DW-MSA without cyclic shift; DSW-MSA after cyclic shift + attention mask.
    """

    def __init__(
        self,
        dim: int,
        window_size: Size2T,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.offset_scale = float(offset_scale)

        self.offset_norm = nn.LayerNorm(dim)

        self.deform_gate = nn.Parameter(
            torch.full((num_heads, 1, 1), -1.5)
        )

        self.lepe_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=True,
        )
        self.lepe_scale = nn.Parameter(
            torch.zeros(num_heads, 1, 1)
        )

        self.scale = (
            qk_scale
            if qk_scale is not None
            else self.head_dim ** -0.5
        )

        window_h, window_w = self.window_size
        relative_position_count = (
            (2 * window_h - 1)
            * (2 * window_w - 1)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                relative_position_count,
                num_heads,
            )
        )

        coords_h = torch.arange(window_h)
        coords_w = torch.arange(window_w)
        coords = torch.stack(
            torch.meshgrid(
                coords_h,
                coords_w,
                indexing="ij",
            )
        )
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = (
            coords_flatten[:, :, None]
            - coords_flatten[:, None, :]
        )
        relative_coords = (
            relative_coords
            .permute(1, 2, 0)
            .contiguous()
        )
        relative_coords[:, :, 0] += window_h - 1
        relative_coords[:, :, 1] += window_w - 1
        relative_coords[:, :, 0] *= 2 * window_w - 1

        relative_position_index = (
            relative_coords.sum(-1)
        )
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )

        self.offset_net = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                dim,
                2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.output_refine = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.output_refine_scale = nn.Parameter(
            torch.tensor(0.0)
        )

        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

        # Identity-preserving deformation initialization.
        nn.init.zeros_(
            self.offset_net[-1].weight
        )
        nn.init.zeros_(
            self.offset_net[-1].bias
        )
        nn.init.zeros_(
            self.lepe_conv.weight
        )
        nn.init.zeros_(
            self.lepe_conv.bias
        )
        nn.init.zeros_(
            self.output_refine[-1].weight
        )
        nn.init.zeros_(
            self.output_refine[-1].bias
        )

        self.last_offsets = None
        self.last_deform_gate = None
        self.last_lepe_scale = None

    def _make_sampling_grid(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape
        window_h, window_w = self.window_size

        if num_tokens != window_h * window_w:
            raise ValueError(
                f"Token count {num_tokens} does not match "
                f"window size {self.window_size}."
            )

        offset_input = self.offset_norm(x)
        feature = (
            offset_input.transpose(1, 2)
            .contiguous()
            .view(
                batch_windows,
                channels,
                window_h,
                window_w,
            )
        )

        offsets = self.offset_net(feature)
        offsets = (
            offsets
            - offsets.mean(
                dim=(2, 3),
                keepdim=True,
            )
        )
        offsets = (
            torch.tanh(offsets)
            * self.offset_scale
        )

        # Diagnostics only; detach so the stored tensor does not retain
        # the previous autograd graph.
        self.last_offsets = offsets.detach()

        dtype = x.dtype
        device = x.device

        base_y, base_x = torch.meshgrid(
            torch.arange(
                window_h,
                device=device,
                dtype=dtype,
            ),
            torch.arange(
                window_w,
                device=device,
                dtype=dtype,
            ),
            indexing="ij",
        )
        base_grid = torch.stack(
            (base_x, base_y),
            dim=-1,
        )
        base_grid = (
            base_grid.unsqueeze(0)
            .expand(
                batch_windows,
                -1,
                -1,
                -1,
            )
        )

        offsets = (
            offsets
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        sampling_grid = (
            base_grid + offsets
        )

        sampling_grid_x = (
            sampling_grid[..., 0]
            .clamp(
                0.0,
                float(
                    max(
                        window_w - 1,
                        0,
                    )
                ),
            )
        )
        sampling_grid_y = (
            sampling_grid[..., 1]
            .clamp(
                0.0,
                float(
                    max(
                        window_h - 1,
                        0,
                    )
                ),
            )
        )
        sampling_grid = torch.stack(
            (
                sampling_grid_x,
                sampling_grid_y,
            ),
            dim=-1,
        )

        if window_w > 1:
            sampling_grid[..., 0] = (
                2.0
                * sampling_grid[..., 0]
                / (window_w - 1)
                - 1.0
            )
        else:
            sampling_grid[..., 0] = 0.0

        if window_h > 1:
            sampling_grid[..., 1] = (
                2.0
                * sampling_grid[..., 1]
                / (window_h - 1)
                - 1.0
            )
        else:
            sampling_grid[..., 1] = 0.0

        return sampling_grid

    def _deform_sample(
        self,
        feature: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_windows,
            num_heads,
            _,
            head_dim,
        ) = feature.shape
        window_h, window_w = self.window_size

        feature = (
            feature
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(
                batch_windows * num_heads,
                head_dim,
                window_h,
                window_w,
            )
        )

        head_grid = (
            grid[:, None, ...]
            .expand(
                -1,
                num_heads,
                -1,
                -1,
                -1,
            )
            .reshape(
                batch_windows * num_heads,
                window_h,
                window_w,
                2,
            )
        )

        sampled = F.grid_sample(
            feature,
            head_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        sampled = (
            sampled.view(
                batch_windows,
                num_heads,
                head_dim,
                window_h * window_w,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        return sampled

    def _local_position_enhancement(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape
        window_h, window_w = self.window_size

        feature = (
            x.transpose(1, 2)
            .contiguous()
            .view(
                batch_windows,
                channels,
                window_h,
                window_w,
            )
        )
        feature = self.lepe_conv(feature)
        feature = (
            feature.flatten(2)
            .transpose(1, 2)
            .reshape(
                batch_windows,
                num_tokens,
                self.num_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return feature

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            batch_windows,
            num_tokens,
            channels,
        ) = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                batch_windows,
                num_tokens,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        sampling_grid = (
            self._make_sampling_grid(x)
        )
        sampled_k = self._deform_sample(
            k,
            sampling_grid,
        )
        sampled_v = self._deform_sample(
            v,
            sampling_grid,
        )

        deform_gate = (
            torch.sigmoid(
                self.deform_gate
            )
            .unsqueeze(0)
        )
        k = (
            k
            + deform_gate
            * (sampled_k - k)
        )
        v = (
            v
            + deform_gate
            * (sampled_v - v)
        )

        lepe = (
            self
            ._local_position_enhancement(x)
        )
        lepe_scale = (
            torch.tanh(
                self.lepe_scale
            )
            .unsqueeze(0)
        )
        v = (
            v
            + lepe_scale
            * lepe
        )

        self.last_deform_gate = (
            deform_gate.detach()
        )
        self.last_lepe_scale = (
            lepe_scale.detach()
        )

        q = q * self.scale
        attn = (
            q @ k.transpose(-2, -1)
        )

        window_h, window_w = (
            self.window_size
        )
        window_tokens = (
            window_h * window_w
        )

        relative_position_bias = (
            self.relative_position_bias_table[
                self.relative_position_index
                .reshape(-1)
            ]
        )
        relative_position_bias = (
            relative_position_bias
            .view(
                window_tokens,
                window_tokens,
                self.num_heads,
            )
            .permute(2, 0, 1)
            .contiguous()
        )

        attn = (
            attn
            + relative_position_bias
            .unsqueeze(0)
        )

        if mask is not None:
            num_windows = mask.shape[0]

            if (
                batch_windows
                % num_windows
                != 0
            ):
                raise ValueError(
                    "Attention batch size must be divisible "
                    "by mask window count."
                )

            attn = attn.view(
                batch_windows
                // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attn = (
                attn
                + mask
                .unsqueeze(0)
                .unsqueeze(2)
            )
            attn = attn.view(
                -1,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

        attn_dtype = attn.dtype
        attn = (
            self.softmax(
                attn.float()
            )
            .to(attn_dtype)
        )
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(
                batch_windows,
                num_tokens,
                channels,
            )
        )
        x = self.proj(x)
        x = self.proj_drop(x)

        refine_scale = torch.tanh(
            self.output_refine_scale
        )
        x = (
            x
            + refine_scale
            * self.output_refine(x)
        )
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"window_size={self.window_size}, "
            f"num_heads={self.num_heads}, "
            f"offset_scale={self.offset_scale}"
        )

    def flops(
        self,
        num_tokens: int,
    ) -> int:
        window_h, window_w = (
            self.window_size
        )

        flops = 0
        flops += (
            num_tokens
            * self.dim
            * 3
            * self.dim
        )
        flops += (
            self.num_heads
            * num_tokens
            * self.head_dim
            * num_tokens
        )
        flops += (
            self.num_heads
            * num_tokens
            * num_tokens
            * self.head_dim
        )
        flops += (
            num_tokens
            * self.dim
            * self.dim
        )
        flops += (
            window_h
            * window_w
            * self.dim
            * 9
        )
        flops += (
            window_h
            * window_w
            * self.dim
            * 2
        )
        flops += (
            window_h
            * window_w
            * self.dim
            * 9
        )
        flops += (
            2
            * num_tokens
            * self.dim
            * self.dim
        )
        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: Size2T = 7,
        shift_size: Size2T = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: NormLayer = nn.LayerNorm,
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.input_resolution = tuple(
            input_resolution
        )
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        input_h, input_w = (
            self.input_resolution
        )
        window_h, window_w = (
            to_2tuple(window_size)
        )
        shift_h, shift_w = (
            to_2tuple(shift_size)
        )

        if input_h <= window_h:
            window_h = input_h
            shift_h = 0
        if input_w <= window_w:
            window_w = input_w
            shift_w = 0

        if not (
            0 <= shift_h < window_h
            and 0 <= shift_w < window_w
        ):
            raise ValueError(
                "Each shift dimension must satisfy "
                "0 <= shift < window."
            )

        if (
            input_h % window_h != 0
            or input_w % window_w != 0
        ):
            raise ValueError(
                f"Input resolution {self.input_resolution} "
                "must be divisible by effective window size "
                f"{(window_h, window_w)}."
            )

        self.window_size = (
            window_h,
            window_w,
        )
        self.shift_size = (
            shift_h,
            shift_w,
        )

        self.norm1 = norm_layer(dim)
        self.attn = (
            DeformableWindowAttention(
                dim=dim,
                window_size=self.window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                offset_scale=offset_scale,
            )
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(
            dim * mlp_ratio
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        attention_mask = (
            self._create_attention_mask()
        )
        self.register_buffer(
            "attn_mask",
            attention_mask,
            persistent=False,
        )

    def _create_attention_mask(
        self,
    ) -> Optional[torch.Tensor]:
        shift_h, shift_w = (
            self.shift_size
        )

        if (
            shift_h == 0
            and shift_w == 0
        ):
            return None

        height, width = (
            self.input_resolution
        )
        window_h, window_w = (
            self.window_size
        )

        img_mask = torch.zeros(
            (1, height, width, 1)
        )

        h_slices = (
            slice(0, -window_h),
            slice(-window_h, -shift_h),
            slice(-shift_h, None),
        )
        w_slices = (
            slice(0, -window_w),
            slice(-window_w, -shift_w),
            slice(-shift_w, None),
        )

        region_id = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[
                    :,
                    h_slice,
                    w_slice,
                    :,
                ] = region_id
                region_id += 1

        mask_windows = window_partition(
            img_mask,
            self.window_size,
        )
        mask_windows = (
            mask_windows.view(
                -1,
                window_h * window_w,
            )
        )

        attention_mask = (
            mask_windows.unsqueeze(1)
            - mask_windows.unsqueeze(2)
        )
        attention_mask = (
            attention_mask.masked_fill(
                attention_mask != 0,
                -100.0,
            )
        )
        attention_mask = (
            attention_mask.masked_fill(
                attention_mask == 0,
                0.0,
            )
        )
        return attention_mask

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        height, width = (
            self.input_resolution
        )
        batch_size, length, channels = (
            x.shape
        )

        if length != height * width:
            raise ValueError(
                f"Input token length {length} does not match "
                f"resolution {height}x{width}."
            )

        shortcut = x
        x = self.norm1(x)
        x = x.view(
            batch_size,
            height,
            width,
            channels,
        )

        shift_h, shift_w = (
            self.shift_size
        )

        if (
            shift_h > 0
            or shift_w > 0
        ):
            shifted_x = torch.roll(
                x,
                shifts=(
                    -shift_h,
                    -shift_w,
                ),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        window_h, window_w = (
            self.window_size
        )
        x_windows = window_partition(
            shifted_x,
            self.window_size,
        )
        x_windows = x_windows.view(
            -1,
            window_h * window_w,
            channels,
        )

        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask,
        )

        attn_windows = (
            attn_windows.view(
                -1,
                window_h,
                window_w,
                channels,
            )
        )
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            height,
            width,
        )

        if (
            shift_h > 0
            or shift_w > 0
        ):
            x = torch.roll(
                shifted_x,
                shifts=(
                    shift_h,
                    shift_w,
                ),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(
            batch_size,
            height * width,
            channels,
        )

        x = (
            shortcut
            + self.drop_path(x)
        )
        x = (
            x
            + self.drop_path(
                self.mlp(
                    self.norm2(x)
                )
            )
        )
        return x

    def flops(self) -> int:
        height, width = (
            self.input_resolution
        )
        window_h, window_w = (
            self.window_size
        )
        num_window_tokens = (
            window_h * window_w
        )
        num_windows = (
            (height * width)
            // num_window_tokens
        )

        flops = 0
        flops += (
            self.dim
            * height
            * width
        )
        flops += (
            num_windows
            * self.attn.flops(
                num_window_tokens
            )
        )
        flops += int(
            2
            * height
            * width
            * self.dim
            * self.dim
            * self.mlp_ratio
        )
        flops += (
            self.dim
            * height
            * width
        )
        return flops


class PatchMerging(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: NormLayer = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.input_resolution = tuple(
            input_resolution
        )
        self.dim = dim
        self.norm = norm_layer(
            4 * dim
        )
        self.reduction = nn.Linear(
            4 * dim,
            2 * dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        height, width = (
            self.input_resolution
        )
        (
            batch_size,
            length,
            channels,
        ) = x.shape

        if length != height * width:
            raise ValueError(
                f"Input token length {length} does not match "
                f"resolution {height}x{width}."
            )

        if (
            height % 2 != 0
            or width % 2 != 0
        ):
            raise ValueError(
                "PatchMerging requires an even resolution, "
                f"got ({height}, {width})."
            )

        x = x.view(
            batch_size,
            height,
            width,
            channels,
        )

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat(
            (x0, x1, x2, x3),
            dim=-1,
        )
        x = x.view(
            batch_size,
            -1,
            4 * channels,
        )

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def flops(self) -> int:
        height, width = (
            self.input_resolution
        )
        flops = (
            height
            * width
            * self.dim
        )
        flops += (
            (height // 2)
            * (width // 2)
            * 4
            * self.dim
            * 2
            * self.dim
        )
        return flops


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: Size2T,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[
            float,
            Sequence[float],
        ] = 0.0,
        norm_layer: NormLayer = nn.LayerNorm,
        downsample: Optional[type[nn.Module]] = None,
        use_checkpoint: bool = False,
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.input_resolution = tuple(
            input_resolution
        )
        self.depth = depth
        self.use_checkpoint = (
            use_checkpoint
        )

        window_h, window_w = (
            to_2tuple(window_size)
        )
        shift_size = (
            window_h // 2,
            window_w // 2,
        )

        if isinstance(
            drop_path,
            Sequence,
        ):
            if len(drop_path) != depth:
                raise ValueError(
                    "drop_path sequence length must equal stage depth."
                )
            drop_path_values = list(
                drop_path
            )
        else:
            drop_path_values = [
                float(drop_path)
            ] * depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=(
                        self.input_resolution
                    ),
                    num_heads=num_heads,
                    window_size=(
                        window_h,
                        window_w,
                    ),
                    shift_size=(
                        (0, 0)
                        if block_index % 2 == 0
                        else shift_size
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path_values[
                            block_index
                        ]
                    ),
                    norm_layer=norm_layer,
                    offset_scale=(
                        offset_scale
                    ),
                )
                for block_index
                in range(depth)
            ]
        )

        self.downsample = (
            downsample(
                self.input_resolution,
                dim=dim,
                norm_layer=norm_layer,
            )
            if downsample is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            if (
                self.use_checkpoint
                and self.training
            ):
                x = (
                    checkpoint.checkpoint(
                        block,
                        x,
                        use_reentrant=False,
                    )
                )
            else:
                x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def flops(self) -> int:
        flops = sum(
            block.flops()
            for block in self.blocks
        )
        if self.downsample is not None:
            flops += (
                self.downsample.flops()
            )
        return flops


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Size2T = 224,
        patch_size: Size2T = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[
            NormLayer
        ] = None,
    ) -> None:
        super().__init__()

        self.img_size = to_2tuple(
            img_size
        )
        self.patch_size = to_2tuple(
            patch_size
        )

        if (
            self.img_size[0]
            % self.patch_size[0]
            != 0
            or self.img_size[1]
            % self.patch_size[1]
            != 0
        ):
            raise ValueError(
                f"Image size {self.img_size} must be divisible "
                f"by patch size {self.patch_size}."
            )

        self.patches_resolution = (
            self.img_size[0]
            // self.patch_size[0],
            self.img_size[1]
            // self.patch_size[1],
        )
        self.num_patches = (
            self.patches_resolution[0]
            * self.patches_resolution[1]
        )

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = (
            norm_layer(embed_dim)
            if norm_layer is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width = (
            x.shape
        )

        if (
            height,
            width,
        ) != self.img_size:
            raise ValueError(
                f"Input image size ({height}, {width}) does not "
                f"match configured size {self.img_size}."
            )

        x = self.proj(x)
        x = (
            x.flatten(2)
            .transpose(1, 2)
        )

        if self.norm is not None:
            x = self.norm(x)

        return x

    def flops(self) -> int:
        output_h, output_w = (
            self.patches_resolution
        )
        kernel_area = (
            self.patch_size[0]
            * self.patch_size[1]
        )

        flops = (
            output_h
            * output_w
            * self.embed_dim
            * self.in_chans
            * kernel_area
        )

        if self.norm is not None:
            flops += (
                output_h
                * output_w
                * self.embed_dim
            )

        return flops


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size: Size2T = 224,
        patch_size: Size2T = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Sequence[int] = (
            2,
            2,
            6,
            2,
        ),
        num_heads: Sequence[int] = (
            3,
            6,
            12,
            24,
        ),
        window_size: Size2T = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[
            float
        ] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: NormLayer = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        deformable_offset_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        if len(depths) != len(num_heads):
            raise ValueError(
                "depths and num_heads must have the same length."
            )

        self.img_size = to_2tuple(
            img_size
        )
        self.patch_size = to_2tuple(
            patch_size
        )
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.num_features = int(
            embed_dim
            * 2
            ** (
                self.num_layers - 1
            )
        )

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=(
                norm_layer
                if patch_norm
                else None
            ),
        )

        num_patches = (
            self.patch_embed
            .num_patches
        )
        self.patches_resolution = (
            self.patch_embed
            .patches_resolution
        )

        minimum_downsample_factor = (
            2
            ** (
                self.num_layers - 1
            )
        )
        if (
            self.patches_resolution[0]
            % minimum_downsample_factor
            != 0
            or self.patches_resolution[1]
            % minimum_downsample_factor
            != 0
        ):
            raise ValueError(
                f"Patch resolution {self.patches_resolution} must "
                f"be divisible by {minimum_downsample_factor} "
                f"for {self.num_layers} stages."
            )

        if self.ape:
            self.absolute_pos_embed = (
                nn.Parameter(
                    torch.zeros(
                        1,
                        num_patches,
                        embed_dim,
                    )
                )
            )
            trunc_normal_(
                self.absolute_pos_embed,
                std=0.02,
            )
        else:
            self.absolute_pos_embed = (
                None
            )

        self.pos_drop = nn.Dropout(
            p=drop_rate
        )

        total_depth = sum(depths)
        drop_path_values = (
            torch.linspace(
                0,
                drop_path_rate,
                total_depth,
            )
            .tolist()
        )

        self.layers = (
            nn.ModuleList()
        )
        depth_offset = 0

        for layer_index in range(
            self.num_layers
        ):
            layer_dim = int(
                embed_dim
                * 2
                ** layer_index
            )
            layer_resolution = (
                self.patches_resolution[0]
                // (
                    2
                    ** layer_index
                ),
                self.patches_resolution[1]
                // (
                    2
                    ** layer_index
                ),
            )
            layer_depth = depths[
                layer_index
            ]

            layer = BasicLayer(
                dim=layer_dim,
                input_resolution=(
                    layer_resolution
                ),
                depth=layer_depth,
                num_heads=(
                    num_heads[
                        layer_index
                    ]
                ),
                window_size=(
                    window_size
                ),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=(
                    attn_drop_rate
                ),
                drop_path=(
                    drop_path_values[
                        depth_offset:
                        depth_offset
                        + layer_depth
                    ]
                ),
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging
                    if (
                        layer_index
                        < self.num_layers
                        - 1
                    )
                    else None
                ),
                use_checkpoint=(
                    use_checkpoint
                ),
                offset_scale=(
                    deformable_offset_scale
                ),
            )

            self.layers.append(layer)
            depth_offset += (
                layer_depth
            )

        self.norm = norm_layer(
            self.num_features
        )
        self.avgpool = (
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = (
            nn.Linear(
                self.num_features,
                num_classes,
            )
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(
            module,
            nn.Linear,
        ):
            trunc_normal_(
                module.weight,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0,
                )
        elif isinstance(
            module,
            nn.LayerNorm,
        ):
            nn.init.constant_(
                module.bias,
                0,
            )
            nn.init.constant_(
                module.weight,
                1.0,
            )

    @torch.jit.ignore
    def no_weight_decay(
        self,
    ) -> set[str]:
        return {
            "absolute_pos_embed"
        }

    @torch.jit.ignore
    def no_weight_decay_keywords(
        self,
    ) -> set[str]:
        return {
            "relative_position_bias_table"
        }

    def forward_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.patch_embed(x)

        if (
            self.absolute_pos_embed
            is not None
        ):
            x = (
                x
                + self.absolute_pos_embed
            )

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.forward_tokens(x)
        x = self.avgpool(
            x.transpose(1, 2)
        )
        x = torch.flatten(x, 1)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self) -> int:
        flops = (
            self.patch_embed.flops()
        )
        for layer in self.layers:
            flops += layer.flops()

        final_h = (
            self.patches_resolution[0]
            // (
                2
                ** (
                    self.num_layers - 1
                )
            )
        )
        final_w = (
            self.patches_resolution[1]
            // (
                2
                ** (
                    self.num_layers - 1
                )
            )
        )

        flops += (
            self.num_features
            * final_h
            * final_w
        )
        flops += (
            self.num_features
            * self.num_classes
        )
        return flops


def build_dwmm_swin_base(
    num_classes: int = 0,
    img_size: int = 224,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.10,
    use_checkpoint: bool = False,
    deformable_offset_scale: float = 1.0,
) -> SwinTransformer:
    return SwinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=SWIN_B_EMBED_DIM,
        depths=SWIN_B_DEPTHS,
        num_heads=(
            SWIN_B_NUM_HEADS
        ),
        window_size=(
            SWIN_B_WINDOW_SIZE
        ),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=drop_rate,
        attn_drop_rate=(
            attn_drop_rate
        ),
        drop_path_rate=(
            drop_path_rate
        ),
        ape=False,
        patch_norm=True,
        use_checkpoint=(
            use_checkpoint
        ),
        deformable_offset_scale=(
            deformable_offset_scale
        ),
    )


def _extract_checkpoint_state(
    checkpoint,
):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    for key in (
        "model",
        "state_dict",
        "model_ema",
    ):
        value = checkpoint.get(key)
        if isinstance(
            value,
            dict,
        ):
            return value

    return checkpoint


def _extract_mae_dwmm_encoder_state(
    checkpoint,
):
    state = (
        _extract_checkpoint_state(
            checkpoint
        )
    )
    encoder = {}

    has_backbone_prefix = any(
        key.startswith(
            "backbone."
        )
        or key.startswith(
            "module.backbone."
        )
        or key.startswith(
            "model.backbone."
        )
        for key in state
    )

    for key, value in state.items():
        clean = key

        if clean.startswith(
            "module."
        ):
            clean = clean[
                len("module.") :
            ]
        if clean.startswith(
            "model."
        ):
            clean = clean[
                len("model.") :
            ]

        if has_backbone_prefix:
            if not clean.startswith(
                "backbone."
            ):
                continue
            clean = clean[
                len("backbone.") :
            ]

        if (
            clean.startswith(
                "classifier."
            )
            or clean.startswith(
                "decoder."
            )
            or clean.startswith(
                "decoder_"
            )
            or clean == "mask_token"
        ):
            continue

        encoder[clean] = value

    return encoder


def _parameter_coverage(
    state_dict,
    model,
):
    model_state = (
        model.state_dict()
    )
    total = 0
    matched = 0
    matched_keys = set()

    for key, tensor in (
        model_state.items()
    ):
        total += tensor.numel()
        other = state_dict.get(key)

        if (
            other is not None
            and tuple(other.shape)
            == tuple(tensor.shape)
        ):
            matched += (
                tensor.numel()
            )
            matched_keys.add(key)

    return (
        matched
        / max(1, total),
        matched_keys,
    )


class SwinMAEDWMMPretrain(
    nn.Module
):
    """
    Swin-B DWMM masked-autoencoder-style pretraining.

    224x224
      -> 4x4 patch embedding (56x56)
      -> grouped 75% masking
      -> DW-MSA / DSW-MSA Swin-B hierarchy
      -> 7x7x1024 latent
      -> lightweight decoder
      -> 56x56 RGB patch-vector prediction
    """

    def __init__(
        self,
        model_name: str = (
            SWIN_MAE_DWMM_MODEL_NAME
        ),
        img_size: int = 224,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        mask_window: int = 4,
        decoder_dim: int = 256,
        norm_pix_loss: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.10,
        use_checkpoint: bool = False,
        deformable_offset_scale: float = 1.0,
    ):
        super().__init__()

        if (
            model_name
            != SWIN_MAE_DWMM_MODEL_NAME
        ):
            raise ValueError(
                f"Only {SWIN_MAE_DWMM_MODEL_NAME} "
                "is configured."
            )

        if (
            img_size != 224
            or patch_size != 4
        ):
            raise ValueError(
                "Configured for img_size=224 and patch_size=4."
            )

        self.model_name = model_name
        self.img_size = int(
            img_size
        )
        self.patch_size = int(
            patch_size
        )
        self.mask_ratio = float(
            mask_ratio
        )
        self.mask_window = int(
            mask_window
        )
        self.norm_pix_loss = bool(
            norm_pix_loss
        )

        self.backbone = (
            build_dwmm_swin_base(
                num_classes=0,
                img_size=img_size,
                drop_rate=drop_rate,
                attn_drop_rate=(
                    attn_drop_rate
                ),
                drop_path_rate=(
                    drop_path_rate
                ),
                use_checkpoint=(
                    use_checkpoint
                ),
                deformable_offset_scale=(
                    deformable_offset_scale
                ),
            )
        )

        self.embed_dim = int(
            self.backbone.embed_dim
        )
        self.num_features = int(
            self.backbone.num_features
        )
        self.patch_grid = tuple(
            int(value)
            for value in (
                self.backbone
                .patch_embed
                .patches_resolution
            )
        )

        if (
            self.patch_grid[0]
            % self.mask_window
            != 0
            or self.patch_grid[1]
            % self.mask_window
            != 0
        ):
            raise ValueError(
                "mask_window must divide the patch grid exactly: "
                f"grid={self.patch_grid}, "
                f"window={self.mask_window}"
            )

        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.embed_dim,
            )
        )
        nn.init.normal_(
            self.mask_token,
            std=0.02,
        )

        # 7x7 -> 14 -> 28 -> 56.
        self.decoder = nn.Sequential(
            nn.Conv2d(
                self.num_features,
                decoder_dim * 4,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                decoder_dim * 4,
                decoder_dim * 2,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                decoder_dim * 2,
                decoder_dim,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                decoder_dim,
                decoder_dim // 2,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),
            nn.Conv2d(
                decoder_dim // 2,
                patch_size
                * patch_size
                * 3,
                kernel_size=1,
            ),
        )

    def patchify(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        p = self.patch_size
        (
            batch,
            channels,
            height,
            width,
        ) = images.shape

        if (
            height != self.img_size
            or width != self.img_size
        ):
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size}, "
                f"got {height}x{width}."
            )

        grid_h = height // p
        grid_w = width // p

        x = images.reshape(
            batch,
            channels,
            grid_h,
            p,
            grid_w,
            p,
        )
        x = (
            x.permute(
                0,
                2,
                4,
                3,
                5,
                1,
            )
            .contiguous()
        )
        return x.reshape(
            batch,
            grid_h * grid_w,
            p * p * channels,
        )

    def unpatchify(
        self,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        p = self.patch_size
        grid_h, grid_w = (
            self.patch_grid
        )
        batch, count, dim = (
            patches.shape
        )

        expected_dim = (
            p * p * 3
        )
        if (
            count
            != grid_h * grid_w
            or dim != expected_dim
        ):
            raise ValueError(
                "Unexpected patch tensor for unpatchify: "
                f"{tuple(patches.shape)}"
            )

        x = patches.view(
            batch,
            grid_h,
            grid_w,
            p,
            p,
            3,
        )
        x = (
            x.permute(
                0,
                5,
                1,
                3,
                2,
                4,
            )
            .contiguous()
        )
        return x.view(
            batch,
            3,
            grid_h * p,
            grid_w * p,
        )

    def window_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        grid_h, grid_w = (
            self.patch_grid
        )
        group = self.mask_window

        coarse_h = (
            grid_h // group
        )
        coarse_w = (
            grid_w // group
        )
        coarse_n = (
            coarse_h * coarse_w
        )

        num_mask = int(
            round(
                coarse_n
                * self.mask_ratio
            )
        )
        num_mask = min(
            max(num_mask, 1),
            coarse_n - 1,
        )

        noise = torch.rand(
            batch_size,
            coarse_n,
            device=device,
        )
        ids = torch.argsort(
            noise,
            dim=1,
        )

        coarse_mask = torch.zeros(
            batch_size,
            coarse_n,
            device=device,
            dtype=torch.float32,
        )
        coarse_mask.scatter_(
            1,
            ids[:, :num_mask],
            1.0,
        )
        coarse_mask = (
            coarse_mask.reshape(
                batch_size,
                coarse_h,
                coarse_w,
            )
        )

        mask = (
            coarse_mask
            .repeat_interleave(
                group,
                dim=1,
            )
            .repeat_interleave(
                group,
                dim=2,
            )
        )
        return mask.reshape(
            batch_size,
            grid_h * grid_w,
        )

    def forward_encoder(
        self,
        images: torch.Tensor,
    ):
        x = (
            self.backbone
            .patch_embed(images)
        )

        batch, tokens, channels = (
            x.shape
        )
        expected_tokens = (
            self.patch_grid[0]
            * self.patch_grid[1]
        )

        if (
            tokens != expected_tokens
            or channels
            != self.embed_dim
        ):
            raise RuntimeError(
                "Unexpected patch embedding shape: "
                f"{tuple(x.shape)}"
            )

        if (
            self.backbone
            .absolute_pos_embed
            is not None
        ):
            x = (
                x
                + self.backbone
                .absolute_pos_embed
            )

        mask = self.window_mask(
            batch_size=batch,
            device=x.device,
        )

        mask_token = (
            self.mask_token
            .to(
                dtype=x.dtype,
                device=x.device,
            )
            .expand(
                batch,
                tokens,
                -1,
            )
        )

        x = torch.where(
            mask.unsqueeze(-1).bool(),
            mask_token,
            x,
        )
        x = (
            self.backbone
            .pos_drop(x)
        )

        for layer in (
            self.backbone.layers
        ):
            x = layer(x)

        x = self.backbone.norm(x)
        return x, mask

    def forward_decoder(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 3:
            raise RuntimeError(
                "Expected BNC latent, got "
                f"{tuple(latent.shape)}"
            )

        batch, tokens, channels = (
            latent.shape
        )

        final_grid_h = (
            self.patch_grid[0]
            // 8
        )
        final_grid_w = (
            self.patch_grid[1]
            // 8
        )

        if (
            tokens
            != final_grid_h
            * final_grid_w
            or channels
            != self.num_features
        ):
            raise RuntimeError(
                "Unexpected final DWMM latent: "
                f"{tuple(latent.shape)}"
            )

        x = (
            latent
            .transpose(1, 2)
            .contiguous()
            .view(
                batch,
                channels,
                final_grid_h,
                final_grid_w,
            )
        )
        x = self.decoder(x)

        grid_h, grid_w = (
            self.patch_grid
        )
        if x.shape[-2:] != (
            grid_h,
            grid_w,
        ):
            x = F.interpolate(
                x,
                size=(
                    grid_h,
                    grid_w,
                ),
                mode="bilinear",
                align_corners=False,
            )

        x = (
            x.permute(
                0,
                2,
                3,
                1,
            )
            .contiguous()
            .reshape(
                batch,
                grid_h * grid_w,
                self.patch_size
                * self.patch_size
                * 3,
            )
        )
        return x

    def forward_loss(
        self,
        images: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self.patchify(
            images
        )

        if self.norm_pix_loss:
            mean = target.mean(
                dim=-1,
                keepdim=True,
            )
            var = target.var(
                dim=-1,
                keepdim=True,
                unbiased=False,
            )
            target = (
                (target - mean)
                / torch.sqrt(
                    var + 1e-6
                )
            )

        loss = (
            (pred - target)
            .pow(2)
            .mean(dim=-1)
        )
        return (
            (loss * mask).sum()
            / mask.sum()
            .clamp_min(1.0)
        )

    def forward(
        self,
        images: torch.Tensor,
    ):
        latent, mask = (
            self.forward_encoder(
                images
            )
        )
        pred = self.forward_decoder(
            latent
        )
        loss = self.forward_loss(
            images,
            pred,
            mask,
        )
        return loss, pred, mask

    @torch.no_grad()
    def reconstruct_for_visualization(
        self,
        images: torch.Tensor,
    ):
        """
        Return RGB tensors in [0, 1]:
            original, masked_input, reconstruction, mask

        If norm_pix_loss=True, prediction is inverted with each target
        patch's mean/std for visualization only. Training loss itself is
        unchanged.
        """
        latent, mask = (
            self.forward_encoder(
                images
            )
        )
        pred = self.forward_decoder(
            latent
        )

        target_patches = (
            self.patchify(images)
        )

        if self.norm_pix_loss:
            patch_mean = (
                target_patches.mean(
                    dim=-1,
                    keepdim=True,
                )
            )
            patch_var = (
                target_patches.var(
                    dim=-1,
                    keepdim=True,
                    unbiased=False,
                )
            )
            pred_for_image = (
                pred
                * torch.sqrt(
                    patch_var + 1e-6
                )
                + patch_mean
            )
        else:
            pred_for_image = pred

        reconstructed_norm = (
            self.unpatchify(
                pred_for_image
            )
        )

        grid_h, grid_w = (
            self.patch_grid
        )
        pixel_mask = (
            mask.view(
                images.shape[0],
                1,
                grid_h,
                grid_w,
            )
            .repeat_interleave(
                self.patch_size,
                dim=2,
            )
            .repeat_interleave(
                self.patch_size,
                dim=3,
            )
        )

        masked_norm = (
            images
            * (
                1.0
                - pixel_mask
            )
        )

        mean = images.new_tensor(
            IMAGENET_MEAN
        ).view(
            1,
            3,
            1,
            1,
        )
        std = images.new_tensor(
            IMAGENET_STD
        ).view(
            1,
            3,
            1,
            1,
        )

        original_rgb = (
            images * std + mean
        ).clamp(
            0.0,
            1.0,
        )
        masked_rgb = (
            masked_norm
            * std
            + mean
        ).clamp(
            0.0,
            1.0,
        )
        reconstructed_rgb = (
            reconstructed_norm
            * std
            + mean
        ).clamp(
            0.0,
            1.0,
        )

        return (
            original_rgb,
            masked_rgb,
            reconstructed_rgb,
            mask,
        )


class SwinMAEDWMMClassifier(
    nn.Module
):
    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 2,
        model_name: str = (
            SWIN_MAE_DWMM_MODEL_NAME
        ),
        drop_rate: float = 0.10,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.10,
        freeze_backbone: bool = False,
        use_checkpoint: bool = False,
        deformable_offset_scale: float = 1.0,
    ):
        super().__init__()

        if (
            model_name
            != SWIN_MAE_DWMM_MODEL_NAME
        ):
            raise ValueError(
                f"Only {SWIN_MAE_DWMM_MODEL_NAME} "
                "is supported."
            )

        if not pretrained_path:
            raise ValueError(
                "Pass the Swin-MAE+DWMM checkpoint with --pretrained."
            )

        pretrained_path = (
            os.path.expanduser(
                str(pretrained_path)
            )
        )
        if not os.path.isfile(
            pretrained_path
        ):
            raise FileNotFoundError(
                "\nSwin-MAE+DWMM checkpoint not found:\n"
                f"{pretrained_path}\n"
            )

        self.model_name = (
            model_name
        )
        self.backbone = (
            build_dwmm_swin_base(
                num_classes=0,
                img_size=224,
                drop_rate=drop_rate,
                attn_drop_rate=(
                    attn_drop_rate
                ),
                drop_path_rate=(
                    drop_path_rate
                ),
                use_checkpoint=(
                    use_checkpoint
                ),
                deformable_offset_scale=(
                    deformable_offset_scale
                ),
            )
        )
        self.num_features = int(
            self.backbone.num_features
        )

        checkpoint_data = (
            torch.load(
                pretrained_path,
                map_location="cpu",
                weights_only=False,
            )
        )
        encoder_state = (
            _extract_mae_dwmm_encoder_state(
                checkpoint_data
            )
        )

        (
            coverage,
            matched_keys,
        ) = _parameter_coverage(
            encoder_state,
            self.backbone,
        )

        print(
            "=" * 72,
            flush=True,
        )
        print(
            "[Swin-MAE+DWMM] Loading MAE-pretrained "
            "Deformable Swin-B encoder",
            flush=True,
        )
        print(
            "[Swin-MAE+DWMM] Checkpoint: "
            f"{pretrained_path}",
            flush=True,
        )
        print(
            "[Swin-MAE+DWMM] Encoder parameter coverage: "
            f"{coverage * 100.0:.2f}%",
            flush=True,
        )

        if coverage < 0.95:
            missing_preview = [
                key
                for key
                in self.backbone
                .state_dict()
                .keys()
                if key
                not in matched_keys
            ][:20]

            raise RuntimeError(
                "Swin-MAE+DWMM encoder checkpoint coverage too low. "
                f"Coverage={coverage * 100.0:.2f}%. "
                f"First unmatched keys: {missing_preview}"
            )

        msg = (
            self.backbone
            .load_state_dict(
                encoder_state,
                strict=False,
            )
        )

        print(
            "[Swin-MAE+DWMM] Missing keys: "
            f"{len(msg.missing_keys)}",
            flush=True,
        )
        print(
            "[Swin-MAE+DWMM] Unexpected keys: "
            f"{len(msg.unexpected_keys)}",
            flush=True,
        )
        print(
            "=" * 72,
            flush=True,
        )

        if freeze_backbone:
            for parameter in (
                self.backbone
                .parameters()
            ):
                parameter.requires_grad = (
                    False
                )

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=float(
                    drop_rate
                )
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

    def forward_features(
        self,
        images: torch.Tensor,
    ):
        return (
            self.backbone
            .forward_features(images)
        )

    def forward(
        self,
        images: torch.Tensor,
    ):
        return self.classifier(
            self.forward_features(
                images
            )
        )


def build_swin_mae_dwmm(
    config=None,
    num_classes: Optional[int] = None,
    model_name: str = (
        SWIN_MAE_DWMM_MODEL_NAME
    ),
    drop_rate: float = 0.10,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.10,
    freeze_backbone: bool = False,
    pretrained_path: Optional[str] = None,
    use_checkpoint: bool = False,
    deformable_offset_scale: float = 1.0,
):
    if (
        config is not None
        and hasattr(
            config,
            "MODEL",
        )
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
            "ATTN_DROP_RATE",
        ):
            attn_drop_rate = float(
                config.MODEL
                .ATTN_DROP_RATE
            )

        if hasattr(
            config.MODEL,
            "DROP_PATH_RATE",
        ):
            drop_path_rate = float(
                config.MODEL
                .DROP_PATH_RATE
            )

        if (
            hasattr(
                config,
                "TRAIN",
            )
            and hasattr(
                config.TRAIN,
                "USE_CHECKPOINT",
            )
        ):
            use_checkpoint = bool(
                config.TRAIN
                .USE_CHECKPOINT
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

    return SwinMAEDWMMClassifier(
        pretrained_path=(
            pretrained_path
        ),
        num_classes=num_classes,
        model_name=model_name,
        drop_rate=drop_rate,
        attn_drop_rate=(
            attn_drop_rate
        ),
        drop_path_rate=(
            drop_path_rate
        ),
        freeze_backbone=(
            freeze_backbone
        ),
        use_checkpoint=(
            use_checkpoint
        ),
        deformable_offset_scale=(
            deformable_offset_scale
        ),
    )


def build_swin(
    config,
) -> SwinTransformer:
    """
    Compatibility builder for this DWMM Swin implementation.
    """
    attn_drop_rate = getattr(
        config.MODEL,
        "ATTN_DROP_RATE",
        0.0,
    )
    deformable_offset_scale = getattr(
        config.MODEL.SWIN,
        "DEFORMABLE_OFFSET_SCALE",
        1.0,
    )

    return SwinTransformer(
        img_size=(
            config.DATA.IMG_SIZE
        ),
        patch_size=(
            config.MODEL.SWIN
            .PATCH_SIZE
        ),
        in_chans=(
            config.MODEL.SWIN
            .IN_CHANS
        ),
        num_classes=(
            config.MODEL
            .NUM_CLASSES
        ),
        embed_dim=(
            config.MODEL.SWIN
            .EMBED_DIM
        ),
        depths=tuple(
            config.MODEL.SWIN
            .DEPTHS
        ),
        num_heads=tuple(
            config.MODEL.SWIN
            .NUM_HEADS
        ),
        window_size=(
            config.MODEL.SWIN
            .WINDOW_SIZE
        ),
        mlp_ratio=(
            config.MODEL.SWIN
            .MLP_RATIO
        ),
        qkv_bias=(
            config.MODEL.SWIN
            .QKV_BIAS
        ),
        qk_scale=(
            config.MODEL.SWIN
            .QK_SCALE
        ),
        drop_rate=(
            config.MODEL
            .DROP_RATE
        ),
        attn_drop_rate=(
            attn_drop_rate
        ),
        drop_path_rate=(
            config.MODEL
            .DROP_PATH_RATE
        ),
        ape=(
            config.MODEL.SWIN.APE
        ),
        patch_norm=(
            config.MODEL.SWIN
            .PATCH_NORM
        ),
        use_checkpoint=(
            config.TRAIN
            .USE_CHECKPOINT
        ),
        deformable_offset_scale=(
            deformable_offset_scale
        ),
    )


# ======================================================================
# Swin-B + MAE + DWMM + Contrastive Learning
# ======================================================================

class DWMMCLProjectionMLP(nn.Module):
    """LayerNorm projection head for stable small-batch SSL."""
    def __init__(self, in_dim=1024, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DWMMCLFeatureDecoder(nn.Module):
    """Online-only predictor before the contrastive projector."""
    def __init__(self, dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SwinMAEDWMMCLPretrain(SwinMAEDWMMPretrain):
    """
    Swin-B + MAE + DWMM + CL.

    Online:
      masked image -> DWMM Swin-B -> MAE decoder -> L_MAE
                                  -> pooled latent -> predictor -> projector -> q

    Momentum:
      weak positive -> EMA DWMM Swin-B -> EMA projector -> k

    Negatives:
      MoCo-style queue

    Total:
      L = L_MAE + lambda_CL * L_CL
    """
    def __init__(
        self,
        model_name=SWIN_MAE_DWMM_MODEL_NAME,
        img_size=224,
        patch_size=4,
        mask_ratio=0.75,
        mask_window=4,
        decoder_dim=256,
        norm_pix_loss=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.10,
        use_checkpoint=False,
        deformable_offset_scale=1.0,
        projection_dim=256,
        projection_hidden_dim=2048,
        temperature=0.20,
        contrast_weight=0.10,
        queue_size=4096,
    ):
        super().__init__(
            model_name=model_name,
            img_size=img_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            mask_window=mask_window,
            decoder_dim=decoder_dim,
            norm_pix_loss=norm_pix_loss,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            deformable_offset_scale=deformable_offset_scale,
        )

        self.temperature = float(temperature)
        self.contrast_weight = float(contrast_weight)
        self.queue_size = int(queue_size)

        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.contrast_weight < 0:
            raise ValueError("contrast_weight must be >= 0")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be > 0")

        self.feature_decoder = DWMMCLFeatureDecoder(
            dim=self.num_features,
        )
        self.projector = DWMMCLProjectionMLP(
            in_dim=self.num_features,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
        )

        self.momentum_backbone = copy.deepcopy(self.backbone)
        for layer in self.momentum_backbone.layers:
            if hasattr(layer, "use_checkpoint"):
                layer.use_checkpoint = False

        self.momentum_projector = copy.deepcopy(self.projector)

        for p in self.momentum_backbone.parameters():
            p.requires_grad = False
        for p in self.momentum_projector.parameters():
            p.requires_grad = False

        queue = torch.randn(projection_dim, self.queue_size)
        queue = F.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long),
        )

        self.momentum_backbone.eval()
        self.momentum_projector.eval()

    def train(self, mode=True):
        super().train(mode)
        # Deterministic EMA target: no DropPath/dropout on the key branch.
        self.momentum_backbone.eval()
        self.momentum_projector.eval()
        return self

    @staticmethod
    def _pool_latent(latent):
        if latent.ndim == 3:
            return latent.mean(dim=1)
        if latent.ndim == 4:
            if latent.shape[-1] > latent.shape[1]:
                return latent.mean(dim=(1, 2))
            return latent.mean(dim=(2, 3))
        if latent.ndim == 2:
            return latent
        raise RuntimeError(
            f"Unexpected DWMM latent for CL: {tuple(latent.shape)}"
        )

    @torch.no_grad()
    def momentum_update(self, momentum):
        momentum = float(momentum)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0,1]")

        for online, target in zip(
            self.backbone.parameters(),
            self.momentum_backbone.parameters(),
        ):
            target.data.mul_(momentum).add_(
                online.data,
                alpha=1.0 - momentum,
            )

        for online, target in zip(
            self.projector.parameters(),
            self.momentum_projector.parameters(),
        ):
            target.data.mul_(momentum).add_(
                online.data,
                alpha=1.0 - momentum,
            )

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        keys = F.normalize(keys, dim=1)
        batch_size = int(keys.shape[0])
        ptr = int(self.queue_ptr.item())

        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size:].T)
            self.queue_ptr.zero_()
            return

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            remain = batch_size - first
            self.queue[:, :remain] = keys[first:].T

        self.queue_ptr[0] = (
            ptr + batch_size
        ) % self.queue_size

    def contrastive_loss(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        positive = torch.einsum(
            "nc,nc->n",
            (q, k),
        ).unsqueeze(1)

        negative = torch.einsum(
            "nc,ck->nk",
            (q, self.queue.detach().clone()),
        )

        logits = torch.cat(
            (positive, negative),
            dim=1,
        ) / self.temperature

        labels = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device,
        )
        loss_cl = F.cross_entropy(logits, labels)
        return loss_cl, q, k

    @torch.no_grad()
    def reconstruct_for_visualization(
        self,
        images,
    ):
        (
            original_rgb,
            _mean_filled_masked_rgb,
            reconstructed_rgb,
            mask,
        ) = super().reconstruct_for_visualization(
            images
        )

        grid_h, grid_w = self.patch_grid
        pixel_mask = (
            mask.view(
                images.shape[0],
                1,
                grid_h,
                grid_w,
            )
            .repeat_interleave(
                self.patch_size,
                dim=2,
            )
            .repeat_interleave(
                self.patch_size,
                dim=3,
            )
        )

        # Visualization only: use black missing regions so reconstruction
        # quality is easy to inspect. Training input/masking is unchanged.
        black_masked_rgb = (
            original_rgb
            * (
                1.0
                - pixel_mask
            )
        )

        return (
            original_rgb,
            black_masked_rgb,
            reconstructed_rgb,
            mask,
        )

    def forward(self, online_images, target_images):
        latent_online, mask = self.forward_encoder(
            online_images
        )

        pred_pixel = self.forward_decoder(
            latent_online
        )
        loss_mae = self.forward_loss(
            online_images,
            pred_pixel,
            mask,
        )

        pooled_online = self._pool_latent(
            latent_online
        )
        q = self.projector(
            self.feature_decoder(
                pooled_online
            )
        )

        with torch.no_grad():
            target_feature = (
                self.momentum_backbone
                .forward_features(
                    target_images
                )
            )
            k = self.momentum_projector(
                target_feature
            )

        loss_cl, q_norm, k_norm = self.contrastive_loss(
            q,
            k,
        )

        total_loss = (
            loss_mae
            + self.contrast_weight * loss_cl
        )

        with torch.no_grad():
            self.dequeue_and_enqueue(
                k_norm
            )

        return {
            "loss": total_loss,
            "loss_mae": loss_mae,
            "loss_cl": loss_cl,
            "mask_ratio_actual": mask.float().mean(),
            "q_norm": q_norm.norm(dim=1).mean(),
            "k_norm": k_norm.norm(dim=1).mean(),
        }


class SwinMAEDWMMCLClassifier(SwinMAEDWMMClassifier):
    """
    Fine-tuning uses only the online DWMM Swin-B encoder.
    Decoder, EMA encoder, projector and queue are discarded.
    """
    pass


def build_swin_mae_dwmm_cl(
    config=None,
    num_classes=None,
    model_name=SWIN_MAE_DWMM_MODEL_NAME,
    drop_rate=0.10,
    attn_drop_rate=0.0,
    drop_path_rate=0.10,
    freeze_backbone=False,
    pretrained_path=None,
    use_checkpoint=False,
    deformable_offset_scale=1.0,
):
    if config is not None and hasattr(config, "MODEL"):
        if num_classes is None:
            num_classes = int(config.MODEL.NUM_CLASSES)

        if hasattr(config.MODEL, "DROP_RATE"):
            drop_rate = float(config.MODEL.DROP_RATE)
        if hasattr(config.MODEL, "ATTN_DROP_RATE"):
            attn_drop_rate = float(config.MODEL.ATTN_DROP_RATE)
        if hasattr(config.MODEL, "DROP_PATH_RATE"):
            drop_path_rate = float(config.MODEL.DROP_PATH_RATE)

        if (
            hasattr(config, "TRAIN")
            and hasattr(config.TRAIN, "USE_CHECKPOINT")
        ):
            use_checkpoint = bool(
                config.TRAIN.USE_CHECKPOINT
            )

        if (
            pretrained_path is None
            and hasattr(config, "PRETRAINED")
            and config.PRETRAINED
        ):
            pretrained_path = str(config.PRETRAINED)

    if num_classes is None:
        num_classes = 2

    return SwinMAEDWMMCLClassifier(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        model_name=model_name,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
        use_checkpoint=use_checkpoint,
        deformable_offset_scale=deformable_offset_scale,
    )
