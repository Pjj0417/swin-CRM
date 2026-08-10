# # --------------------------------------------------------
# # 可移动窗口
# # Copyright (c) 2021 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ze Liu
# # Modified by Zhenda Xie
# # --------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows


# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops


# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         if self.shift_size > 0:
#             # calculate attention mask for SW-MSA
#             H, W = self.input_resolution
#             img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask)

#     def forward(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#         x = x.view(B, H * W, C)

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops


# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.

#     Args:
#         input_resolution (tuple[int]): Resolution of input feature.
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """

#     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)

#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

#         x = x.view(B, H, W, C)

#         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
#         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
#         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
#         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
#         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
#         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

#         x = self.norm(x)
#         x = self.reduction(x)

#         return x

#     def extra_repr(self) -> str:
#         return f"input_resolution={self.input_resolution}, dim={self.dim}"

#     def flops(self):
#         H, W = self.input_resolution
#         flops = H * W * self.dim
#         flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
#         return flops


# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self, dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  drop=drop, attn_drop=attn_drop,
#                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None

#     def forward(self, x):
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops


# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding

#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         if norm_layer is not None:
#             self.norm = norm_layer(embed_dim)
#         else:
#             self.norm = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#         if self.norm is not None:
#             x = self.norm(x)
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops


# class SwinTransformer(nn.Module):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030

#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, **kwargs):
#         super().__init__()

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans

#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.mlp_ratio = mlp_ratio

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution

#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
#                                                  patches_resolution[1] // (2 ** i_layer)),
#                                depth=depths[i_layer],
#                                num_heads=num_heads[i_layer],
#                                window_size=window_size,
#                                mlp_ratio=self.mlp_ratio,
#                                qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                norm_layer=norm_layer,
#                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                use_checkpoint=use_checkpoint)
#             self.layers.append(layer)

#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}

#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)

#         for layer in self.layers:
#             x = layer(x)

#         x = self.norm(x)  # B L C
#         x = self.avgpool(x.transpose(1, 2))  # B C 1
#         x = torch.flatten(x, 1)
#         return x

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x

#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops


# def build_swin(config):
#     model = SwinTransformer(
#         img_size=config.DATA.IMG_SIZE,
#         patch_size=config.MODEL.SWIN.PATCH_SIZE,
#         in_chans=config.MODEL.SWIN.IN_CHANS,
#         num_classes=config.MODEL.NUM_CLASSES,
#         embed_dim=config.MODEL.SWIN.EMBED_DIM,
#         depths=config.MODEL.SWIN.DEPTHS,
#         num_heads=config.MODEL.SWIN.NUM_HEADS,
#         window_size=config.MODEL.SWIN.WINDOW_SIZE,
#         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
#         qk_scale=config.MODEL.SWIN.QK_SCALE,
#         drop_rate=config.MODEL.DROP_RATE,
#         drop_path_rate=config.MODEL.DROP_PATH_RATE,
#         ape=config.MODEL.SWIN.APE,
#         patch_norm=config.MODEL.SWIN.PATCH_NORM,
#         use_checkpoint=config.TRAIN.USE_CHECKPOINT)

#     return model

# #的确更好，第一版
# """
# Deformable Swin Transformer implementation using DW-MSA and DSW-MSA.

# Based on:
# - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# - SimMIM / Microsoft reference implementation

# Improvements in this consolidated version:
# 1. Removes duplicated/commented code.
# 2. Uses torch.meshgrid(..., indexing="ij") for modern PyTorch compatibility.
# 3. Avoids mutable list defaults.
# 4. Adds clear shape and divisibility validation.
# 5. Supports tuple image/window sizes.
# 6. Uses checkpoint(..., use_reentrant=False) when enabled.
# 7. Provides configurable attention dropout in build_swin.
# """

# from typing import Callable, Optional, Sequence, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# Size2T = Union[int, Tuple[int, int]]
# NormLayer = Callable[[int], nn.Module]


# class Mlp(nn.Module):
#     """Feed-forward network used inside each Swin Transformer block."""

#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: Optional[int] = None,
#         out_features: Optional[int] = None,
#         act_layer: Callable[[], nn.Module] = nn.GELU,
#         drop: float = 0.0,
#     ) -> None:
#         super().__init__()

#         hidden_features = hidden_features or in_features
#         out_features = out_features or in_features

#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x


# def window_partition(x: torch.Tensor, window_size: Size2T) -> torch.Tensor:
#     """
#     Partition a feature map into non-overlapping windows.

#     Args:
#         x: Tensor with shape (B, H, W, C).
#         window_size: Integer or (window_height, window_width).

#     Returns:
#         Windows with shape:
#         (B * num_windows, window_height, window_width, C).
#     """
#     window_h, window_w = to_2tuple(window_size)
#     batch_size, height, width, channels = x.shape

#     if height % window_h != 0 or width % window_w != 0:
#         raise ValueError(
#             f"Feature map size ({height}, {width}) must be divisible by "
#             f"window size ({window_h}, {window_w})."
#         )

#     x = x.view(
#         batch_size,
#         height // window_h,
#         window_h,
#         width // window_w,
#         window_w,
#         channels,
#     )
#     windows = (
#         x.permute(0, 1, 3, 2, 4, 5)
#         .contiguous()
#         .view(-1, window_h, window_w, channels)
#     )
#     return windows


# def window_reverse(
#     windows: torch.Tensor,
#     window_size: Size2T,
#     height: int,
#     width: int,
# ) -> torch.Tensor:
#     """
#     Merge windows back into a feature map.

#     Args:
#         windows: Tensor with shape
#             (B * num_windows, window_height, window_width, C).
#         window_size: Integer or (window_height, window_width).
#         height: Output feature-map height.
#         width: Output feature-map width.

#     Returns:
#         Tensor with shape (B, H, W, C).
#     """
#     window_h, window_w = to_2tuple(window_size)

#     if height % window_h != 0 or width % window_w != 0:
#         raise ValueError(
#             f"Output size ({height}, {width}) must be divisible by "
#             f"window size ({window_h}, {window_w})."
#         )

#     windows_per_image = (height // window_h) * (width // window_w)

#     if windows.shape[0] % windows_per_image != 0:
#         raise ValueError(
#             "The number of windows is incompatible with the requested output size."
#         )

#     batch_size = windows.shape[0] // windows_per_image

#     x = windows.view(
#         batch_size,
#         height // window_h,
#         width // window_w,
#         window_h,
#         window_w,
#         -1,
#     )
#     x = (
#         x.permute(0, 1, 3, 2, 4, 5)
#         .contiguous()
#         .view(batch_size, height, width, -1)
#     )
#     return x


# class DeformableWindowAttention(nn.Module):
#     """
#     Deformable window-based multi-head self-attention.

#     When used without cyclic shift, this module forms DW-MSA.
#     When used after cyclic shift and with an attention mask, it forms DSW-MSA.

#     A lightweight 3x3 convolution predicts a 2D offset for every token.
#     The offsets are bounded with tanh, then used to bilinearly resample
#     the key and value feature maps inside each local window.
#     """

#     def __init__(
#         self,
#         dim: int,
#         window_size: Size2T,
#         num_heads: int,
#         qkv_bias: bool = True,
#         qk_scale: Optional[float] = None,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#         offset_scale: float = 2.0,
#     ) -> None:
#         super().__init__()

#         if dim % num_heads != 0:
#             raise ValueError(
#                 f"dim ({dim}) must be divisible by num_heads ({num_heads})."
#             )

#         self.dim = dim
#         self.window_size = to_2tuple(window_size)
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.offset_scale = float(offset_scale)

#         self.scale = (
#             qk_scale if qk_scale is not None else self.head_dim ** -0.5
#         )

#         window_h, window_w = self.window_size
#         relative_position_count = (
#             (2 * window_h - 1) * (2 * window_w - 1)
#         )

#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros(relative_position_count, num_heads)
#         )

#         coords_h = torch.arange(window_h)
#         coords_w = torch.arange(window_w)
#         coords = torch.stack(
#             torch.meshgrid(coords_h, coords_w, indexing="ij")
#         )
#         coords_flatten = torch.flatten(coords, 1)

#         relative_coords = (
#             coords_flatten[:, :, None] - coords_flatten[:, None, :]
#         )
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#         relative_coords[:, :, 0] += window_h - 1
#         relative_coords[:, :, 1] += window_w - 1
#         relative_coords[:, :, 0] *= 2 * window_w - 1

#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer(
#             "relative_position_index",
#             relative_position_index,
#             persistent=False,
#         )

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

#         # Predict one (dx, dy) offset for each token in each window.
#         self.offset_net = nn.Sequential(
#             nn.Conv2d(
#                 dim,
#                 dim,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 groups=dim,
#                 bias=True,
#             ),
#             nn.GELU(),
#             nn.Conv2d(
#                 dim,
#                 2,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#                 bias=True,
#             ),
#         )

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)

#         trunc_normal_(self.relative_position_bias_table, std=0.02)

#         # Start from ordinary W-MSA/SW-MSA and learn deformation gradually.
#         nn.init.zeros_(self.offset_net[-1].weight)
#         nn.init.zeros_(self.offset_net[-1].bias)

#     def _make_sampling_grid(
#         self,
#         x: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Create a normalized deformable sampling grid for grid_sample.

#         Args:
#             x: Window tokens with shape (B_windows, N, C).

#         Returns:
#             Grid with shape (B_windows, Wh, Ww, 2), ordered as (x, y).
#         """
#         batch_windows, num_tokens, channels = x.shape
#         window_h, window_w = self.window_size

#         if num_tokens != window_h * window_w:
#             raise ValueError(
#                 f"Token count {num_tokens} does not match window size "
#                 f"{self.window_size}."
#             )

#         feature = (
#             x.transpose(1, 2)
#             .contiguous()
#             .view(batch_windows, channels, window_h, window_w)
#         )

#         # (B_windows, 2, Wh, Ww), channel order: dx, dy.
#         offsets = torch.tanh(self.offset_net(feature))
#         offsets = offsets * self.offset_scale

#         dtype = x.dtype
#         device = x.device

#         base_y, base_x = torch.meshgrid(
#             torch.arange(window_h, device=device, dtype=dtype),
#             torch.arange(window_w, device=device, dtype=dtype),
#             indexing="ij",
#         )
#         base_grid = torch.stack((base_x, base_y), dim=-1)
#         base_grid = base_grid.unsqueeze(0).expand(
#             batch_windows, -1, -1, -1
#         )

#         offsets = offsets.permute(0, 2, 3, 1).contiguous()
#         sampling_grid = base_grid + offsets

#         # Normalize pixel coordinates to [-1, 1] for align_corners=True.
#         if window_w > 1:
#             sampling_grid[..., 0] = (
#                 2.0 * sampling_grid[..., 0] / (window_w - 1) - 1.0
#             )
#         else:
#             sampling_grid[..., 0] = 0.0

#         if window_h > 1:
#             sampling_grid[..., 1] = (
#                 2.0 * sampling_grid[..., 1] / (window_h - 1) - 1.0
#             )
#         else:
#             sampling_grid[..., 1] = 0.0

#         return sampling_grid

#     def _deform_sample(
#         self,
#         feature: torch.Tensor,
#         grid: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Bilinearly sample a per-head K or V feature map.

#         Args:
#             feature: (B_windows, num_heads, N, head_dim).
#             grid: (B_windows, Wh, Ww, 2).

#         Returns:
#             Sampled feature with shape
#             (B_windows, num_heads, N, head_dim).
#         """
#         batch_windows, num_heads, _, head_dim = feature.shape
#         window_h, window_w = self.window_size

#         feature = (
#             feature.permute(0, 1, 3, 2)
#             .contiguous()
#             .view(
#                 batch_windows * num_heads,
#                 head_dim,
#                 window_h,
#                 window_w,
#             )
#         )

#         head_grid = (
#             grid[:, None, ...]
#             .expand(-1, num_heads, -1, -1, -1)
#             .reshape(
#                 batch_windows * num_heads,
#                 window_h,
#                 window_w,
#                 2,
#             )
#         )

#         sampled = torch.nn.functional.grid_sample(
#             feature,
#             head_grid,
#             mode="bilinear",
#             padding_mode="border",
#             align_corners=True,
#         )

#         sampled = (
#             sampled.view(
#                 batch_windows,
#                 num_heads,
#                 head_dim,
#                 window_h * window_w,
#             )
#             .permute(0, 1, 3, 2)
#             .contiguous()
#         )
#         return sampled

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Args:
#             x: Window tokens with shape (B_windows, N, C).
#             mask: Shifted-window attention mask, or None.

#         Returns:
#             Tensor with shape (B_windows, N, C).
#         """
#         batch_windows, num_tokens, channels = x.shape

#         qkv = (
#             self.qkv(x)
#             .reshape(
#                 batch_windows,
#                 num_tokens,
#                 3,
#                 self.num_heads,
#                 self.head_dim,
#             )
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv.unbind(0)

#         sampling_grid = self._make_sampling_grid(x)
#         k = self._deform_sample(k, sampling_grid)
#         v = self._deform_sample(v, sampling_grid)

#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)

#         window_h, window_w = self.window_size
#         window_tokens = window_h * window_w

#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.reshape(-1)
#         ]
#         relative_position_bias = relative_position_bias.view(
#             window_tokens,
#             window_tokens,
#             self.num_heads,
#         )
#         relative_position_bias = relative_position_bias.permute(
#             2, 0, 1
#         ).contiguous()

#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             num_windows = mask.shape[0]
#             if batch_windows % num_windows != 0:
#                 raise ValueError(
#                     "Attention batch size must be divisible by mask window count."
#                 )

#             attn = attn.view(
#                 batch_windows // num_windows,
#                 num_windows,
#                 self.num_heads,
#                 num_tokens,
#                 num_tokens,
#             )
#             attn = attn + mask.unsqueeze(0).unsqueeze(2)
#             attn = attn.view(
#                 -1,
#                 self.num_heads,
#                 num_tokens,
#                 num_tokens,
#             )

#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)

#         x = (
#             (attn @ v)
#             .transpose(1, 2)
#             .reshape(batch_windows, num_tokens, channels)
#         )
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return (
#             f"dim={self.dim}, window_size={self.window_size}, "
#             f"num_heads={self.num_heads}, offset_scale={self.offset_scale}"
#         )

#     def flops(self, num_tokens: int) -> int:
#         window_h, window_w = self.window_size

#         flops = 0
#         flops += num_tokens * self.dim * 3 * self.dim
#         flops += self.num_heads * num_tokens * self.head_dim * num_tokens
#         flops += self.num_heads * num_tokens * num_tokens * self.head_dim
#         flops += num_tokens * self.dim * self.dim

#         # Approximate offset predictor cost.
#         flops += window_h * window_w * self.dim * 9
#         flops += window_h * window_w * self.dim * 2
#         return flops


# class SwinTransformerBlock(nn.Module):
#     """A single Swin Transformer block."""

#     def __init__(
#         self,
#         dim: int,
#         input_resolution: Tuple[int, int],
#         num_heads: int,
#         window_size: Size2T = 7,
#         shift_size: Size2T = 0,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         qk_scale: Optional[float] = None,
#         drop: float = 0.0,
#         attn_drop: float = 0.0,
#         drop_path: float = 0.0,
#         act_layer: Callable[[], nn.Module] = nn.GELU,
#         norm_layer: NormLayer = nn.LayerNorm,
#         offset_scale: float = 2.0,
#     ) -> None:
#         super().__init__()

#         self.dim = dim
#         self.input_resolution = tuple(input_resolution)
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio

#         input_h, input_w = self.input_resolution
#         window_h, window_w = to_2tuple(window_size)
#         shift_h, shift_w = to_2tuple(shift_size)

#         if input_h <= window_h:
#             window_h = input_h
#             shift_h = 0
#         if input_w <= window_w:
#             window_w = input_w
#             shift_w = 0

#         if not (0 <= shift_h < window_h and 0 <= shift_w < window_w):
#             raise ValueError(
#                 "Each shift dimension must satisfy 0 <= shift < window."
#             )

#         if input_h % window_h != 0 or input_w % window_w != 0:
#             raise ValueError(
#                 f"Input resolution {self.input_resolution} must be divisible "
#                 f"by effective window size {(window_h, window_w)}."
#             )

#         self.window_size = (window_h, window_w)
#         self.shift_size = (shift_h, shift_w)

#         self.norm1 = norm_layer(dim)
#         self.attn = DeformableWindowAttention(
#             dim=dim,
#             window_size=self.window_size,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             attn_drop=attn_drop,
#             proj_drop=drop,
#             offset_scale=offset_scale,
#         )

#         self.drop_path = (
#             DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
#         )
#         self.norm2 = norm_layer(dim)

#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(
#             in_features=dim,
#             hidden_features=mlp_hidden_dim,
#             act_layer=act_layer,
#             drop=drop,
#         )

#         attention_mask = self._create_attention_mask()
#         self.register_buffer(
#             "attn_mask",
#             attention_mask,
#             persistent=False,
#         )

#     def _create_attention_mask(self) -> Optional[torch.Tensor]:
#         shift_h, shift_w = self.shift_size
#         if shift_h == 0 and shift_w == 0:
#             return None

#         height, width = self.input_resolution
#         window_h, window_w = self.window_size

#         img_mask = torch.zeros((1, height, width, 1))

#         h_slices = (
#             slice(0, -window_h),
#             slice(-window_h, -shift_h),
#             slice(-shift_h, None),
#         )
#         w_slices = (
#             slice(0, -window_w),
#             slice(-window_w, -shift_w),
#             slice(-shift_w, None),
#         )

#         region_id = 0
#         for h_slice in h_slices:
#             for w_slice in w_slices:
#                 img_mask[:, h_slice, w_slice, :] = region_id
#                 region_id += 1

#         mask_windows = window_partition(img_mask, self.window_size)
#         mask_windows = mask_windows.view(
#             -1, window_h * window_w
#         )

#         attention_mask = (
#             mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         )
#         attention_mask = attention_mask.masked_fill(
#             attention_mask != 0,
#             -100.0,
#         )
#         attention_mask = attention_mask.masked_fill(
#             attention_mask == 0,
#             0.0,
#         )
#         return attention_mask

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         height, width = self.input_resolution
#         batch_size, length, channels = x.shape

#         if length != height * width:
#             raise ValueError(
#                 f"Input token length {length} does not match "
#                 f"resolution {height}x{width}."
#             )

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(batch_size, height, width, channels)

#         shift_h, shift_w = self.shift_size

#         if shift_h > 0 or shift_w > 0:
#             shifted_x = torch.roll(
#                 x,
#                 shifts=(-shift_h, -shift_w),
#                 dims=(1, 2),
#             )
#         else:
#             shifted_x = x

#         window_h, window_w = self.window_size
#         x_windows = window_partition(
#             shifted_x,
#             self.window_size,
#         )
#         x_windows = x_windows.view(
#             -1,
#             window_h * window_w,
#             channels,
#         )

#         attn_windows = self.attn(
#             x_windows,
#             mask=self.attn_mask,
#         )

#         attn_windows = attn_windows.view(
#             -1,
#             window_h,
#             window_w,
#             channels,
#         )
#         shifted_x = window_reverse(
#             attn_windows,
#             self.window_size,
#             height,
#             width,
#         )

#         if shift_h > 0 or shift_w > 0:
#             x = torch.roll(
#                 shifted_x,
#                 shifts=(shift_h, shift_w),
#                 dims=(1, 2),
#             )
#         else:
#             x = shifted_x

#         x = x.view(batch_size, height * width, channels)

#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(
#             self.mlp(self.norm2(x))
#         )
#         return x

#     def extra_repr(self) -> str:
#         return (
#             f"dim={self.dim}, "
#             f"input_resolution={self.input_resolution}, "
#             f"num_heads={self.num_heads}, "
#             f"window_size={self.window_size}, "
#             f"shift_size={self.shift_size}, "
#             f"mlp_ratio={self.mlp_ratio}"
#         )

#     def flops(self) -> int:
#         height, width = self.input_resolution
#         window_h, window_w = self.window_size
#         num_window_tokens = window_h * window_w
#         num_windows = (height * width) // num_window_tokens

#         flops = 0
#         flops += self.dim * height * width
#         flops += num_windows * self.attn.flops(num_window_tokens)
#         flops += int(
#             2 * height * width * self.dim * self.dim * self.mlp_ratio
#         )
#         flops += self.dim * height * width
#         return flops


# class PatchMerging(nn.Module):
#     """Downsample tokens by 2x spatially and expand channels by 2x."""

#     def __init__(
#         self,
#         input_resolution: Tuple[int, int],
#         dim: int,
#         norm_layer: NormLayer = nn.LayerNorm,
#     ) -> None:
#         super().__init__()

#         self.input_resolution = tuple(input_resolution)
#         self.dim = dim
#         self.norm = norm_layer(4 * dim)
#         self.reduction = nn.Linear(
#             4 * dim,
#             2 * dim,
#             bias=False,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         height, width = self.input_resolution
#         batch_size, length, channels = x.shape

#         if length != height * width:
#             raise ValueError(
#                 f"Input token length {length} does not match "
#                 f"resolution {height}x{width}."
#             )
#         if height % 2 != 0 or width % 2 != 0:
#             raise ValueError(
#                 f"PatchMerging requires an even resolution, got "
#                 f"({height}, {width})."
#             )

#         x = x.view(batch_size, height, width, channels)

#         x0 = x[:, 0::2, 0::2, :]
#         x1 = x[:, 1::2, 0::2, :]
#         x2 = x[:, 0::2, 1::2, :]
#         x3 = x[:, 1::2, 1::2, :]

#         x = torch.cat((x0, x1, x2, x3), dim=-1)
#         x = x.view(batch_size, -1, 4 * channels)

#         x = self.norm(x)
#         x = self.reduction(x)
#         return x

#     def extra_repr(self) -> str:
#         return (
#             f"input_resolution={self.input_resolution}, dim={self.dim}"
#         )

#     def flops(self) -> int:
#         height, width = self.input_resolution
#         flops = height * width * self.dim
#         flops += (
#             (height // 2)
#             * (width // 2)
#             * 4
#             * self.dim
#             * 2
#             * self.dim
#         )
#         return flops


# class BasicLayer(nn.Module):
#     """One hierarchical stage of the Swin Transformer."""

#     def __init__(
#         self,
#         dim: int,
#         input_resolution: Tuple[int, int],
#         depth: int,
#         num_heads: int,
#         window_size: Size2T,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         qk_scale: Optional[float] = None,
#         drop: float = 0.0,
#         attn_drop: float = 0.0,
#         drop_path: Union[float, Sequence[float]] = 0.0,
#         norm_layer: NormLayer = nn.LayerNorm,
#         downsample: Optional[type[nn.Module]] = None,
#         use_checkpoint: bool = False,
#         offset_scale: float = 2.0,
#     ) -> None:
#         super().__init__()

#         self.dim = dim
#         self.input_resolution = tuple(input_resolution)
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint

#         window_h, window_w = to_2tuple(window_size)
#         shift_size = (window_h // 2, window_w // 2)

#         if isinstance(drop_path, Sequence):
#             if len(drop_path) != depth:
#                 raise ValueError(
#                     "drop_path sequence length must equal stage depth."
#                 )
#             drop_path_values = list(drop_path)
#         else:
#             drop_path_values = [float(drop_path)] * depth

#         self.blocks = nn.ModuleList(
#             [
#                 SwinTransformerBlock(
#                     dim=dim,
#                     input_resolution=self.input_resolution,
#                     num_heads=num_heads,
#                     window_size=(window_h, window_w),
#                     shift_size=(0, 0) if block_index % 2 == 0 else shift_size,
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop,
#                     attn_drop=attn_drop,
#                     drop_path=drop_path_values[block_index],
#                     norm_layer=norm_layer,
#                     offset_scale=offset_scale,
#                 )
#                 for block_index in range(depth)
#             ]
#         )

#         self.downsample = (
#             downsample(
#                 self.input_resolution,
#                 dim=dim,
#                 norm_layer=norm_layer,
#             )
#             if downsample is not None
#             else None
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for block in self.blocks:
#             if self.use_checkpoint and self.training:
#                 x = checkpoint.checkpoint(
#                     block,
#                     x,
#                     use_reentrant=False,
#                 )
#             else:
#                 x = block(x)

#         if self.downsample is not None:
#             x = self.downsample(x)

#         return x

#     def extra_repr(self) -> str:
#         return (
#             f"dim={self.dim}, "
#             f"input_resolution={self.input_resolution}, "
#             f"depth={self.depth}"
#         )

#     def flops(self) -> int:
#         flops = sum(block.flops() for block in self.blocks)
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops


# class PatchEmbed(nn.Module):
#     """Convert an image into a sequence of non-overlapping patch tokens."""

#     def __init__(
#         self,
#         img_size: Size2T = 224,
#         patch_size: Size2T = 4,
#         in_chans: int = 3,
#         embed_dim: int = 96,
#         norm_layer: Optional[NormLayer] = None,
#     ) -> None:
#         super().__init__()

#         self.img_size = to_2tuple(img_size)
#         self.patch_size = to_2tuple(patch_size)

#         if (
#             self.img_size[0] % self.patch_size[0] != 0
#             or self.img_size[1] % self.patch_size[1] != 0
#         ):
#             raise ValueError(
#                 f"Image size {self.img_size} must be divisible by "
#                 f"patch size {self.patch_size}."
#             )

#         self.patches_resolution = (
#             self.img_size[0] // self.patch_size[0],
#             self.img_size[1] // self.patch_size[1],
#         )
#         self.num_patches = (
#             self.patches_resolution[0]
#             * self.patches_resolution[1]
#         )

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         self.proj = nn.Conv2d(
#             in_chans,
#             embed_dim,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#         )
#         self.norm = (
#             norm_layer(embed_dim)
#             if norm_layer is not None
#             else None
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, _, height, width = x.shape

#         if (height, width) != self.img_size:
#             raise ValueError(
#                 f"Input image size ({height}, {width}) does not match "
#                 f"configured size {self.img_size}."
#             )

#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x

#     def flops(self) -> int:
#         output_h, output_w = self.patches_resolution
#         kernel_area = self.patch_size[0] * self.patch_size[1]

#         flops = (
#             output_h
#             * output_w
#             * self.embed_dim
#             * self.in_chans
#             * kernel_area
#         )

#         if self.norm is not None:
#             flops += output_h * output_w * self.embed_dim

#         return flops


# class SwinTransformer(nn.Module):
#     """
#     Hierarchical Swin Transformer for image classification.

#     Default configuration corresponds to Swin-Tiny:
#         embed_dim=96
#         depths=(2, 2, 6, 2)
#         num_heads=(3, 6, 12, 24)
#     """

#     def __init__(
#         self,
#         img_size: Size2T = 224,
#         patch_size: Size2T = 4,
#         in_chans: int = 3,
#         num_classes: int = 1000,
#         embed_dim: int = 96,
#         depths: Sequence[int] = (2, 2, 6, 2),
#         num_heads: Sequence[int] = (3, 6, 12, 24),
#         window_size: Size2T = 7,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         qk_scale: Optional[float] = None,
#         drop_rate: float = 0.0,
#         attn_drop_rate: float = 0.0,
#         drop_path_rate: float = 0.1,
#         norm_layer: NormLayer = nn.LayerNorm,
#         ape: bool = False,
#         patch_norm: bool = True,
#         use_checkpoint: bool = False,
#         deformable_offset_scale: float = 2.0,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         del kwargs

#         if len(depths) != len(num_heads):
#             raise ValueError(
#                 "depths and num_heads must have the same length."
#             )

#         self.img_size = to_2tuple(img_size)
#         self.patch_size = to_2tuple(patch_size)
#         self.in_chans = in_chans
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.mlp_ratio = mlp_ratio
#         self.num_features = int(
#             embed_dim * 2 ** (self.num_layers - 1)
#         )

#         self.patch_embed = PatchEmbed(
#             img_size=self.img_size,
#             patch_size=self.patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if patch_norm else None,
#         )

#         num_patches = self.patch_embed.num_patches
#         self.patches_resolution = (
#             self.patch_embed.patches_resolution
#         )

#         minimum_downsample_factor = 2 ** (self.num_layers - 1)
#         if (
#             self.patches_resolution[0] % minimum_downsample_factor != 0
#             or self.patches_resolution[1] % minimum_downsample_factor != 0
#         ):
#             raise ValueError(
#                 f"Patch resolution {self.patches_resolution} must be "
#                 f"divisible by {minimum_downsample_factor} for "
#                 f"{self.num_layers} stages."
#             )

#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(
#                 torch.zeros(
#                     1,
#                     num_patches,
#                     embed_dim,
#                 )
#             )
#             trunc_normal_(
#                 self.absolute_pos_embed,
#                 std=0.02,
#             )
#         else:
#             self.absolute_pos_embed = None

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         total_depth = sum(depths)
#         drop_path_values = torch.linspace(
#             0,
#             drop_path_rate,
#             total_depth,
#         ).tolist()

#         self.layers = nn.ModuleList()
#         depth_offset = 0

#         for layer_index in range(self.num_layers):
#             layer_dim = int(embed_dim * 2 ** layer_index)
#             layer_resolution = (
#                 self.patches_resolution[0] // (2 ** layer_index),
#                 self.patches_resolution[1] // (2 ** layer_index),
#             )
#             layer_depth = depths[layer_index]

#             layer = BasicLayer(
#                 dim=layer_dim,
#                 input_resolution=layer_resolution,
#                 depth=layer_depth,
#                 num_heads=num_heads[layer_index],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=drop_path_values[
#                     depth_offset : depth_offset + layer_depth
#                 ],
#                 norm_layer=norm_layer,
#                 downsample=(
#                     PatchMerging
#                     if layer_index < self.num_layers - 1
#                     else None
#                 ),
#                 use_checkpoint=use_checkpoint,
#                 offset_scale=deformable_offset_scale,
#             )

#             self.layers.append(layer)
#             depth_offset += layer_depth

#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = (
#             nn.Linear(self.num_features, num_classes)
#             if num_classes > 0
#             else nn.Identity()
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, module: nn.Module) -> None:
#         if isinstance(module, nn.Linear):
#             trunc_normal_(module.weight, std=0.02)
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.constant_(module.bias, 0)
#             nn.init.constant_(module.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self) -> set[str]:
#         return {"absolute_pos_embed"}

#     @torch.jit.ignore
#     def no_weight_decay_keywords(self) -> set[str]:
#         return {"relative_position_bias_table"}

#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.patch_embed(x)

#         if self.absolute_pos_embed is not None:
#             x = x + self.absolute_pos_embed

#         x = self.pos_drop(x)

#         for layer in self.layers:
#             x = layer(x)

#         x = self.norm(x)
#         x = self.avgpool(x.transpose(1, 2))
#         x = torch.flatten(x, 1)
#         return x

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x

#     def flops(self) -> int:
#         flops = self.patch_embed.flops()

#         for layer in self.layers:
#             flops += layer.flops()

#         final_h = self.patches_resolution[0] // (
#             2 ** (self.num_layers - 1)
#         )
#         final_w = self.patches_resolution[1] // (
#             2 ** (self.num_layers - 1)
#         )

#         flops += self.num_features * final_h * final_w
#         flops += self.num_features * self.num_classes
#         return flops


# def build_swin(config) -> SwinTransformer:
#     """
#     Build a Swin Transformer from a project configuration object.

#     Expected configuration fields:
#         config.DATA.IMG_SIZE
#         config.MODEL.NUM_CLASSES
#         config.MODEL.DROP_RATE
#         config.MODEL.ATTN_DROP_RATE      optional
#         config.MODEL.DROP_PATH_RATE
#         config.MODEL.SWIN.PATCH_SIZE
#         config.MODEL.SWIN.IN_CHANS
#         config.MODEL.SWIN.EMBED_DIM
#         config.MODEL.SWIN.DEPTHS
#         config.MODEL.SWIN.NUM_HEADS
#         config.MODEL.SWIN.WINDOW_SIZE
#         config.MODEL.SWIN.MLP_RATIO
#         config.MODEL.SWIN.QKV_BIAS
#         config.MODEL.SWIN.QK_SCALE
#         config.MODEL.SWIN.APE
#         config.MODEL.SWIN.PATCH_NORM
#         config.TRAIN.USE_CHECKPOINT
#     """
#     attn_drop_rate = getattr(
#         config.MODEL,
#         "ATTN_DROP_RATE",
#         0.0,
#     )

#     deformable_offset_scale = getattr(
#         config.MODEL.SWIN,
#         "DEFORMABLE_OFFSET_SCALE",
#         2.0,
#     )

#     model = SwinTransformer(
#         img_size=config.DATA.IMG_SIZE,
#         patch_size=config.MODEL.SWIN.PATCH_SIZE,
#         in_chans=config.MODEL.SWIN.IN_CHANS,
#         num_classes=config.MODEL.NUM_CLASSES,
#         embed_dim=config.MODEL.SWIN.EMBED_DIM,
#         depths=tuple(config.MODEL.SWIN.DEPTHS),
#         num_heads=tuple(config.MODEL.SWIN.NUM_HEADS),
#         window_size=config.MODEL.SWIN.WINDOW_SIZE,
#         mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#         qkv_bias=config.MODEL.SWIN.QKV_BIAS,
#         qk_scale=config.MODEL.SWIN.QK_SCALE,
#         drop_rate=config.MODEL.DROP_RATE,
#         attn_drop_rate=attn_drop_rate,
#         drop_path_rate=config.MODEL.DROP_PATH_RATE,
#         ape=config.MODEL.SWIN.APE,
#         patch_norm=config.MODEL.SWIN.PATCH_NORM,
#         use_checkpoint=config.TRAIN.USE_CHECKPOINT,
#         deformable_offset_scale=deformable_offset_scale,
#     )

#     return model


#第1版本v2
"""
Accuracy-oriented Deformable Swin Transformer using DW-MSA and DSW-MSA.

Based on:
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- SimMIM / Microsoft reference implementation

Improvements in this consolidated version:
1. Removes duplicated/commented code.
2. Uses torch.meshgrid(..., indexing="ij") for modern PyTorch compatibility.
3. Avoids mutable list defaults.
4. Adds clear shape and divisibility validation.
5. Supports tuple image/window sizes.
6. Uses checkpoint(..., use_reentrant=False) when enabled.
7. Provides configurable attention dropout in build_swin.
"""

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


Size2T = Union[int, Tuple[int, int]]
NormLayer = Callable[[int], nn.Module]


class Mlp(nn.Module):
    """Feed-forward network used inside each Swin Transformer block."""

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


def window_partition(x: torch.Tensor, window_size: Size2T) -> torch.Tensor:
    """
    Partition a feature map into non-overlapping windows.

    Args:
        x: Tensor with shape (B, H, W, C).
        window_size: Integer or (window_height, window_width).

    Returns:
        Windows with shape:
        (B * num_windows, window_height, window_width, C).
    """
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
    """
    Merge windows back into a feature map.

    Args:
        windows: Tensor with shape
            (B * num_windows, window_height, window_width, C).
        window_size: Integer or (window_height, window_width).
        height: Output feature-map height.
        width: Output feature-map width.

    Returns:
        Tensor with shape (B, H, W, C).
    """
    window_h, window_w = to_2tuple(window_size)

    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(
            f"Output size ({height}, {width}) must be divisible by "
            f"window size ({window_h}, {window_w})."
        )

    windows_per_image = (height // window_h) * (width // window_w)

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
    Deformable window-based multi-head self-attention.

    When used without cyclic shift, this module forms DW-MSA.
    When used after cyclic shift and with an attention mask, it forms DSW-MSA.

    A lightweight 3x3 convolution predicts a 2D offset for every token.
    The offsets are bounded with tanh, then used to bilinearly resample
    the key and value feature maps inside each local window.
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

        # Normalize only the offset branch; the main Swin path is unchanged.
        self.offset_norm = nn.LayerNorm(dim)

        # Per-head residual deformation strength. Different attention heads
        # can learn different deformation ratios instead of sharing one gate.
        # The offset predictor is zero initialized, so the initial behavior
        # still exactly matches ordinary W-MSA / SW-MSA.
        self.deform_gate = nn.Parameter(
            torch.full((num_heads, 1, 1), -1.5)
        )

        # Local positional enhancement on the value branch. Its learnable
        # scale starts at zero, preserving the original Swin initialization.
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
            qk_scale if qk_scale is not None else self.head_dim ** -0.5
        )

        window_h, window_w = self.window_size
        relative_position_count = (
            (2 * window_h - 1) * (2 * window_w - 1)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_position_count, num_heads)
        )

        coords_h = torch.arange(window_h)
        coords_w = torch.arange(window_w)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_h - 1
        relative_coords[:, :, 1] += window_w - 1
        relative_coords[:, :, 0] *= 2 * window_w - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Predict one (dx, dy) offset for each token in each window.
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

        # Zero-initialized channel refinement. It can improve representation
        # capacity while preserving the original output at initialization.
        self.output_refine = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.output_refine_scale = nn.Parameter(torch.tensor(0.0))

        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Start from ordinary W-MSA/SW-MSA and learn deformation gradually.
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        nn.init.zeros_(self.lepe_conv.weight)
        nn.init.zeros_(self.lepe_conv.bias)
        nn.init.zeros_(self.output_refine[-1].weight)
        nn.init.zeros_(self.output_refine[-1].bias)

    def _make_sampling_grid(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a normalized deformable sampling grid for grid_sample.

        Args:
            x: Window tokens with shape (B_windows, N, C).

        Returns:
            Grid with shape (B_windows, Wh, Ww, 2), ordered as (x, y).
        """
        batch_windows, num_tokens, channels = x.shape
        window_h, window_w = self.window_size

        if num_tokens != window_h * window_w:
            raise ValueError(
                f"Token count {num_tokens} does not match window size "
                f"{self.window_size}."
            )

        offset_input = self.offset_norm(x)
        feature = (
            offset_input.transpose(1, 2)
            .contiguous()
            .view(batch_windows, channels, window_h, window_w)
        )

        # (B_windows, 2, Wh, Ww), channel order: dx, dy.
        offsets = self.offset_net(feature)

        # Remove whole-window translation and retain local deformation.
        offsets = offsets - offsets.mean(dim=(2, 3), keepdim=True)

        # Bound offsets; feature-level residual gating controls how much
        # deformation enters attention.
        offsets = torch.tanh(offsets) * self.offset_scale
        self.last_offsets = offsets

        dtype = x.dtype
        device = x.device

        base_y, base_x = torch.meshgrid(
            torch.arange(window_h, device=device, dtype=dtype),
            torch.arange(window_w, device=device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack((base_x, base_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(
            batch_windows, -1, -1, -1
        )

        offsets = offsets.permute(0, 2, 3, 1).contiguous()
        sampling_grid = base_grid + offsets

        # Keep sampling coordinates inside the local window.
        sampling_grid_x = sampling_grid[..., 0].clamp(
            0.0, float(max(window_w - 1, 0))
        )
        sampling_grid_y = sampling_grid[..., 1].clamp(
            0.0, float(max(window_h - 1, 0))
        )
        sampling_grid = torch.stack(
            (sampling_grid_x, sampling_grid_y), dim=-1
        )

        # Normalize pixel coordinates to [-1, 1] for align_corners=True.
        if window_w > 1:
            sampling_grid[..., 0] = (
                2.0 * sampling_grid[..., 0] / (window_w - 1) - 1.0
            )
        else:
            sampling_grid[..., 0] = 0.0

        if window_h > 1:
            sampling_grid[..., 1] = (
                2.0 * sampling_grid[..., 1] / (window_h - 1) - 1.0
            )
        else:
            sampling_grid[..., 1] = 0.0

        return sampling_grid

    def _deform_sample(
        self,
        feature: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinearly sample a per-head K or V feature map.

        Args:
            feature: (B_windows, num_heads, N, head_dim).
            grid: (B_windows, Wh, Ww, 2).

        Returns:
            Sampled feature with shape
            (B_windows, num_heads, N, head_dim).
        """
        batch_windows, num_heads, _, head_dim = feature.shape
        window_h, window_w = self.window_size

        feature = (
            feature.permute(0, 1, 3, 2)
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
            .expand(-1, num_heads, -1, -1, -1)
            .reshape(
                batch_windows * num_heads,
                window_h,
                window_w,
                2,
            )
        )

        sampled = torch.nn.functional.grid_sample(
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
        """
        Depth-wise local positional enhancement for the value branch.

        Args:
            x: Window tokens with shape (B_windows, N, C).

        Returns:
            Tensor with shape (B_windows, num_heads, N, head_dim).
        """
        batch_windows, num_tokens, channels = x.shape
        window_h, window_w = self.window_size

        feature = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_windows, channels, window_h, window_w)
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
        """
        Args:
            x: Window tokens with shape (B_windows, N, C).
            mask: Shifted-window attention mask, or None.

        Returns:
            Tensor with shape (B_windows, N, C).
        """
        batch_windows, num_tokens, channels = x.shape

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

        sampling_grid = self._make_sampling_grid(x)
        sampled_k = self._deform_sample(k, sampling_grid)
        sampled_v = self._deform_sample(v, sampling_grid)

        # Per-head identity-preserving deformable fusion.
        deform_gate = torch.sigmoid(self.deform_gate).unsqueeze(0)
        k = k + deform_gate * (sampled_k - k)
        v = v + deform_gate * (sampled_v - v)

        # Add local positional enhancement to V. The branch starts disabled
        # because lepe_scale is zero initialized.
        lepe = self._local_position_enhancement(x)
        lepe_scale = torch.tanh(self.lepe_scale).unsqueeze(0)
        v = v + lepe_scale * lepe

        self.last_deform_gate = deform_gate.detach()
        self.last_lepe_scale = lepe_scale.detach()

        # Keep original Swin scaled dot-product attention.
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        window_h, window_w = self.window_size
        window_tokens = window_h * window_w

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            window_tokens,
            window_tokens,
            self.num_heads,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            if batch_windows % num_windows != 0:
                raise ValueError(
                    "Attention batch size must be divisible by mask window count."
                )

            attn = attn.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(
                -1,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

        # FP32 softmax avoids occasional overflow under mixed precision.
        attn_dtype = attn.dtype
        attn = self.softmax(attn.float()).to(attn_dtype)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(batch_windows, num_tokens, channels)
        )
        x = self.proj(x)
        x = self.proj_drop(x)

        refine_scale = torch.tanh(self.output_refine_scale)
        x = x + refine_scale * self.output_refine(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"num_heads={self.num_heads}, offset_scale={self.offset_scale}"
        )

    def flops(self, num_tokens: int) -> int:
        window_h, window_w = self.window_size

        flops = 0
        flops += num_tokens * self.dim * 3 * self.dim
        flops += self.num_heads * num_tokens * self.head_dim * num_tokens
        flops += self.num_heads * num_tokens * num_tokens * self.head_dim
        flops += num_tokens * self.dim * self.dim

        # Approximate offset predictor cost.
        flops += window_h * window_w * self.dim * 9
        flops += window_h * window_w * self.dim * 2

        # LEPE depth-wise convolution and output refinement MLP.
        flops += window_h * window_w * self.dim * 9
        flops += 2 * num_tokens * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """A single Swin Transformer block."""

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
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        input_h, input_w = self.input_resolution
        window_h, window_w = to_2tuple(window_size)
        shift_h, shift_w = to_2tuple(shift_size)

        if input_h <= window_h:
            window_h = input_h
            shift_h = 0
        if input_w <= window_w:
            window_w = input_w
            shift_w = 0

        if not (0 <= shift_h < window_h and 0 <= shift_w < window_w):
            raise ValueError(
                "Each shift dimension must satisfy 0 <= shift < window."
            )

        if input_h % window_h != 0 or input_w % window_w != 0:
            raise ValueError(
                f"Input resolution {self.input_resolution} must be divisible "
                f"by effective window size {(window_h, window_w)}."
            )

        self.window_size = (window_h, window_w)
        self.shift_size = (shift_h, shift_w)

        self.norm1 = norm_layer(dim)
        self.attn = DeformableWindowAttention(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            offset_scale=offset_scale,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        attention_mask = self._create_attention_mask()
        self.register_buffer(
            "attn_mask",
            attention_mask,
            persistent=False,
        )

    def _create_attention_mask(self) -> Optional[torch.Tensor]:
        shift_h, shift_w = self.shift_size
        if shift_h == 0 and shift_w == 0:
            return None

        height, width = self.input_resolution
        window_h, window_w = self.window_size

        img_mask = torch.zeros((1, height, width, 1))

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
                img_mask[:, h_slice, w_slice, :] = region_id
                region_id += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(
            -1, window_h * window_w
        )

        attention_mask = (
            mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        )
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            -100.0,
        )
        attention_mask = attention_mask.masked_fill(
            attention_mask == 0,
            0.0,
        )
        return attention_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, length, channels = x.shape

        if length != height * width:
            raise ValueError(
                f"Input token length {length} does not match "
                f"resolution {height}x{width}."
            )

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch_size, height, width, channels)

        shift_h, shift_w = self.shift_size

        if shift_h > 0 or shift_w > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-shift_h, -shift_w),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        window_h, window_w = self.window_size
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

        attn_windows = attn_windows.view(
            -1,
            window_h,
            window_w,
            channels,
        )
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            height,
            width,
        )

        if shift_h > 0 or shift_w > 0:
            x = torch.roll(
                shifted_x,
                shifts=(shift_h, shift_w),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(batch_size, height * width, channels)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(
            self.mlp(self.norm2(x))
        )
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, "
            f"window_size={self.window_size}, "
            f"shift_size={self.shift_size}, "
            f"mlp_ratio={self.mlp_ratio}"
        )

    def flops(self) -> int:
        height, width = self.input_resolution
        window_h, window_w = self.window_size
        num_window_tokens = window_h * window_w
        num_windows = (height * width) // num_window_tokens

        flops = 0
        flops += self.dim * height * width
        flops += num_windows * self.attn.flops(num_window_tokens)
        flops += int(
            2 * height * width * self.dim * self.dim * self.mlp_ratio
        )
        flops += self.dim * height * width
        return flops


class PatchMerging(nn.Module):
    """Downsample tokens by 2x spatially and expand channels by 2x."""

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: NormLayer = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.input_resolution = tuple(input_resolution)
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(
            4 * dim,
            2 * dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch_size, length, channels = x.shape

        if length != height * width:
            raise ValueError(
                f"Input token length {length} does not match "
                f"resolution {height}x{width}."
            )
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"PatchMerging requires an even resolution, got "
                f"({height}, {width})."
            )

        x = x.view(batch_size, height, width, channels)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat((x0, x1, x2, x3), dim=-1)
        x = x.view(batch_size, -1, 4 * channels)

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"input_resolution={self.input_resolution}, dim={self.dim}"
        )

    def flops(self) -> int:
        height, width = self.input_resolution
        flops = height * width * self.dim
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
    """One hierarchical stage of the Swin Transformer."""

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
        drop_path: Union[float, Sequence[float]] = 0.0,
        norm_layer: NormLayer = nn.LayerNorm,
        downsample: Optional[type[nn.Module]] = None,
        use_checkpoint: bool = False,
        offset_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        window_h, window_w = to_2tuple(window_size)
        shift_size = (window_h // 2, window_w // 2)

        if isinstance(drop_path, Sequence):
            if len(drop_path) != depth:
                raise ValueError(
                    "drop_path sequence length must equal stage depth."
                )
            drop_path_values = list(drop_path)
        else:
            drop_path_values = [float(drop_path)] * depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    window_size=(window_h, window_w),
                    shift_size=(0, 0) if block_index % 2 == 0 else shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_values[block_index],
                    norm_layer=norm_layer,
                    offset_scale=offset_scale,
                )
                for block_index in range(depth)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"input_resolution={self.input_resolution}, "
            f"depth={self.depth}"
        )

    def flops(self) -> int:
        flops = sum(block.flops() for block in self.blocks)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """Convert an image into a sequence of non-overlapping patch tokens."""

    def __init__(
        self,
        img_size: Size2T = 224,
        patch_size: Size2T = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        if (
            self.img_size[0] % self.patch_size[0] != 0
            or self.img_size[1] % self.patch_size[1] != 0
        ):
            raise ValueError(
                f"Image size {self.img_size} must be divisible by "
                f"patch size {self.patch_size}."
            )

        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape

        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size ({height}, {width}) does not match "
                f"configured size {self.img_size}."
            )

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def flops(self) -> int:
        output_h, output_w = self.patches_resolution
        kernel_area = self.patch_size[0] * self.patch_size[1]

        flops = (
            output_h
            * output_w
            * self.embed_dim
            * self.in_chans
            * kernel_area
        )

        if self.norm is not None:
            flops += output_h * output_w * self.embed_dim

        return flops


class SwinTransformer(nn.Module):
    """
    Hierarchical Swin Transformer for image classification.

    Default configuration corresponds to Swin-Tiny:
        embed_dim=96
        depths=(2, 2, 6, 2)
        num_heads=(3, 6, 12, 24)
    """

    def __init__(
        self,
        img_size: Size2T = 224,
        patch_size: Size2T = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Size2T = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
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

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.num_features = int(
            embed_dim * 2 ** (self.num_layers - 1)
        )

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        num_patches = self.patch_embed.num_patches
        self.patches_resolution = (
            self.patch_embed.patches_resolution
        )

        minimum_downsample_factor = 2 ** (self.num_layers - 1)
        if (
            self.patches_resolution[0] % minimum_downsample_factor != 0
            or self.patches_resolution[1] % minimum_downsample_factor != 0
        ):
            raise ValueError(
                f"Patch resolution {self.patches_resolution} must be "
                f"divisible by {minimum_downsample_factor} for "
                f"{self.num_layers} stages."
            )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    num_patches,
                    embed_dim,
                )
            )
            trunc_normal_(
                self.absolute_pos_embed,
                std=0.02,
            )
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum(depths)
        drop_path_values = torch.linspace(
            0,
            drop_path_rate,
            total_depth,
        ).tolist()

        self.layers = nn.ModuleList()
        depth_offset = 0

        for layer_index in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** layer_index)
            layer_resolution = (
                self.patches_resolution[0] // (2 ** layer_index),
                self.patches_resolution[1] // (2 ** layer_index),
            )
            layer_depth = depths[layer_index]

            layer = BasicLayer(
                dim=layer_dim,
                input_resolution=layer_resolution,
                depth=layer_depth,
                num_heads=num_heads[layer_index],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_values[
                    depth_offset : depth_offset + layer_depth
                ],
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging
                    if layer_index < self.num_layers - 1
                    else None
                ),
                use_checkpoint=use_checkpoint,
                offset_scale=deformable_offset_scale,
            )

            self.layers.append(layer)
            depth_offset += layer_depth

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        return {"relative_position_bias_table"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self) -> int:
        flops = self.patch_embed.flops()

        for layer in self.layers:
            flops += layer.flops()

        final_h = self.patches_resolution[0] // (
            2 ** (self.num_layers - 1)
        )
        final_w = self.patches_resolution[1] // (
            2 ** (self.num_layers - 1)
        )

        flops += self.num_features * final_h * final_w
        flops += self.num_features * self.num_classes
        return flops


def build_swin(config) -> SwinTransformer:
    """
    Build a Swin Transformer from a project configuration object.

    Expected configuration fields:
        config.DATA.IMG_SIZE
        config.MODEL.NUM_CLASSES
        config.MODEL.DROP_RATE
        config.MODEL.ATTN_DROP_RATE      optional
        config.MODEL.DROP_PATH_RATE
        config.MODEL.SWIN.PATCH_SIZE
        config.MODEL.SWIN.IN_CHANS
        config.MODEL.SWIN.EMBED_DIM
        config.MODEL.SWIN.DEPTHS
        config.MODEL.SWIN.NUM_HEADS
        config.MODEL.SWIN.WINDOW_SIZE
        config.MODEL.SWIN.MLP_RATIO
        config.MODEL.SWIN.QKV_BIAS
        config.MODEL.SWIN.QK_SCALE
        config.MODEL.SWIN.APE
        config.MODEL.SWIN.PATCH_NORM
        config.TRAIN.USE_CHECKPOINT
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

    model = SwinTransformer(
        img_size=config.DATA.IMG_SIZE,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        embed_dim=config.MODEL.SWIN.EMBED_DIM,
        depths=tuple(config.MODEL.SWIN.DEPTHS),
        num_heads=tuple(config.MODEL.SWIN.NUM_HEADS),
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        qk_scale=config.MODEL.SWIN.QK_SCALE,
        drop_rate=config.MODEL.DROP_RATE,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        deformable_offset_scale=deformable_offset_scale,
    )

    return model

