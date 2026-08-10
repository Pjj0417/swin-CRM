import copy
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


SWIN_CMAE_MODEL_CANDIDATES = (
    "swin_base_patch4_window7_224.ms_in1k",
    "swin_base_patch4_window7_224",
)


def _create_swin_base(
    drop_path_rate: float = 0.10,
):
    last_error = None

    for model_name in SWIN_CMAE_MODEL_CANDIDATES:
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                drop_path_rate=float(drop_path_rate),
            )
            return model, model_name
        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Could not create Swin-B/4/7/224 with the installed timm. "
        f"Tried: {SWIN_CMAE_MODEL_CANDIDATES}. "
        f"Last error: {last_error}"
    )


def _patch_grid(backbone) -> Tuple[int, int]:
    grid = getattr(
        backbone.patch_embed,
        "grid_size",
        None,
    )

    if grid is not None:
        return int(grid[0]), int(grid[1])

    return (56, 56)


def _ensure_mask_shape(
    mask: torch.Tensor,
    x: torch.Tensor,
    grid: Tuple[int, int],
):
    """
    Adapt [B, H*W] mask to the patch embedding layout.

    Current timm Swin: [B,H,W,C]
    Older timm Swin:   [B,H*W,C]
    """
    b = x.shape[0]
    h, w = grid

    if x.ndim == 4:
        return mask.reshape(b, h, w, 1)

    if x.ndim == 3:
        return mask.reshape(b, h * w, 1)

    raise RuntimeError(
        "Unexpected Swin patch embedding layout: "
        f"{tuple(x.shape)}"
    )


def _mask_token_for(
    mask_token: torch.Tensor,
    x: torch.Tensor,
):
    if x.ndim == 4:
        return mask_token

    if x.ndim == 3:
        return mask_token.reshape(
            1,
            1,
            mask_token.shape[-1],
        )

    raise RuntimeError(
        f"Unexpected activation layout: {tuple(x.shape)}"
    )


def _to_bchw(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Convert final Swin feature map to BCHW.

    Current timm returns BHWC; older implementations may return BLC.
    """
    if x.ndim == 4:
        # BHWC is the normal timm Swin representation.
        if x.shape[-1] >= x.shape[1]:
            return x.permute(
                0, 3, 1, 2
            ).contiguous()

        return x

    if x.ndim == 3:
        b, n, c = x.shape
        spatial = int(
            round(math.sqrt(n))
        )
        if spatial * spatial != n:
            raise RuntimeError(
                f"Cannot reshape {n} Swin tokens "
                "to a square feature map."
            )

        return (
            x.reshape(
                b,
                spatial,
                spatial,
                c,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    raise RuntimeError(
        f"Unexpected Swin feature shape: {tuple(x.shape)}"
    )


def _pool_swin_features(
    x: torch.Tensor,
) -> torch.Tensor:
    if x.ndim == 4:
        # Current timm: BHWC.
        if x.shape[-1] >= x.shape[1]:
            return x.mean(dim=(1, 2))

        return x.mean(dim=(2, 3))

    if x.ndim == 3:
        return x.mean(dim=1)

    if x.ndim == 2:
        return x

    raise RuntimeError(
        f"Unexpected feature layout: {tuple(x.shape)}"
    )


def _forward_from_patch_embeddings(
    backbone,
    x: torch.Tensor,
):
    """
    Continue a timm Swin forward pass after patch_embed.
    """
    if hasattr(backbone, "absolute_pos_embed"):
        absolute_pos_embed = getattr(
            backbone,
            "absolute_pos_embed",
        )
        if absolute_pos_embed is not None:
            if x.ndim == 3:
                x = x + absolute_pos_embed
            elif x.ndim == 4:
                b, h, w, c = x.shape
                pos = absolute_pos_embed
                if pos.ndim == 3:
                    pos = pos.reshape(
                        1,
                        h,
                        w,
                        c,
                    )
                x = x + pos

    if hasattr(backbone, "pos_drop"):
        x = backbone.pos_drop(x)

    x = backbone.layers(x)
    x = backbone.norm(x)
    return x


def _full_encoder_features(
    backbone,
    images: torch.Tensor,
):
    return backbone.forward_features(images)


def _extract_checkpoint_state(
    checkpoint,
):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Swin-CMAE checkpoint must be a dict."
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


def _extract_online_encoder(
    checkpoint,
):
    state = _extract_checkpoint_state(
        checkpoint
    )
    encoder = {}

    prefixes = (
        "online_encoder.",
        "module.online_encoder.",
        "model.online_encoder.",
    )
    has_online = any(
        any(
            key.startswith(prefix)
            for prefix in prefixes
        )
        for key in state
    )

    for key, value in state.items():
        clean = key

        if clean.startswith("module."):
            clean = clean[len("module."):]

        if clean.startswith("model."):
            clean = clean[len("model."):]

        if has_online:
            if not clean.startswith(
                "online_encoder."
            ):
                continue
            clean = clean[
                len("online_encoder.") :
            ]

        # A checkpoint already extracted to encoder-only
        # can be loaded as-is.
        if (
            clean.startswith("pixel_decoder.")
            or clean.startswith("feature_decoder.")
            or clean.startswith("projector.")
            or clean.startswith("momentum_")
            or clean.startswith("classifier.")
            or clean == "mask_token"
            or clean.startswith("queue")
        ):
            continue

        encoder[clean] = value

    return encoder


def _parameter_coverage(
    state_dict,
    model,
):
    model_state = model.state_dict()

    total_numel = 0
    matched_numel = 0
    matched_keys = set()

    for key, tensor in model_state.items():
        # Position indices are deterministic buffers and
        # can differ across timm revisions.
        if (
            "relative_position_index"
            in key
            or "attn_mask"
            in key
        ):
            continue

        total_numel += tensor.numel()

        other = state_dict.get(key)
        if (
            other is not None
            and tuple(other.shape)
            == tuple(tensor.shape)
        ):
            matched_numel += (
                tensor.numel()
            )
            matched_keys.add(key)

    coverage = (
        matched_numel
        / max(1, total_numel)
    )

    return coverage, matched_keys


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                out_dim,
            ),
        )

    def forward(self, x):
        return self.net(x)


class FeatureDecoder(nn.Module):
    """
    Lightweight online feature decoder.

    CMAE uses a feature decoder so that contrastive supervision is
    applied to complemented online features rather than directly to
    incomplete masked patch features.
    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        if x.ndim == 4:
            # BHWC.
            if x.shape[-1] >= x.shape[1]:
                x = self.norm(x)
                x = self.mlp(x)
                return x.mean(
                    dim=(1, 2)
                )

            # BCHW fallback.
            x = x.permute(
                0, 2, 3, 1
            )
            x = self.norm(x)
            x = self.mlp(x)
            return x.mean(
                dim=(1, 2)
            )

        if x.ndim == 3:
            x = self.norm(x)
            x = self.mlp(x)
            return x.mean(dim=1)

        raise RuntimeError(
            f"Unexpected feature decoder input: "
            f"{tuple(x.shape)}"
        )


class PixelDecoder(nn.Module):
    """
    Decode final 7x7 Swin-B features back to the 56x56 patch grid.
    Each output position predicts one normalized 4x4 RGB patch.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        decoder_dim: int = 256,
        patch_size: int = 4,
    ):
        super().__init__()

        patch_dim = (
            patch_size
            * patch_size
            * 3
        )

        self.patch_size = int(
            patch_size
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_dim,
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
                patch_dim,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        latent,
        patch_grid=(56, 56),
    ):
        x = _to_bchw(latent)
        x = self.decoder(x)

        if x.shape[-2:] != patch_grid:
            x = F.interpolate(
                x,
                size=patch_grid,
                mode="bilinear",
                align_corners=False,
            )

        b = x.shape[0]

        return (
            x.permute(
                0, 2, 3, 1
            )
            .contiguous()
            .reshape(
                b,
                patch_grid[0]
                * patch_grid[1],
                -1,
            )
        )


class SwinCMAEPretrain(nn.Module):
    """
    Swin-B + CMAE-style self-supervised pretraining.

    Online branch:
        masked Swin-B encoder
        -> pixel decoder for MAE reconstruction
        -> feature decoder + projector for contrastive query

    Momentum branch:
        full-view EMA Swin-B encoder
        -> EMA projector for contrastive key

    A MoCo-style queue is added for practical single-GPU training with
    small batches. This queue is an adaptation for this project, not a
    claim of exact architectural identity with the published CMAE.
    """

    def __init__(
        self,
        mask_ratio: float = 0.75,
        mask_window: int = 4,
        decoder_dim: int = 256,
        projection_dim: int = 256,
        projection_hidden_dim: int = 2048,
        temperature: float = 0.20,
        contrast_weight: float = 0.10,
        queue_size: int = 4096,
        drop_path_rate: float = 0.10,
        norm_pix_loss: bool = True,
    ):
        super().__init__()

        (
            self.online_encoder,
            self.model_name,
        ) = _create_swin_base(
            drop_path_rate=drop_path_rate,
        )

        self.momentum_encoder = copy.deepcopy(
            self.online_encoder
        )

        self.embed_dim = int(
            self.online_encoder.embed_dim
        )
        self.num_features = int(
            self.online_encoder.num_features
        )
        self.patch_grid = _patch_grid(
            self.online_encoder
        )
        self.patch_size = 4

        self.mask_ratio = float(
            mask_ratio
        )
        self.mask_window = int(
            mask_window
        )
        self.temperature = float(
            temperature
        )
        self.contrast_weight = float(
            contrast_weight
        )
        self.norm_pix_loss = bool(
            norm_pix_loss
        )
        self.queue_size = int(
            queue_size
        )

        h, w = self.patch_grid

        if (
            h % self.mask_window != 0
            or w % self.mask_window != 0
        ):
            raise ValueError(
                "mask_window must divide the "
                f"patch grid {self.patch_grid}."
            )

        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                1,
                self.embed_dim,
            )
        )
        nn.init.normal_(
            self.mask_token,
            std=0.02,
        )

        self.pixel_decoder = PixelDecoder(
            in_dim=self.num_features,
            decoder_dim=decoder_dim,
            patch_size=self.patch_size,
        )

        self.feature_decoder = FeatureDecoder(
            dim=self.num_features,
        )

        self.projector = ProjectionMLP(
            in_dim=self.num_features,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
        )

        self.momentum_projector = copy.deepcopy(
            self.projector
        )

        for parameter in (
            self.momentum_encoder.parameters()
        ):
            parameter.requires_grad = False

        for parameter in (
            self.momentum_projector.parameters()
        ):
            parameter.requires_grad = False

        queue = torch.randn(
            projection_dim,
            self.queue_size,
        )
        queue = F.normalize(
            queue,
            dim=0,
        )

        self.register_buffer(
            "queue",
            queue,
        )
        self.register_buffer(
            "queue_ptr",
            torch.zeros(
                1,
                dtype=torch.long,
            ),
        )

    def patchify(
        self,
        images: torch.Tensor,
    ):
        p = self.patch_size
        b, c, h, w = images.shape

        if (
            h % p != 0
            or w % p != 0
        ):
            raise ValueError(
                "Image size must be divisible "
                f"by patch_size={p}."
            )

        gh = h // p
        gw = w // p

        x = images.reshape(
            b,
            c,
            gh,
            p,
            gw,
            p,
        )

        x = (
            x.permute(
                0, 2, 4, 3, 5, 1
            )
            .contiguous()
        )

        return x.reshape(
            b,
            gh * gw,
            p * p * c,
        )

    def window_mask(
        self,
        batch_size: int,
        device,
    ):
        h, w = self.patch_grid
        r = self.mask_window

        coarse_h = h // r
        coarse_w = w // r
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
            dtype=torch.float32,
            device=device,
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
                r,
                dim=1,
            )
            .repeat_interleave(
                r,
                dim=2,
            )
        )

        return mask.reshape(
            batch_size,
            h * w,
        )

    def encode_online_masked(
        self,
        images,
    ):
        x = (
            self.online_encoder
            .patch_embed(images)
        )

        mask = self.window_mask(
            batch_size=images.shape[0],
            device=images.device,
        )

        mask_for_x = _ensure_mask_shape(
            mask,
            x,
            self.patch_grid,
        )

        token = _mask_token_for(
            self.mask_token,
            x,
        ).to(
            dtype=x.dtype,
            device=x.device,
        )

        x = torch.where(
            mask_for_x.bool(),
            token,
            x,
        )

        latent = (
            _forward_from_patch_embeddings(
                self.online_encoder,
                x,
            )
        )

        return latent, mask

    @torch.no_grad()
    def encode_momentum_full(
        self,
        images,
    ):
        return _full_encoder_features(
            self.momentum_encoder,
            images,
        )

    def reconstruction_loss(
        self,
        images,
        pred,
        mask,
    ):
        target = self.patchify(images)

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
                target - mean
            ) / torch.sqrt(
                var + 1e-6
            )

        per_patch = (
            (pred - target)
            .pow(2)
            .mean(dim=-1)
        )

        return (
            (per_patch * mask).sum()
            / mask.sum().clamp_min(1.0)
        )

    def contrastive_loss(
        self,
        q,
        k,
    ):
        """
        MoCo-style InfoNCE:
          positive = matching momentum key
          negatives = persistent feature queue
        """
        q = F.normalize(
            q,
            dim=1,
        )
        k = F.normalize(
            k,
            dim=1,
        )

        positive = torch.einsum(
            "nc,nc->n",
            [q, k],
        ).unsqueeze(1)

        negative = torch.einsum(
            "nc,ck->nk",
            [
                q,
                self.queue
                .detach()
                .clone(),
            ],
        )

        logits = torch.cat(
            [positive, negative],
            dim=1,
        )
        logits = (
            logits
            / self.temperature
        )

        labels = torch.zeros(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device,
        )

        loss = F.cross_entropy(
            logits,
            labels,
        )

        return loss, q, k

    @torch.no_grad()
    def momentum_update(
        self,
        momentum: float,
    ):
        momentum = float(momentum)

        for online, target in zip(
            self.online_encoder.parameters(),
            self.momentum_encoder.parameters(),
        ):
            target.data.mul_(
                momentum
            ).add_(
                online.data,
                alpha=1.0 - momentum,
            )

        for online, target in zip(
            self.projector.parameters(),
            self.momentum_projector.parameters(),
        ):
            target.data.mul_(
                momentum
            ).add_(
                online.data,
                alpha=1.0 - momentum,
            )

    @torch.no_grad()
    def dequeue_and_enqueue(
        self,
        keys,
    ):
        keys = F.normalize(
            keys,
            dim=1,
        )

        batch_size = int(
            keys.shape[0]
        )

        ptr = int(
            self.queue_ptr.item()
        )

        if batch_size >= self.queue_size:
            keys = keys[
                -self.queue_size :
            ]
            self.queue.copy_(
                keys.T
            )
            self.queue_ptr.zero_()
            return

        end = ptr + batch_size

        if end <= self.queue_size:
            self.queue[
                :,
                ptr:end,
            ] = keys.T
        else:
            first = (
                self.queue_size - ptr
            )
            self.queue[
                :,
                ptr:,
            ] = keys[
                :first
            ].T

            remainder = (
                batch_size - first
            )
            self.queue[
                :,
                :remainder,
            ] = keys[
                first:
            ].T

        self.queue_ptr[0] = (
            ptr + batch_size
        ) % self.queue_size

    def forward(
        self,
        online_images,
        target_images,
    ):
        latent_online, mask = (
            self.encode_online_masked(
                online_images
            )
        )

        pred_pixel = (
            self.pixel_decoder(
                latent_online,
                patch_grid=self.patch_grid,
            )
        )

        loss_mae = (
            self.reconstruction_loss(
                online_images,
                pred_pixel,
                mask,
            )
        )

        feature_online = (
            self.feature_decoder(
                latent_online
            )
        )
        q = self.projector(
            feature_online
        )

        with torch.no_grad():
            latent_target = (
                self.encode_momentum_full(
                    target_images
                )
            )
            feature_target = (
                _pool_swin_features(
                    latent_target
                )
            )
            k = (
                self.momentum_projector(
                    feature_target
                )
            )

        (
            loss_cl,
            q_norm,
            k_norm,
        ) = self.contrastive_loss(
            q,
            k,
        )

        total_loss = (
            loss_mae
            + self.contrast_weight
            * loss_cl
        )

        with torch.no_grad():
            self.dequeue_and_enqueue(
                k_norm
            )

        return {
            "loss": total_loss,
            "loss_mae": loss_mae,
            "loss_cl": loss_cl,
            "mask_ratio_actual": (
                mask.float().mean()
            ),
            "q_norm": (
                q_norm.norm(
                    dim=1
                ).mean()
            ),
            "k_norm": (
                k_norm.norm(
                    dim=1
                ).mean()
            ),
        }


class SwinCMAEClassifier(nn.Module):
    """
    Fatigue/nofatigue classifier initialized from the online Swin-B
    encoder of a Swin-CMAE pretraining checkpoint.
    """

    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 2,
        drop_rate: float = 0.10,
        drop_path_rate: float = 0.10,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if not pretrained_path:
            raise ValueError(
                "Swin-CMAE fine-tuning requires "
                "--pretrained pointing to a CMAE checkpoint."
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
                "\nSwin-CMAE checkpoint not found:\n"
                f"{pretrained_path}\n"
            )

        (
            self.backbone,
            self.model_name,
        ) = _create_swin_base(
            drop_path_rate=drop_path_rate,
        )

        self.num_features = int(
            self.backbone.num_features
        )

        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )

        encoder_state = (
            _extract_online_encoder(
                checkpoint
            )
        )

        coverage, matched_keys = (
            _parameter_coverage(
                encoder_state,
                self.backbone,
            )
        )

        print(
            "=" * 70,
            flush=True,
        )
        print(
            "[Swin-CMAE] Loading online Swin-B encoder",
            flush=True,
        )
        print(
            f"[Swin-CMAE] Checkpoint: "
            f"{pretrained_path}",
            flush=True,
        )
        print(
            "[Swin-CMAE] Encoder parameter coverage: "
            f"{coverage * 100.0:.2f}%",
            flush=True,
        )

        if coverage < 0.95:
            missing = [
                key
                for key in self.backbone.state_dict()
                if (
                    key not in matched_keys
                    and "relative_position_index"
                    not in key
                    and "attn_mask"
                    not in key
                )
            ][:20]

            raise RuntimeError(
                "Swin-CMAE encoder coverage is too low. "
                f"Coverage={coverage * 100.0:.2f}%. "
                f"First unmatched keys: {missing}"
            )

        msg = self.backbone.load_state_dict(
            encoder_state,
            strict=False,
        )

        missing_filtered = [
            key
            for key in msg.missing_keys
            if (
                "relative_position_index"
                not in key
                and "attn_mask"
                not in key
            )
        ]

        print(
            "[Swin-CMAE] Missing non-buffer keys: "
            f"{len(missing_filtered)}",
            flush=True,
        )
        print(
            "[Swin-CMAE] Unexpected keys: "
            f"{len(msg.unexpected_keys)}",
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

    def forward_features(
        self,
        images,
    ):
        features = (
            self.backbone
            .forward_features(images)
        )

        if hasattr(
            self.backbone,
            "forward_head",
        ):
            return (
                self.backbone
                .forward_head(
                    features,
                    pre_logits=True,
                )
            )

        return _pool_swin_features(
            features
        )

    def forward(
        self,
        images,
    ):
        return self.classifier(
            self.forward_features(
                images
            )
        )


def build_swin_cmae(
    config=None,
    num_classes: Optional[int] = None,
    model_name: str = "swin_base_patch4_window7_224",
    drop_rate: float = 0.10,
    drop_path_rate: float = 0.10,
    freeze_backbone: bool = False,
    pretrained_path: Optional[str] = None,
):
    # model_name is accepted for CLI compatibility; this implementation
    # intentionally fixes the architecture to Swin-B/4/7/224.
    del model_name

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

    return SwinCMAEClassifier(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
    )
