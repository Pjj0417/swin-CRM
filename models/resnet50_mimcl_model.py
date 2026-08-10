import copy
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def _make_resnet50():
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    return model


def _forward_resnet_map(
    backbone: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    TorchVision ResNet-50 feature map before global average pooling:
    [B, 2048, 7, 7] for 224x224 input.
    """
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x


def _pool_feature_map(
    x: torch.Tensor,
) -> torch.Tensor:
    return F.adaptive_avg_pool2d(
        x,
        output_size=1,
    ).flatten(1)


def _extract_checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "ResNet50-MIMCL checkpoint must be a dict."
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


def _extract_online_encoder(checkpoint):
    """
    Extract only online ResNet-50 encoder weights from a pretraining
    checkpoint produced by the paired training script.
    """
    state = _extract_checkpoint_state(
        checkpoint
    )

    prefixes = (
        "online_encoder.",
        "module.online_encoder.",
        "model.online_encoder.",
    )
    has_online_prefix = any(
        any(
            key.startswith(prefix)
            for prefix in prefixes
        )
        for key in state
    )

    encoder = {}

    for key, value in state.items():
        clean = key

        if clean.startswith("module."):
            clean = clean[len("module."):]

        if clean.startswith("model."):
            clean = clean[len("model."):]

        if has_online_prefix:
            if not clean.startswith(
                "online_encoder."
            ):
                continue
            clean = clean[
                len("online_encoder.") :
            ]

        if (
            clean.startswith("pixel_decoder.")
            or clean.startswith("feature_decoder.")
            or clean.startswith("projector.")
            or clean.startswith("momentum_")
            or clean.startswith("classifier.")
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
        total_numel += tensor.numel()

        other = state_dict.get(key)
        if (
            other is not None
            and tuple(other.shape)
            == tuple(tensor.shape)
        ):
            matched_numel += tensor.numel()
            matched_keys.add(key)

    return (
        matched_numel / max(1, total_numel),
        matched_keys,
    )


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 2048,
        out_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                in_dim,
                hidden_dim,
            ),
            nn.BatchNorm1d(
                hidden_dim
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                hidden_dim,
                out_dim,
            ),
        )

    def forward(self, x):
        return self.net(x)


class FeatureDecoder(nn.Module):
    """
    Small feature-completion head for the masked online branch before
    contrastive projection.
    """

    def __init__(
        self,
        dim: int = 2048,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class PixelDecoder(nn.Module):
    """
    ResNet-50 final feature map:
        7x7x2048
    Decoder:
        7 -> 14 -> 28 -> 56 -> 112 -> 224
    Output:
        RGB reconstruction in [0, 1].
    """

    def __init__(
        self,
        in_dim: int = 2048,
        decoder_dim: int = 256,
    ):
        super().__init__()

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

            nn.ConvTranspose2d(
                decoder_dim // 2,
                decoder_dim // 4,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),

            nn.ConvTranspose2d(
                decoder_dim // 4,
                3,
                kernel_size=2,
                stride=2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class ResNet50MIMCLPretrain(nn.Module):
    """
    ResNet50 + Masked Image Modeling + Contrastive Learning.

    Online branch:
        block-masked normalized image
          -> ResNet50
          -> pixel decoder -> masked RGB reconstruction loss
          -> feature decoder + projector -> contrastive query

    Momentum branch:
        weak positive view
          -> EMA ResNet50
          -> EMA projector -> contrastive key

    For single-GPU / small-batch training, a MoCo-style queue is used
    to provide sufficient negatives.
    """

    def __init__(
        self,
        mask_ratio: float = 0.60,
        mask_patch_size: int = 32,
        decoder_dim: int = 256,
        projection_dim: int = 256,
        projection_hidden_dim: int = 2048,
        temperature: float = 0.20,
        contrast_weight: float = 0.10,
        queue_size: int = 4096,
    ):
        super().__init__()

        self.model_name = "resnet50"

        self.online_encoder = (
            _make_resnet50()
        )
        self.momentum_encoder = (
            copy.deepcopy(
                self.online_encoder
            )
        )

        self.num_features = 2048
        self.mask_ratio = float(
            mask_ratio
        )
        self.mask_patch_size = int(
            mask_patch_size
        )
        self.temperature = float(
            temperature
        )
        self.contrast_weight = float(
            contrast_weight
        )
        self.queue_size = int(
            queue_size
        )

        if 224 % self.mask_patch_size != 0:
            raise ValueError(
                "mask_patch_size must divide 224 exactly. "
                f"Got {self.mask_patch_size}."
            )

        self.pixel_decoder = PixelDecoder(
            in_dim=self.num_features,
            decoder_dim=decoder_dim,
        )

        self.feature_decoder = (
            FeatureDecoder(
                dim=self.num_features
            )
        )

        self.projector = ProjectionMLP(
            in_dim=self.num_features,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
        )

        self.momentum_projector = (
            copy.deepcopy(
                self.projector
            )
        )

        for parameter in (
            self.momentum_encoder
            .parameters()
        ):
            parameter.requires_grad = False

        for parameter in (
            self.momentum_projector
            .parameters()
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

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

    def block_mask(
        self,
        batch_size: int,
        device,
    ):
        """
        Returns:
          coarse_mask: [B,1,G,G]
          pixel_mask:  [B,1,224,224]

        1 = masked, 0 = visible.
        """
        grid = (
            224
            // self.mask_patch_size
        )
        total_blocks = (
            grid * grid
        )

        num_mask = int(
            round(
                total_blocks
                * self.mask_ratio
            )
        )
        num_mask = min(
            max(num_mask, 1),
            total_blocks - 1,
        )

        noise = torch.rand(
            batch_size,
            total_blocks,
            device=device,
        )
        ids = torch.argsort(
            noise,
            dim=1,
        )

        mask = torch.zeros(
            batch_size,
            total_blocks,
            device=device,
            dtype=torch.float32,
        )
        mask.scatter_(
            1,
            ids[:, :num_mask],
            1.0,
        )

        coarse_mask = mask.view(
            batch_size,
            1,
            grid,
            grid,
        )

        pixel_mask = (
            coarse_mask
            .repeat_interleave(
                self.mask_patch_size,
                dim=2,
            )
            .repeat_interleave(
                self.mask_patch_size,
                dim=3,
            )
        )

        return (
            coarse_mask,
            pixel_mask,
        )

    def denormalize(
        self,
        images,
    ):
        return (
            images
            * self.imagenet_std
            + self.imagenet_mean
        ).clamp(
            0.0,
            1.0,
        )

    def reconstruction_loss(
        self,
        normalized_images,
        prediction,
        pixel_mask,
    ):
        """
        SimMIM-style direct RGB regression.
        L1 loss is computed only over masked pixels.
        """
        target = self.denormalize(
            normalized_images
        )

        if prediction.shape[-2:] != (
            224,
            224,
        ):
            prediction = (
                F.interpolate(
                    prediction,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        per_pixel = (
            prediction - target
        ).abs().mean(
            dim=1,
            keepdim=True,
        )

        return (
            (per_pixel * pixel_mask)
            .sum()
            / pixel_mask.sum()
            .clamp_min(1.0)
        )

    def contrastive_loss(
        self,
        q,
        k,
    ):
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
            [
                positive,
                negative,
            ],
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
        momentum = float(
            momentum
        )

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
            self.queue.copy_(
                keys[
                    -self.queue_size :
                ].T
            )
            self.queue_ptr.zero_()
            return

        end = (
            ptr + batch_size
        )

        if end <= self.queue_size:
            self.queue[
                :,
                ptr:end,
            ] = keys.T
        else:
            first = (
                self.queue_size
                - ptr
            )
            self.queue[
                :,
                ptr:,
            ] = keys[
                :first
            ].T

            remain = (
                batch_size
                - first
            )
            self.queue[
                :,
                :remain,
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
        (
            coarse_mask,
            pixel_mask,
        ) = self.block_mask(
            batch_size=(
                online_images
                .shape[0]
            ),
            device=(
                online_images
                .device
            ),
        )

        # Normalized zero corresponds approximately to mean RGB,
        # providing a neutral fill for masked image regions.
        masked_images = (
            online_images
            * (
                1.0
                - pixel_mask
            )
        )

        online_map = (
            _forward_resnet_map(
                self.online_encoder,
                masked_images,
            )
        )

        reconstruction = (
            self.pixel_decoder(
                online_map
            )
        )

        loss_mim = (
            self.reconstruction_loss(
                normalized_images=(
                    online_images
                ),
                prediction=(
                    reconstruction
                ),
                pixel_mask=(
                    pixel_mask
                ),
            )
        )

        online_feature = (
            _pool_feature_map(
                online_map
            )
        )
        online_feature = (
            self.feature_decoder(
                online_feature
            )
        )
        q = self.projector(
            online_feature
        )

        with torch.no_grad():
            target_map = (
                _forward_resnet_map(
                    self.momentum_encoder,
                    target_images,
                )
            )
            target_feature = (
                _pool_feature_map(
                    target_map
                )
            )
            k = (
                self.momentum_projector(
                    target_feature
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
            loss_mim
            + self.contrast_weight
            * loss_cl
        )

        with torch.no_grad():
            self.dequeue_and_enqueue(
                k_norm
            )

        return {
            "loss": total_loss,
            "loss_mim": loss_mim,
            "loss_mae": loss_mim,
            "loss_cl": loss_cl,
            "mask_ratio_actual": (
                coarse_mask
                .float()
                .mean()
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


class ResNet50MIMCLClassifier(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 2,
        drop_rate: float = 0.10,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if not pretrained_path:
            raise ValueError(
                "ResNet50-MIMCL fine-tuning requires "
                "--pretrained pointing to the MIM+CL checkpoint."
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
                "\nResNet50-MIMCL checkpoint not found:\n"
                f"{pretrained_path}\n"
            )

        self.backbone = (
            _make_resnet50()
        )
        self.num_features = 2048

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

        (
            coverage,
            matched_keys,
        ) = _parameter_coverage(
            encoder_state,
            self.backbone,
        )

        print(
            "=" * 70,
            flush=True,
        )
        print(
            "[ResNet50-MIMCL] Loading online ResNet-50 encoder",
            flush=True,
        )
        print(
            "[ResNet50-MIMCL] Checkpoint: "
            f"{pretrained_path}",
            flush=True,
        )
        print(
            "[ResNet50-MIMCL] Encoder parameter coverage: "
            f"{coverage * 100.0:.2f}%",
            flush=True,
        )

        if coverage < 0.95:
            missing = [
                key
                for key
                in self.backbone.state_dict()
                if key not in matched_keys
            ][:20]

            raise RuntimeError(
                "ResNet50-MIMCL encoder coverage is too low. "
                f"Coverage={coverage * 100.0:.2f}%. "
                f"First unmatched keys: {missing}"
            )

        msg = (
            self.backbone
            .load_state_dict(
                encoder_state,
                strict=False,
            )
        )

        print(
            "[ResNet50-MIMCL] Missing keys: "
            f"{len(msg.missing_keys)}",
            flush=True,
        )
        print(
            "[ResNet50-MIMCL] Unexpected keys: "
            f"{len(msg.unexpected_keys)}",
            flush=True,
        )
        print(
            "=" * 70,
            flush=True,
        )

        if freeze_backbone:
            for parameter in (
                self.backbone
                .parameters()
            ):
                parameter.requires_grad = False

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
        images,
    ):
        feature_map = (
            _forward_resnet_map(
                self.backbone,
                images,
            )
        )
        return _pool_feature_map(
            feature_map
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


def build_resnet50_mimcl(
    config=None,
    num_classes: Optional[int] = None,
    model_name: str = "resnet50",
    drop_rate: float = 0.10,
    drop_path_rate: float = 0.0,
    freeze_backbone: bool = False,
    pretrained_path: Optional[str] = None,
):
    # Accepted for compatibility with the unified training CLI.
    del model_name
    del drop_path_rate

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

    return ResNet50MIMCLClassifier(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        drop_rate=drop_rate,
        freeze_backbone=freeze_backbone,
    )
