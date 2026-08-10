import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


SWIN_MAE_MODEL_NAME = "swin_base_patch4_window7_224"


def _to_bhwc(x: torch.Tensor, embed_dim: int, grid_size: Tuple[int, int]):
    if x.ndim == 4:
        if x.shape[1] == embed_dim and x.shape[-1] != embed_dim:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x

    if x.ndim == 3:
        b, n, c = x.shape
        h, w = grid_size
        if n != h * w:
            raise RuntimeError(
                f"Cannot reshape patch tokens: n={n}, grid={grid_size}"
            )
        return x.reshape(b, h, w, c)

    raise RuntimeError(
        f"Unexpected patch embedding shape: {tuple(x.shape)}"
    )


def _extract_checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    for key in ("model", "state_dict", "model_ema"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def _extract_mae_encoder_state(checkpoint):
    state = _extract_checkpoint_state(checkpoint)
    encoder = {}

    has_backbone_prefix = any(
        key.startswith("backbone.")
        or key.startswith("module.backbone.")
        or key.startswith("model.backbone.")
        for key in state
    )

    for key, value in state.items():
        clean = key

        if clean.startswith("module."):
            clean = clean[len("module."):]
        if clean.startswith("model."):
            clean = clean[len("model."):]

        if has_backbone_prefix:
            if not clean.startswith("backbone."):
                continue
            clean = clean[len("backbone."):]

        if (
            clean.startswith("head.fc.")
            or clean.startswith("classifier.")
            or clean.startswith("decoder.")
            or clean.startswith("decoder_")
            or clean == "mask_token"
        ):
            continue

        encoder[clean] = value

    return encoder


def _parameter_coverage(state_dict, model):
    model_state = model.state_dict()
    total = 0
    matched = 0
    matched_keys = set()

    for key, tensor in model_state.items():
        total += tensor.numel()
        other = state_dict.get(key)
        if other is not None and tuple(other.shape) == tuple(tensor.shape):
            matched += tensor.numel()
            matched_keys.add(key)

    return matched / max(1, total), matched_keys


class SwinMAEPretrain(nn.Module):
    """
    Swin-B masked autoencoder-style pretraining model.

    224x224 input -> 4x4 patch embedding -> grouped/window masking
    -> Swin-B hierarchical encoder -> lightweight asymmetric decoder
    -> RGB patch reconstruction.

    The regular Swin spatial grid is preserved; masked positions are
    replaced by a learnable mask token. Loss is computed only on masked
    patches.
    """

    def __init__(
        self,
        model_name: str = SWIN_MAE_MODEL_NAME,
        img_size: int = 224,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        mask_window: int = 4,
        decoder_dim: int = 256,
        norm_pix_loss: bool = True,
        drop_path_rate: float = 0.10,
    ):
        super().__init__()

        if model_name != SWIN_MAE_MODEL_NAME:
            raise ValueError(
                f"Only {SWIN_MAE_MODEL_NAME} is configured."
            )
        if img_size != 224 or patch_size != 4:
            raise ValueError(
                "Configured for img_size=224 and patch_size=4."
            )

        self.model_name = model_name
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.mask_window = int(mask_window)
        self.norm_pix_loss = bool(norm_pix_loss)

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=float(drop_path_rate),
        )

        self.embed_dim = int(self.backbone.embed_dim)
        self.num_features = int(self.backbone.num_features)

        grid_size = getattr(
            self.backbone.patch_embed,
            "grid_size",
            (img_size // patch_size, img_size // patch_size),
        )
        self.patch_grid = (
            int(grid_size[0]),
            int(grid_size[1]),
        )

        if (
            self.patch_grid[0] % self.mask_window != 0
            or self.patch_grid[1] % self.mask_window != 0
        ):
            raise ValueError(
                "mask_window must divide the patch grid exactly: "
                f"grid={self.patch_grid}, window={self.mask_window}"
            )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, self.embed_dim)
        )
        nn.init.normal_(self.mask_token, std=0.02)

        # Swin-B final spatial map is 7x7. Upsample to 56x56 patch grid.
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
                patch_size * patch_size * 3,
                kernel_size=1,
            ),
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, c, h, w = images.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"Expected {self.img_size}x{self.img_size}, got {h}x{w}."
            )

        gh, gw = h // p, w // p
        x = images.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.reshape(b, gh * gw, p * p * c)

    def window_mask(self, batch_size: int, device: torch.device):
        gh, gw = self.patch_grid
        r = self.mask_window
        coarse_h = gh // r
        coarse_w = gw // r
        coarse_n = coarse_h * coarse_w

        num_mask = int(round(coarse_n * self.mask_ratio))
        num_mask = min(max(num_mask, 1), coarse_n - 1)

        noise = torch.rand(batch_size, coarse_n, device=device)
        ids = torch.argsort(noise, dim=1)

        coarse_mask = torch.zeros(
            batch_size,
            coarse_n,
            device=device,
            dtype=torch.float32,
        )
        coarse_mask.scatter_(1, ids[:, :num_mask], 1.0)
        coarse_mask = coarse_mask.reshape(
            batch_size,
            coarse_h,
            coarse_w,
        )

        mask = coarse_mask.repeat_interleave(
            r, dim=1
        ).repeat_interleave(
            r, dim=2
        )
        return mask.reshape(batch_size, gh * gw)

    def forward_encoder(self, images: torch.Tensor):
        x = self.backbone.patch_embed(images)
        x = _to_bhwc(
            x,
            embed_dim=self.embed_dim,
            grid_size=self.patch_grid,
        )

        b, h, w, _ = x.shape
        mask = self.window_mask(
            batch_size=b,
            device=x.device,
        ).reshape(b, h, w, 1)

        x = torch.where(
            mask.bool(),
            self.mask_token.to(dtype=x.dtype),
            x,
        )

        x = self.backbone.layers(x)
        x = self.backbone.norm(x)
        return x, mask.reshape(b, h * w)

    def forward_decoder(self, latent: torch.Tensor):
        if latent.ndim != 4:
            raise RuntimeError(
                f"Expected BHWC latent, got {tuple(latent.shape)}"
            )

        x = latent.permute(0, 3, 1, 2).contiguous()
        x = self.decoder(x)

        gh, gw = self.patch_grid
        if x.shape[-2:] != (gh, gw):
            x = F.interpolate(
                x,
                size=(gh, gw),
                mode="bilinear",
                align_corners=False,
            )

        return (
            x.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(
                x.shape[0],
                gh * gw,
                self.patch_size * self.patch_size * 3,
            )
        )

    def forward_loss(
        self,
        images: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ):
        target = self.patchify(images)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(
                dim=-1,
                keepdim=True,
                unbiased=False,
            )
            target = (target - mean) / torch.sqrt(var + 1e-6)

        loss = (pred - target).pow(2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, images: torch.Tensor):
        latent, mask = self.forward_encoder(images)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(images, pred, mask)
        return loss, pred, mask


class SwinMAEClassifier(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        num_classes: int = 2,
        model_name: str = SWIN_MAE_MODEL_NAME,
        drop_rate: float = 0.10,
        drop_path_rate: float = 0.10,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if model_name != SWIN_MAE_MODEL_NAME:
            raise ValueError(
                f"Only {SWIN_MAE_MODEL_NAME} is supported."
            )
        if not pretrained_path:
            raise ValueError(
                "Pass the MAE checkpoint with --pretrained."
            )

        pretrained_path = os.path.expanduser(str(pretrained_path))
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                "\nSwin-MAE checkpoint not found:\n"
                f"{pretrained_path}\n"
            )

        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=float(drop_path_rate),
        )
        self.num_features = int(self.backbone.num_features)

        checkpoint = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )
        encoder_state = _extract_mae_encoder_state(checkpoint)

        coverage, matched_keys = _parameter_coverage(
            encoder_state,
            self.backbone,
        )

        print("=" * 70, flush=True)
        print("[Swin-MAE] Loading MAE-pretrained Swin-B encoder", flush=True)
        print(f"[Swin-MAE] Checkpoint: {pretrained_path}", flush=True)
        print(
            f"[Swin-MAE] Encoder parameter coverage: "
            f"{coverage * 100.0:.2f}%",
            flush=True,
        )

        if coverage < 0.95:
            missing_preview = [
                key
                for key in self.backbone.state_dict().keys()
                if key not in matched_keys
            ][:20]
            raise RuntimeError(
                "Swin-MAE encoder checkpoint coverage too low. "
                f"Coverage={coverage * 100.0:.2f}%. "
                f"First unmatched keys: {missing_preview}"
            )

        msg = self.backbone.load_state_dict(
            encoder_state,
            strict=False,
        )
        print(
            f"[Swin-MAE] Missing keys: {len(msg.missing_keys)}",
            flush=True,
        )
        print(
            f"[Swin-MAE] Unexpected keys: {len(msg.unexpected_keys)}",
            flush=True,
        )
        print("=" * 70, flush=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if float(drop_rate) > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(float(drop_rate)),
                nn.Linear(self.num_features, int(num_classes)),
            )
        else:
            self.classifier = nn.Linear(
                self.num_features,
                int(num_classes),
            )

        linear = (
            self.classifier[-1]
            if isinstance(self.classifier, nn.Sequential)
            else self.classifier
        )
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear.bias)

    def forward_features(self, images: torch.Tensor):
        x = self.backbone.forward_features(images)
        return self.backbone.forward_head(
            x,
            pre_logits=True,
        )

    def forward(self, images: torch.Tensor):
        return self.classifier(self.forward_features(images))


def build_swin_mae(
    config=None,
    num_classes: Optional[int] = None,
    model_name: str = SWIN_MAE_MODEL_NAME,
    drop_rate: float = 0.10,
    drop_path_rate: float = 0.10,
    freeze_backbone: bool = False,
    pretrained_path: Optional[str] = None,
):
    if config is not None and hasattr(config, "MODEL"):
        if num_classes is None:
            num_classes = int(config.MODEL.NUM_CLASSES)
        if hasattr(config.MODEL, "DROP_RATE"):
            drop_rate = float(config.MODEL.DROP_RATE)
        if hasattr(config.MODEL, "DROP_PATH_RATE"):
            drop_path_rate = float(config.MODEL.DROP_PATH_RATE)
        if (
            pretrained_path is None
            and hasattr(config, "PRETRAINED")
            and config.PRETRAINED
        ):
            pretrained_path = str(config.PRETRAINED)

    if num_classes is None:
        num_classes = 2

    return SwinMAEClassifier(
        pretrained_path=pretrained_path,
        num_classes=num_classes,
        model_name=model_name,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
    )
