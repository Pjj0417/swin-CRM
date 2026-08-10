import os

import torch
import torch.nn as nn
import timm


DEIT_WEIGHT_DIR = "/root/shared-nvme/uploads"

SUPPORTED_DEIT_MODELS = {
    "deit_base_patch16_224",
}


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unexpected DeiT checkpoint format. "
            "Expected a state_dict/checkpoint dictionary."
        )

    for key in ("model", "state_dict", "model_ema"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    return checkpoint


def _clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        clean_key = key

        # Common wrappers.
        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        # Remove the original ImageNet-1K classifier.
        # A new fatigue/nofatigue classifier is created below.
        if (
            clean_key.startswith("head.")
            or clean_key.startswith("head_dist.")
            or clean_key.startswith("classifier.")
        ):
            continue

        cleaned[clean_key] = value

    return cleaned


def _resolve_pretrained_path(model_name, pretrained_path=None):
    if pretrained_path:
        pretrained_path = os.path.expanduser(str(pretrained_path))

        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                "\nDeiT pretrained weight not found:\n"
                f"{pretrained_path}\n"
            )

        return pretrained_path

    candidates = []

    if model_name == "deit_base_patch16_224":
        candidates.extend(
            [
                os.path.join(
                    DEIT_WEIGHT_DIR,
                    "deit_base_patch16_224-b5f2ef4d.pth",
                ),
                os.path.join(
                    DEIT_WEIGHT_DIR,
                    "deit_base_patch16_224.pth",
                ),
            ]
        )

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "\nDeiT pretrained weight not found.\n"
        "Pass the official checkpoint with --pretrained.\n"
        "Expected DeiT-B checkpoint, for example:\n"
        f"{os.path.join(DEIT_WEIGHT_DIR, 'deit_base_patch16_224-b5f2ef4d.pth')}\n"
    )


class DeiTClassifier(nn.Module):
    """
    DeiT-Base/16 224 backbone + task-specific classification head.

    The backbone is created through timm without downloading weights.
    Official/local ImageNet-1K pretrained weights are then loaded from disk.
    The original 1000-class ImageNet head is discarded and replaced by a
    new classifier for fatigue / nofatigue.
    """

    def __init__(
        self,
        model_name="deit_base_patch16_224",
        num_classes=2,
        drop_rate=0.2,
        drop_path_rate=0.0,
        freeze_backbone=False,
        pretrained_path=None,
    ):
        super().__init__()

        model_name = str(model_name).lower()

        if model_name not in SUPPORTED_DEIT_MODELS:
            raise ValueError(
                f"Unsupported DeiT model: {model_name}. "
                f"Available: {sorted(SUPPORTED_DEIT_MODELS)}"
            )

        self.model_name = model_name

        weight_path = _resolve_pretrained_path(
            model_name=model_name,
            pretrained_path=pretrained_path,
        )

        print("=" * 70, flush=True)
        print("[DeiT] Loading local pretrained model", flush=True)
        print(f"[DeiT] Model: {model_name}", flush=True)
        print(f"[DeiT] Weight path: {weight_path}", flush=True)
        print(
            f"[DeiT] Drop path rate: {float(drop_path_rate):.4f}",
            flush=True,
        )

        # num_classes=0 makes timm expose representation features rather
        # than the original ImageNet-1K logits.
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=float(drop_path_rate),
        )

        self.num_features = int(self.backbone.num_features)

        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
        )
        checkpoint = _extract_state_dict(checkpoint)
        checkpoint = _clean_state_dict(checkpoint)

        msg = self.backbone.load_state_dict(
            checkpoint,
            strict=False,
        )

        print("[DeiT] Local pretrained weights loaded", flush=True)
        print(
            f"[DeiT] Missing keys: {len(msg.missing_keys)}",
            flush=True,
        )
        print(
            f"[DeiT] Unexpected keys: {len(msg.unexpected_keys)}",
            flush=True,
        )

        if msg.missing_keys:
            print(
                "[DeiT] First missing keys:",
                msg.missing_keys[:10],
                flush=True,
            )

        if msg.unexpected_keys:
            print(
                "[DeiT] First unexpected keys:",
                msg.unexpected_keys[:10],
                flush=True,
            )

        print("=" * 70, flush=True)

        if freeze_backbone:
            print("[DeiT] Backbone frozen", flush=True)

            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if float(drop_rate) > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=float(drop_rate)),
                nn.Linear(
                    self.num_features,
                    int(num_classes),
                ),
            )
        else:
            self.classifier = nn.Linear(
                self.num_features,
                int(num_classes),
            )

        linear_layer = (
            self.classifier[-1]
            if isinstance(self.classifier, nn.Sequential)
            else self.classifier
        )

        nn.init.normal_(
            linear_layer.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(linear_layer.bias)

        print(
            "[DeiT] Classification head created: "
            f"in_features={self.num_features}, "
            f"num_classes={num_classes}, "
            f"drop_rate={drop_rate}",
            flush=True,
        )

    def forward_features(self, images):
        features = self.backbone.forward_features(images)

        # timm DeiT/ViT returns token features from forward_features().
        # forward_head(..., pre_logits=True) extracts the CLS representation.
        if hasattr(self.backbone, "forward_head"):
            features = self.backbone.forward_head(
                features,
                pre_logits=True,
            )
        elif features.ndim == 3:
            features = features[:, 0]

        return features

    def forward(self, images):
        features = self.forward_features(images)
        logits = self.classifier(features)
        return logits


def build_deit(
    config=None,
    num_classes=None,
    model_name="deit_base_patch16_224",
    drop_rate=0.2,
    drop_path_rate=0.0,
    freeze_backbone=False,
    pretrained_path=None,
):
    """
    Supports both:
        build_deit(config)

    and:
        build_deit(
            num_classes=2,
            model_name="deit_base_patch16_224",
            ...
        )
    """

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

    return DeiTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
        pretrained_path=pretrained_path,
    )
