import os

import torch
import torch.nn as nn
import timm


CONVNEXT_WEIGHT_DIR = "/root/shared-nvme/uploads"

SUPPORTED_CONVNEXT_MODELS = {
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
}


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unexpected ConvNeXt checkpoint format. "
            "Expected a state_dict dictionary."
        )

    # Common checkpoint wrappers.
    for key in ("model", "state_dict", "model_ema"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    return checkpoint


def _clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        clean_key = key

        # Common wrappers added by DDP/training frameworks.
        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        # Drop the original ImageNet classifier.
        # Keep head.norm because it is part of the pretrained representation.
        if (
            clean_key.startswith("head.fc.")
            or clean_key.startswith("head.classifier.")
            or clean_key.startswith("classifier.")
            or clean_key.startswith("fc.")
            or clean_key in {"head.weight", "head.bias"}
        ):
            continue

        cleaned[clean_key] = value

    return cleaned


def _resolve_pretrained_path(model_name, pretrained_path=None):
    if pretrained_path:
        pretrained_path = os.path.expanduser(str(pretrained_path))
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                "\nConvNeXt pretrained weight not found:\n"
                f"{pretrained_path}\n"
            )
        return pretrained_path

    candidates = [
        os.path.join(CONVNEXT_WEIGHT_DIR, f"{model_name}.pth"),
        os.path.join(CONVNEXT_WEIGHT_DIR, f"{model_name}_1k_224.pth"),
        os.path.join(CONVNEXT_WEIGHT_DIR, f"{model_name}_1k_224_ema.pth"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "\nConvNeXt pretrained weight not found.\n"
        "Pass the checkpoint with --pretrained, or place one of these files:\n"
        + "\n".join(candidates)
        + "\n"
    )


class ConvNeXtClassifier(nn.Module):
    """
    Local pretrained ConvNeXt backbone + binary/multi-class classifier.

    The backbone is created with timm and pretrained weights are loaded
    from a local checkpoint. The original ImageNet classifier is ignored,
    then a new task-specific classifier is initialized.
    """

    def __init__(
        self,
        model_name="convnext_tiny",
        num_classes=2,
        drop_rate=0.2,
        freeze_backbone=False,
        pretrained_path=None,
    ):
        super().__init__()

        model_name = str(model_name).lower()

        if model_name not in SUPPORTED_CONVNEXT_MODELS:
            raise ValueError(
                f"Unsupported ConvNeXt model: {model_name}. "
                f"Available: {sorted(SUPPORTED_CONVNEXT_MODELS)}"
            )

        self.model_name = model_name

        weight_path = _resolve_pretrained_path(
            model_name=model_name,
            pretrained_path=pretrained_path,
        )

        print("=" * 70, flush=True)
        print("[ConvNeXt] Loading local pretrained model", flush=True)
        print(f"[ConvNeXt] Model: {model_name}", flush=True)
        print(f"[ConvNeXt] Weight path: {weight_path}", flush=True)

        # num_classes=0 makes timm return representation features rather than
        # ImageNet logits. The new classifier below is task-specific.
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
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

        print("[ConvNeXt] Local pretrained weights loaded", flush=True)
        print(
            f"[ConvNeXt] Missing keys: {len(msg.missing_keys)}",
            flush=True,
        )
        print(
            f"[ConvNeXt] Unexpected keys: {len(msg.unexpected_keys)}",
            flush=True,
        )

        if msg.missing_keys:
            print(
                "[ConvNeXt] First missing keys:",
                msg.missing_keys[:10],
                flush=True,
            )
        if msg.unexpected_keys:
            print(
                "[ConvNeXt] First unexpected keys:",
                msg.unexpected_keys[:10],
                flush=True,
            )

        print("=" * 70, flush=True)

        if freeze_backbone:
            print("[ConvNeXt] Backbone frozen", flush=True)
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
            "[ConvNeXt] Classification head created: "
            f"in_features={self.num_features}, "
            f"num_classes={num_classes}, "
            f"drop_rate={drop_rate}",
            flush=True,
        )

    def forward_features(self, images):
        features = self.backbone.forward_features(images)

        # timm ConvNeXt normally exposes forward_head(..., pre_logits=True),
        # which applies the pretrained final normalization + global pooling.
        if hasattr(self.backbone, "forward_head"):
            features = self.backbone.forward_head(
                features,
                pre_logits=True,
            )
        elif features.ndim == 4:
            features = features.mean(dim=(2, 3))

        return features

    def forward(self, images):
        features = self.forward_features(images)
        logits = self.classifier(features)
        return logits


def build_convnext(
    config=None,
    num_classes=None,
    model_name="convnext_tiny",
    drop_rate=0.2,
    freeze_backbone=False,
    pretrained_path=None,
):
    """
    Supports both:
        build_convnext(config)
    and:
        build_convnext(
            num_classes=2,
            model_name="convnext_tiny",
            ...
        )
    """

    if config is not None and hasattr(config, "MODEL"):
        if num_classes is None:
            num_classes = int(config.MODEL.NUM_CLASSES)

        if hasattr(config.MODEL, "DROP_RATE"):
            drop_rate = float(config.MODEL.DROP_RATE)

        if hasattr(config.MODEL, "CONVNEXT_MODEL"):
            model_name = str(config.MODEL.CONVNEXT_MODEL)

        if (
            pretrained_path is None
            and hasattr(config.MODEL, "PRETRAINED")
            and config.MODEL.PRETRAINED
        ):
            pretrained_path = str(config.MODEL.PRETRAINED)

    if num_classes is None:
        num_classes = 2

    return ConvNeXtClassifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        freeze_backbone=freeze_backbone,
        pretrained_path=pretrained_path,
    )
