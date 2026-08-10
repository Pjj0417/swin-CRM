import os

import torch
import torch.nn as nn
import timm

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


WEIGHT_DIR = "/root/shared-nvme/uploads"

SOURCE_MEAN = (0.485, 0.456, 0.406)
SOURCE_STD = (0.229, 0.224, 0.225)
TARGET_MEAN = (0.0, 0.0, 0.0)
TARGET_STD = (1.0, 1.0, 1.0)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unexpected MobileViTv2 checkpoint format. "
            "Expected a checkpoint/state_dict dictionary."
        )

    for key in ("model", "state_dict", "model_ema"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    return checkpoint


def _clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        clean_key = key

        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]

        # Drop original ImageNet classifier only.
        if (
            clean_key.startswith("head.fc.")
            or clean_key.startswith("head.classifier.")
            or clean_key.startswith("classifier.")
            or clean_key.startswith("fc.")
        ):
            continue

        cleaned[clean_key] = value

    return cleaned


def _resolve_pretrained_path(pretrained_path=None):
    if pretrained_path:
        pretrained_path = os.path.expanduser(str(pretrained_path))
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                "\nMobileViTv2 pretrained weight not found:\n"
                f"{pretrained_path}\n"
            )
        return pretrained_path

    candidates = [
        os.path.join(WEIGHT_DIR, "model.safetensors"),
        os.path.join(WEIGHT_DIR, 'mobilevitv2_100_cvnets_in1k.bin'),
        os.path.join(WEIGHT_DIR, 'mobilevitv2_100.pth'),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "\nMobileViTv2 pretrained weight not found.\n"
        "Pass the local checkpoint with --pretrained.\n"
        "Checked:\n" + "\n".join(candidates) + "\n"
    )


def _create_backbone(model_name, drop_path_rate):
    kwargs = dict(
        pretrained=False,
        num_classes=0,
    )

    try:
        return timm.create_model(
            model_name,
            drop_path_rate=float(drop_path_rate),
            **kwargs,
        )
    except TypeError as error:
        # Compatibility with older timm/model implementations.
        if "drop_path" not in str(error).lower():
            raise

        print(
            "[MobileViTv2] drop_path_rate is not supported by this "
            "timm/model version; retrying without it.",
            flush=True,
        )
        return timm.create_model(
            model_name,
            **kwargs,
        )


class MobileViTv2Classifier(nn.Module):
    """
    MobileViTv2 pretrained backbone + fatigue/nofatigue classifier.

    The shared data pipeline outputs ImageNet-normalized tensors.
    This wrapper converts those tensors to the normalization expected
    by the selected pretrained checkpoint before the backbone.
    """

    def __init__(
        self,
        model_name='mobilevitv2_100',
        num_classes=2,
        drop_rate=0.2,
        drop_path_rate=0.0,
        freeze_backbone=False,
        pretrained_path=None,
    ):
        super().__init__()

        model_name = str(model_name).lower()
        if model_name != 'mobilevitv2_100':
            raise ValueError(
                f"Unsupported MobileViTv2 model: {model_name}. "
                "Expected mobilevitv2_100."
            )

        self.model_name = model_name

        self.register_buffer(
            "_source_mean",
            torch.tensor(SOURCE_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_source_std",
            torch.tensor(SOURCE_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_target_mean",
            torch.tensor(TARGET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_target_std",
            torch.tensor(TARGET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        weight_path = _resolve_pretrained_path(pretrained_path)

        print("=" * 70, flush=True)
        print("[MobileViTv2] Loading local pretrained model", flush=True)
        print(f"[MobileViTv2] Model: {model_name}", flush=True)
        print(f"[MobileViTv2] Weight path: {weight_path}", flush=True)
        print(
            "[MobileViTv2] Expected normalization: "
            f"mean={TARGET_MEAN}, std={TARGET_STD}",
            flush=True,
        )

        self.backbone = _create_backbone(
            model_name=model_name,
            drop_path_rate=drop_path_rate,
        )
        self.num_features = int(self.backbone.num_features)

        if str(weight_path).lower().endswith(".safetensors"):
            if load_safetensors is None:
                raise ImportError(
                    "safetensors is required to load this pretrained weight. "
                    "Install it with: pip install safetensors"
                )
            checkpoint = load_safetensors(
                weight_path,
                device="cpu",
            )
        else:
            checkpoint = torch.load(
                weight_path,
                map_location="cpu",
            )

        state_dict = _clean_state_dict(
            _extract_state_dict(checkpoint)
        )

        msg = self.backbone.load_state_dict(
            state_dict,
            strict=False,
        )

        print("[MobileViTv2] Local pretrained weights loaded", flush=True)
        print(
            f"[MobileViTv2] Missing keys: {len(msg.missing_keys)}",
            flush=True,
        )
        print(
            f"[MobileViTv2] Unexpected keys: {len(msg.unexpected_keys)}",
            flush=True,
        )
        if msg.missing_keys:
            print(
                "[MobileViTv2] First missing keys:",
                msg.missing_keys[:10],
                flush=True,
            )
        if msg.unexpected_keys:
            print(
                "[MobileViTv2] First unexpected keys:",
                msg.unexpected_keys[:10],
                flush=True,
            )
        print("=" * 70, flush=True)

        if freeze_backbone:
            print("[MobileViTv2] Backbone frozen", flush=True)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        if float(drop_rate) > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=float(drop_rate)),
                nn.Linear(self.num_features, int(num_classes)),
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
        nn.init.normal_(linear_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear_layer.bias)

        print(
            "[MobileViTv2] Classification head created: "
            f"in_features={self.num_features}, "
            f"num_classes={num_classes}, "
            f"drop_rate={drop_rate}",
            flush=True,
        )

    def _renormalize_input(self, images):
        raw = (
            images * self._source_std.to(images.dtype)
            + self._source_mean.to(images.dtype)
        )
        return (
            raw - self._target_mean.to(images.dtype)
        ) / self._target_std.to(images.dtype)

    def forward_features(self, images):
        images = self._renormalize_input(images)
        features = self.backbone.forward_features(images)

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
        return self.classifier(features)


def build_mobilevitv2(
    config=None,
    num_classes=None,
    model_name='mobilevitv2_100',
    drop_rate=0.2,
    drop_path_rate=0.0,
    freeze_backbone=False,
    pretrained_path=None,
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

    return MobileViTv2Classifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
        pretrained_path=pretrained_path,
    )
