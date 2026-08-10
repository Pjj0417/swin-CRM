import os
import torch
import torch.nn as nn
import timm

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

SUPPORTED_SWINV2_MODELS = {
    "swinv2_cr_small_ns_224.sw_in1k",
}
WEIGHT_DIR = "/root/shared-nvme/uploads"

def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Unexpected SwinV2 checkpoint format.")
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
        if (
            clean_key.startswith("head.fc.")
            or clean_key.startswith("classifier.")
            or clean_key.startswith("fc.")
        ):
            continue
        cleaned[clean_key] = value
    return cleaned

def _coverage(state_dict, model):
    model_state = model.state_dict()
    matched_numel = 0
    total_numel = 0
    matched_keys = set()
    for key, tensor in model_state.items():
        total_numel += tensor.numel()
        if key in state_dict and tuple(state_dict[key].shape) == tuple(tensor.shape):
            matched_numel += tensor.numel()
            matched_keys.add(key)
    return matched_numel / max(1, total_numel), matched_keys

def _resolve_pretrained_path(pretrained_path=None):
    if pretrained_path:
        path = os.path.expanduser(str(pretrained_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "\nSwinV2 pretrained weight not found:\n" + path + "\n"
            )
        return path
    candidates = [
        os.path.join(WEIGHT_DIR, "swinv2_cr_small_ns_224_sw_in1k.safetensors"),
        os.path.join(WEIGHT_DIR, "swinv2_cr_small_ns_224.safetensors"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "\nSwinV2 pretrained weight not found.\n"
        "Pass it with --pretrained.\nChecked:\n" + "\n".join(candidates)
    )

def _create_backbone(model_name, drop_path_rate):
    kwargs = dict(pretrained=False, num_classes=0)
    try:
        return timm.create_model(
            model_name,
            drop_path_rate=float(drop_path_rate),
            **kwargs,
        )
    except TypeError as error:
        if "drop_path" not in str(error).lower():
            raise
        print(
            "[SwinV2] drop_path_rate unsupported; retry without it.",
            flush=True,
        )
        return timm.create_model(model_name, **kwargs)

class SwinV2Classifier(nn.Module):
    def __init__(
        self,
        model_name="swinv2_cr_small_ns_224.sw_in1k",
        num_classes=2,
        drop_rate=0.10,
        drop_path_rate=0.10,
        freeze_backbone=False,
        pretrained_path=None,
    ):
        super().__init__()
        model_name = str(model_name).lower()
        if model_name not in SUPPORTED_SWINV2_MODELS:
            raise ValueError(
                f"Unsupported SwinV2 model: {model_name}. "
                f"Available: {sorted(SUPPORTED_SWINV2_MODELS)}"
            )

        self.model_name = model_name
        weight_path = _resolve_pretrained_path(pretrained_path)
        print("=" * 70, flush=True)
        print("[SwinV2] Loading local pretrained model", flush=True)
        print(f"[SwinV2] Model: {model_name}", flush=True)
        print(f"[SwinV2] Weight path: {weight_path}", flush=True)

        self.backbone = _create_backbone(model_name, drop_path_rate)
        self.num_features = int(self.backbone.num_features)

        if str(weight_path).lower().endswith(".safetensors"):
            if load_safetensors is None:
                raise ImportError("Install safetensors: pip install safetensors")
            checkpoint = load_safetensors(weight_path, device="cpu")
        else:
            checkpoint = torch.load(weight_path, map_location="cpu")

        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        coverage, matched_keys = _coverage(state_dict, self.backbone)
        print(
            f"[SwinV2] Pretrained parameter coverage: {coverage * 100:.2f}%",
            flush=True,
        )
        if coverage < 0.95:
            missing_preview = [
                k for k in self.backbone.state_dict() if k not in matched_keys
            ][:20]
            raise RuntimeError(
                "SwinV2 pretrained checkpoint coverage too low: "
                f"{coverage * 100:.2f}%. "
                f"First unmatched keys: {missing_preview}"
            )

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print("[SwinV2] Local pretrained weights loaded", flush=True)
        print(f"[SwinV2] Missing keys: {len(msg.missing_keys)}", flush=True)
        print(f"[SwinV2] Unexpected keys: {len(msg.unexpected_keys)}", flush=True)
        print("=" * 70, flush=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[SwinV2] Backbone frozen", flush=True)

        if float(drop_rate) > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(float(drop_rate)),
                nn.Linear(self.num_features, int(num_classes)),
            )
        else:
            self.classifier = nn.Linear(self.num_features, int(num_classes))

        linear = self.classifier[-1] if isinstance(self.classifier, nn.Sequential) else self.classifier
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(linear.bias)
        print(
            "[SwinV2] Classification head created: "
            f"in_features={self.num_features}, num_classes={num_classes}",
            flush=True,
        )

    def forward_features(self, images):
        features = self.backbone.forward_features(images)
        if hasattr(self.backbone, "forward_head"):
            return self.backbone.forward_head(features, pre_logits=True)
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        return features

    def forward(self, images):
        return self.classifier(self.forward_features(images))

def build_swinv2(
    config=None,
    num_classes=None,
    model_name="swinv2_cr_small_ns_224.sw_in1k",
    drop_rate=0.10,
    drop_path_rate=0.10,
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
        if pretrained_path is None and hasattr(config, "PRETRAINED") and config.PRETRAINED:
            pretrained_path = str(config.PRETRAINED)
    if num_classes is None:
        num_classes = 2
    return SwinV2Classifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
        pretrained_path=pretrained_path,
    )
