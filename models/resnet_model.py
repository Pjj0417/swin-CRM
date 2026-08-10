# from .build import build_model

import torch.nn as nn

try:
    from torchvision.models import (
        resnet50,
        ResNet50_Weights,
    )
except ImportError:
    from torchvision.models import resnet50
    ResNet50_Weights = None


def build_resnet50(config):
    """
    Build ResNet50 with ImageNet-1K pretrained weights
    and replace its classifier for the current task.
    """
    if ResNet50_Weights is not None:
        model = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2,
        )
    else:
        # Compatibility with older torchvision versions.
        model = resnet50(
            pretrained=True,
        )

    in_features = model.fc.in_features
    num_classes = int(config.MODEL.NUM_CLASSES)
    drop_rate = float(config.MODEL.DROP_RATE)

    if drop_rate > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(
                in_features,
                num_classes,
            ),
        )
    else:
        model.fc = nn.Linear(
            in_features,
            num_classes,
        )

    classifier = (
        model.fc[-1]
        if isinstance(model.fc, nn.Sequential)
        else model.fc
    )

    nn.init.normal_(
        classifier.weight,
        mean=0.0,
        std=0.01,
    )
    nn.init.zeros_(
        classifier.bias,
    )

    return model
