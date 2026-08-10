# import torch.nn as nn

# try:
#     from torchvision.models import (
#         resnet50,
#         ResNet50_Weights,
#     )
# except ImportError:
#     from torchvision.models import resnet50
#     ResNet50_Weights = None


# def build_resnet50(config):
#     """
#     ResNet50 ImageNet pretrained baseline.
#     """

#     if ResNet50_Weights is not None:
#         model = resnet50(
#             weights=ResNet50_Weights.IMAGENET1K_V2
#         )
#     else:
#         model = resnet50(
#             pretrained=True
#         )

#     in_features = model.fc.in_features

#     num_classes = int(
#         config.MODEL.NUM_CLASSES
#     )

#     drop_rate = float(
#         config.MODEL.DROP_RATE
#     )

#     if drop_rate > 0:
#         model.fc = nn.Sequential(
#             nn.Dropout(
#                 p=drop_rate
#             ),
#             nn.Linear(
#                 in_features,
#                 num_classes
#             ),
#         )
#     else:
#         model.fc = nn.Linear(
#             in_features,
#             num_classes
#         )

#     classifier = (
#         model.fc[-1]
#         if isinstance(
#             model.fc,
#             nn.Sequential
#         )
#         else model.fc
#     )

#     nn.init.normal_(
#         classifier.weight,
#         mean=0.0,
#         std=0.01
#     )

#     if classifier.bias is not None:
#         nn.init.zeros_(
#             classifier.bias
#         )

#     return model

# # --------------------------------------------------------
# # SimMIM
# # Copyright (c) 2021 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ze Liu
# # Modified by Zhenda Xie
# # --------------------------------------------------------

# from .swin_transformer import build_swin
# from .vision_transformer import build_vit
# from .resnet_model import build_resnet50
# from .convnext_model import build_convnext
# from .deit_model import build_deit
# from .simmim import build_simmim


# def build_model(config, is_pretrain=True):
#     if is_pretrain:
#         model = build_simmim(config)

#     else:
#         model_type = config.MODEL.TYPE.lower()

#         if model_type == "swin":
#             model = build_swin(config)

#         elif model_type == "vit":
#             model = build_vit(config)

#         elif model_type == "resnet50":
#             model = build_resnet50(config)

#         elif model_type == "convnext":
#             model = build_convnext(config)

#         elif model_type == "deit":
#             model = build_deit(config)

#         else:
#             raise NotImplementedError(
#                 f"Unknown fine-tune model: {model_type}"
#             )

#     return model


# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .resnet_model import build_resnet50
from .dinov2_model import build_dinov2
from .convnext_model import build_convnext
from .deit_model import build_deit
from .mobilevitv2_model import build_mobilevitv2
from .edgenext_model import build_edgenext
from .coatnet_model import build_coatnet
from .simmim import build_simmim


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_simmim(config)

    else:
        model_type = config.MODEL.TYPE.lower()

        if model_type == "swin":
            model = build_swin(config)

        elif model_type == "vit":
            model = build_vit(config)

        elif model_type == "resnet50":
            model = build_resnet50(config)

        elif model_type == "dinov2":
            model = build_dinov2(config)

        elif model_type == "convnext":
            model = build_convnext(config)

        elif model_type == "deit":
            model = build_deit(config)

        elif model_type == "mobilevitv2":
            model = build_mobilevitv2(config)

        elif model_type == "edgenext":
            model = build_edgenext(config)

        elif model_type == "coatnet":
            model = build_coatnet(config)

        else:
            raise NotImplementedError(
                f"Unknown fine-tune model: {model_type}"
            )

    return model
