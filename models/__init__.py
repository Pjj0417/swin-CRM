

# # from .build import build_model

# from .swin_transformer import build_swin
# from .vision_transformer import build_vit
# from .resnet_model import build_resnet50
# from .dinov2_model import build_dinov2
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

#         elif model_type == "dinov2":
#             model = build_dinov2(config)

#         elif model_type == "convnext":
#             model = build_convnext(config)

#         elif model_type == "deit":
#             model = build_deit(config)

#         else:
#             raise NotImplementedError(
#                 f"Unknown fine-tune model: {model_type}"
#             )

#     return model


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

