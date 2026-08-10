# import torch
# import torch.nn as nn


# DINOV2_FEATURE_DIMS = {
#     "dinov2_vits14": 384,
#     "dinov2_vitb14": 768,
#     "dinov2_vitl14": 1024,
#     "dinov2_vitg14": 1536,
# }


# class DINOv2Classifier(nn.Module):
#     """
#     DINOv2 自监督预训练 backbone + 二分类头。
#     """

#     def __init__(
#         self,
#         model_name="dinov2_vits14",
#         num_classes=2,
#         drop_rate=0.2,
#         freeze_backbone=False,
#     ):
#         super().__init__()

#         model_name = str(model_name).lower()

#         if model_name not in DINOV2_FEATURE_DIMS:
#             raise ValueError(
#                 f"不支持的 DINOv2 模型：{model_name}。"
#                 f"可用模型：{list(DINOV2_FEATURE_DIMS.keys())}"
#             )

#         self.model_name = model_name
#         self.num_features = DINOV2_FEATURE_DIMS[model_name]

#         local_repo = "/home/administrator/dinov2"
#         # weights_path = (
#         #     "/home/administrator/.cache/torch/hub/checkpoints/"
#         #     "dinov2_vits14_pretrain.pth"
#         # )
#         weights_path = (
#            "/home/administrator/.cache/torch/hub/checkpoints/"
#            f"{model_name}_pretrain.pth"
#         )

#         self.backbone = torch.hub.load(
#             local_repo,
#             model_name,
#             source="local",
#             pretrained=False,
#         )

#         checkpoint = torch.load(
#             weights_path,
#             map_location="cpu",
#         )

#         if isinstance(checkpoint, dict) and "model" in checkpoint:
#             checkpoint = checkpoint["model"]

#         incompatible = self.backbone.load_state_dict(
#             checkpoint,
#             strict=False,
#         )

#         # DINOv2 的 mask_token 只在掩码预训练时使用。
#         # 普通分类不会经过该参数，因此将其冻结，
#         # 避免 DistributedDataParallel 报 unused parameter。
#         if hasattr(self.backbone, "mask_token"):
#             self.backbone.mask_token.requires_grad_(False)

#         print(
#             "DINOv2 local weights loaded: "
#             f"missing={len(incompatible.missing_keys)}, "
#             f"unexpected={len(incompatible.unexpected_keys)}",
#             flush=True,
#         )

#         if freeze_backbone:
#             for parameter in self.backbone.parameters():
#                 parameter.requires_grad = False

#         if float(drop_rate) > 0.0:
#             self.classifier = nn.Sequential(
#                 nn.Dropout(p=float(drop_rate)),
#                 nn.Linear(
#                     self.num_features,
#                     int(num_classes),
#                 ),
#             )
#         else:
#             self.classifier = nn.Linear(
#                 self.num_features,
#                 int(num_classes),
#             )

#         classifier = (
#             self.classifier[-1]
#             if isinstance(self.classifier, nn.Sequential)
#             else self.classifier
#         )

#         nn.init.normal_(
#             classifier.weight,
#             mean=0.0,
#             std=0.01,
#         )
#         nn.init.zeros_(classifier.bias)

#     def forward_features(self, images):
#         features = self.backbone.forward_features(images)

#         if isinstance(features, dict):
#             if "x_norm_clstoken" in features:
#                 return features["x_norm_clstoken"]

#             if "x_norm_patchtokens" in features:
#                 return features[
#                     "x_norm_patchtokens"
#                 ].mean(dim=1)

#             raise RuntimeError(
#                 "DINOv2 输出中没有找到 "
#                 "x_norm_clstoken 或 x_norm_patchtokens。"
#             )

#         return features

#     def forward(self, images):
#         features = self.forward_features(images)
#         return self.classifier(features)


# def build_dinov2(
#     num_classes=2,
#     model_name="dinov2_vits14",
#     drop_rate=0.2,
#     freeze_backbone=False,
# ):
#     return DINOv2Classifier(
#         model_name=model_name,
#         num_classes=num_classes,
#         drop_rate=drop_rate,
#         freeze_backbone=freeze_backbone,
#     )

# import os
# import torch
# import torch.nn as nn


# DINOV2_FEATURE_DIMS = {
#     "dinov2_vits14": 384,
#     "dinov2_vitb14": 768,
#     "dinov2_vitl14": 1024,
#     "dinov2_vitg14": 1536,
# }


# class DINOv2Classifier(nn.Module):
#     """
#     Offline DINOv2 pretrained backbone + classifier.
#     """

#     def __init__(
#         self,
#         model_name="dinov2_vits14",
#         num_classes=2,
#         drop_rate=0.2,
#         freeze_backbone=False,
#     ):
#         super().__init__()

#         model_name = str(model_name).lower()

#         if model_name not in DINOV2_FEATURE_DIMS:
#             raise ValueError(
#                 f"Unsupported model: {model_name}"
#             )

#         self.model_name = model_name
#         self.num_features = DINOV2_FEATURE_DIMS[model_name]


#         # ==============================
#         # Local DINOv2 repository
#         # ==============================

#         local_repo = "/home/administrator/dinov2"


#         weight_path = (
#             "/home/administrator/.cache/torch/hub/checkpoints/"
#             f"{model_name}_pretrain.pth"
#         )


#         print("=" * 70, flush=True)

#         print(
#             f"[DINOv2] Loading local model: {model_name}",
#             flush=True,
#         )

#         print(
#             f"[DINOv2] Weight path: {weight_path}",
#             flush=True,
#         )


#         if not os.path.exists(weight_path):

#             raise FileNotFoundError(
#                 "\nDINOv2 weight not found:\n"
#                 f"{weight_path}\n\n"
#                 "Please put pretrained weight into:\n"
#                 "/home/administrator/.cache/torch/hub/checkpoints/\n"
#             )


#         # ==============================
#         # Build backbone
#         # ==============================

#         self.backbone = torch.hub.load(
#             local_repo,
#             model_name,
#             source="local",
#             pretrained=False,
#         )


#         checkpoint = torch.load(
#             weight_path,
#             map_location="cpu",
#         )


#         if isinstance(checkpoint, dict):

#             if "model" in checkpoint:
#                 checkpoint = checkpoint["model"]


#         msg = self.backbone.load_state_dict(
#             checkpoint,
#             strict=False,
#         )


#         print(
#             "[DINOv2] Local pretrained weights loaded",
#             flush=True,
#         )

#         print(
#             f"missing keys: {len(msg.missing_keys)}",
#             flush=True,
#         )

#         print(
#             f"unexpected keys: {len(msg.unexpected_keys)}",
#             flush=True,
#         )

#         print("=" * 70, flush=True)



#         # avoid DDP unused parameter
#         if hasattr(self.backbone, "mask_token"):

#             self.backbone.mask_token.requires_grad_(False)



#         if freeze_backbone:

#             print(
#                 "[DINOv2] Backbone frozen",
#                 flush=True,
#             )

#             for p in self.backbone.parameters():

#                 p.requires_grad = False



#         # ==============================
#         # Classification head
#         # ==============================

#         if float(drop_rate) > 0:

#             self.classifier = nn.Sequential(

#                 nn.Dropout(
#                     p=float(drop_rate)
#                 ),

#                 nn.Linear(
#                     self.num_features,
#                     num_classes,
#                 )
#             )

#         else:

#             self.classifier = nn.Linear(
#                 self.num_features,
#                 num_classes,
#             )


#         linear_layer = (
#             self.classifier[-1]
#             if isinstance(
#                 self.classifier,
#                 nn.Sequential
#             )
#             else self.classifier
#         )


#         nn.init.normal_(
#             linear_layer.weight,
#             mean=0.0,
#             std=0.01,
#         )

#         nn.init.zeros_(
#             linear_layer.bias
#         )



#     def forward_features(self, images):

#         features = self.backbone.forward_features(
#             images
#         )


#         if isinstance(features, dict):

#             if "x_norm_clstoken" in features:

#                 return features[
#                     "x_norm_clstoken"
#                 ]


#             if "x_norm_patchtokens" in features:

#                 return features[
#                     "x_norm_patchtokens"
#                 ].mean(dim=1)


#             raise RuntimeError(
#                 "DINOv2 feature output not found"
#             )


#         return features



#     def forward(self, images):

#         features = self.forward_features(
#             images
#         )

#         return self.classifier(
#             features
#         )



# def build_dinov2(
#     num_classes=2,
#     model_name="dinov2_vits14",
#     drop_rate=0.2,
#     freeze_backbone=False,
# ):

#     return DINOv2Classifier(
#         model_name=model_name,
#         num_classes=num_classes,
#         drop_rate=drop_rate,
#         freeze_backbone=freeze_backbone,
#     )


import os

import torch
import torch.nn as nn


DINOV2_FEATURE_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


DINOV2_LOCAL_REPO = "/root/shared-nvme/uploads/dinov2"

DINOV2_WEIGHT_DIR = "/root/shared-nvme/uploads"


class DINOv2Classifier(nn.Module):
    """
    Offline DINOv2 pretrained backbone + classifier.

    Local DINOv2 repository:
        /root/shared-nvme/uploads/dinov2

    Local pretrained weights:
        /root/shared-nvme/uploads/<model_name>_pretrain.pth
    """

    def __init__(
        self,
        model_name="dinov2_vits14",
        num_classes=2,
        drop_rate=0.2,
        freeze_backbone=False,
    ):
        super().__init__()

        model_name = str(
            model_name
        ).lower()

        if model_name not in DINOV2_FEATURE_DIMS:
            raise ValueError(
                f"Unsupported DINOv2 model: "
                f"{model_name}"
            )

        self.model_name = model_name

        self.num_features = (
            DINOV2_FEATURE_DIMS[
                model_name
            ]
        )

        # ==================================================
        # Local DINOv2 repository
        # ==================================================

        local_repo = (
            DINOV2_LOCAL_REPO
        )

        weight_path = os.path.join(
            DINOV2_WEIGHT_DIR,
            f"{model_name}_pretrain.pth",
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            "[DINOv2] Loading local pretrained model",
            flush=True,
        )

        print(
            f"[DINOv2] Model: "
            f"{model_name}",
            flush=True,
        )

        print(
            f"[DINOv2] Local repo: "
            f"{local_repo}",
            flush=True,
        )

        print(
            f"[DINOv2] Weight path: "
            f"{weight_path}",
            flush=True,
        )

        # ==================================================
        # Check local repository
        # ==================================================

        if not os.path.isdir(
            local_repo
        ):
            raise FileNotFoundError(
                "\nDINOv2 local repository "
                "not found:\n"
                f"{local_repo}\n\n"
                "Please place the DINOv2 "
                "repository at:\n"
                f"{local_repo}\n"
            )

        hubconf_path = os.path.join(
            local_repo,
            "hubconf.py",
        )

        if not os.path.isfile(
            hubconf_path
        ):
            raise FileNotFoundError(
                "\nDINOv2 hubconf.py "
                "not found:\n"
                f"{hubconf_path}\n\n"
                "The directory does not "
                "appear to be a valid "
                "DINOv2 repository.\n"
            )

        # ==================================================
        # Check local pretrained weight
        # ==================================================

        if not os.path.isfile(
            weight_path
        ):
            raise FileNotFoundError(
                "\nDINOv2 pretrained weight "
                "not found:\n"
                f"{weight_path}\n\n"
                "Please upload the pretrained "
                "weight to:\n"
                f"{DINOV2_WEIGHT_DIR}\n\n"
                "Expected filename:\n"
                f"{model_name}_pretrain.pth\n"
            )

        # ==================================================
        # Build backbone from local repository
        #
        # pretrained=False is important:
        # do NOT let torch.hub download weights.
        # ==================================================

        print(
            "[DINOv2] Building backbone "
            "from local repository...",
            flush=True,
        )

        self.backbone = (
            torch.hub.load(
                local_repo,
                model_name,
                source="local",
                pretrained=False,
            )
        )

        print(
            "[DINOv2] Backbone created",
            flush=True,
        )

        # ==================================================
        # Load local pretrained weights
        # ==================================================

        print(
            "[DINOv2] Loading local "
            "pretrained weights...",
            flush=True,
        )

        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
        )

        # Some checkpoints may have:
        # {
        #     "model": state_dict
        # }
        if isinstance(
            checkpoint,
            dict,
        ):
            if "model" in checkpoint:
                checkpoint = (
                    checkpoint["model"]
                )

            elif "state_dict" in checkpoint:
                checkpoint = (
                    checkpoint["state_dict"]
                )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise TypeError(
                "Unexpected DINOv2 "
                "checkpoint format. "
                "Expected a state_dict "
                "dictionary."
            )

        # Remove common prefixes if present.
        cleaned_checkpoint = {}

        for key, value in (
            checkpoint.items()
        ):
            clean_key = key

            if clean_key.startswith(
                "module."
            ):
                clean_key = (
                    clean_key[
                        len("module.") :
                    ]
                )

            if clean_key.startswith(
                "backbone."
            ):
                clean_key = (
                    clean_key[
                        len("backbone.") :
                    ]
                )

            cleaned_checkpoint[
                clean_key
            ] = value

        msg = (
            self.backbone.load_state_dict(
                cleaned_checkpoint,
                strict=False,
            )
        )

        print(
            "[DINOv2] Local pretrained "
            "weights loaded",
            flush=True,
        )

        print(
            f"[DINOv2] Missing keys: "
            f"{len(msg.missing_keys)}",
            flush=True,
        )

        print(
            f"[DINOv2] Unexpected keys: "
            f"{len(msg.unexpected_keys)}",
            flush=True,
        )

        if msg.missing_keys:
            print(
                "[DINOv2] First missing "
                "keys:",
                msg.missing_keys[:10],
                flush=True,
            )

        if msg.unexpected_keys:
            print(
                "[DINOv2] First unexpected "
                "keys:",
                msg.unexpected_keys[:10],
                flush=True,
            )

        print(
            "=" * 70,
            flush=True,
        )

        # ==================================================
        # Avoid DDP unused parameter
        # ==================================================

        if hasattr(
            self.backbone,
            "mask_token",
        ):
            self.backbone.mask_token.requires_grad_(
                False
            )

        # ==================================================
        # Optional backbone freezing
        # ==================================================

        if freeze_backbone:

            print(
                "[DINOv2] Backbone frozen",
                flush=True,
            )

            for parameter in (
                self.backbone.parameters()
            ):
                parameter.requires_grad = (
                    False
                )

        # ==================================================
        # Classification head
        # ==================================================

        if float(
            drop_rate
        ) > 0.0:

            self.classifier = nn.Sequential(
                nn.Dropout(
                    p=float(
                        drop_rate
                    )
                ),
                nn.Linear(
                    self.num_features,
                    num_classes,
                ),
            )

        else:

            self.classifier = nn.Linear(
                self.num_features,
                num_classes,
            )

        linear_layer = (
            self.classifier[-1]
            if isinstance(
                self.classifier,
                nn.Sequential,
            )
            else self.classifier
        )

        # New binary classification head.
        nn.init.normal_(
            linear_layer.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(
            linear_layer.bias
        )

        print(
            "[DINOv2] Classification head "
            "created:",
            f"in_features={self.num_features}, "
            f"num_classes={num_classes}, "
            f"drop_rate={drop_rate}",
            flush=True,
        )

    def forward_features(
        self,
        images,
    ):
        features = (
            self.backbone.forward_features(
                images
            )
        )

        if isinstance(
            features,
            dict,
        ):

            # Standard DINOv2 CLS token.
            if (
                "x_norm_clstoken"
                in features
            ):
                return features[
                    "x_norm_clstoken"
                ]

            # Fallback:
            # mean pool patch tokens.
            if (
                "x_norm_patchtokens"
                in features
            ):
                return features[
                    "x_norm_patchtokens"
                ].mean(
                    dim=1
                )

            raise RuntimeError(
                "DINOv2 feature output "
                "not found. Available keys: "
                f"{list(features.keys())}"
            )

        return features

    def forward(
        self,
        images,
    ):
        features = (
            self.forward_features(
                images
            )
        )

        logits = self.classifier(
            features
        )

        return logits


def build_dinov2(
    num_classes=2,
    model_name="dinov2_vits14",
    drop_rate=0.2,
    freeze_backbone=False,
):
    return DINOv2Classifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        freeze_backbone=freeze_backbone,
    )
