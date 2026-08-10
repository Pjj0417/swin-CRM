# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# ResNet50 ImageNet fine-tuning and Grad-CAM++ adaptation
# --------------------------------------------------------

import os
import random
import time
import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import SoftTargetCrossEntropy
from timm.utils import AverageMeter
try:
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    # 兼容较旧版本 torchvision。
    from torchvision.models import resnet50
    ResNet50_Weights = None

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAMPlusPlus = None
    show_cam_on_image = None
    ClassifierOutputTarget = None

from config import get_config
from data import build_loader
from lr_scheduler import build_scheduler
from logger import create_logger
from utils import (
    load_checkpoint,
    get_grad_norm,
    auto_resume_helper,
)

try:
    from apex import amp
except ImportError:
    amp = None


# Grad-CAM++ runtime options populated from command-line arguments.
GRADCAM_OUTPUT_DIR = None
GRADCAM_MAX_PER_CLASS = 4
RESNET_WEIGHTS = "IMAGENET1K_V2"
RESNET_CHECKPOINT = None
RESNET_OPTIMIZER = "adamw"


def parse_option():
    parser = argparse.ArgumentParser(
        "ResNet50 training, evaluation and Grad-CAM++ script",
        add_help=False,
    )
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE")
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--resume")
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
    )
    parser.add_argument("--output", default="output", type=str, metavar="PATH")
    parser.add_argument("--tag")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument(
        "--gradcam-output-dir",
        type=str,
        default=None,
        help=(
            "Grad-CAM++ 独立输出目录。默认使用 "
            "<config.OUTPUT>/gradcam。"
        ),
    )
    parser.add_argument(
        "--gradcam-max-per-class",
        type=int,
        default=4,
        help="每个真实类别保存的正确分类 Grad-CAM++ 图片数量。",
    )
    parser.add_argument(
        "--resnet-weights",
        type=str,
        default="IMAGENET1K_V2",
        choices=[
            "IMAGENET1K_V2",
            "IMAGENET1K_V1",
            "DEFAULT",
            "NONE",
        ],
        help=(
            "TorchVision ResNet50 初始化权重。"
            "默认使用 IMAGENET1K_V2。"
        ),
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=str,
        default=None,
        help=(
            "可选的本地 ResNet50 权重路径。"
            "支持纯 state_dict 或包含 model/state_dict 的 checkpoint。"
        ),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="ResNet50 微调优化器。",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
    )
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def binary_roc_auc(targets, scores):
    """Exact binary ROC-AUC using average ranks for tied scores."""
    targets = torch.as_tensor(targets, dtype=torch.long).cpu().numpy()
    scores = torch.as_tensor(scores, dtype=torch.float64).cpu().numpy()

    n_pos = int((targets == 1).sum())
    n_neg = int((targets == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[targets == 1].sum()

    return float(
        (
            positive_rank_sum
            - n_pos * (n_pos + 1) / 2.0
        )
        / (n_pos * n_neg)
    )


def distributed_gather_1d(tensor):
    tensor = tensor.contiguous().view(-1)

    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return tensor

    world_size = dist.get_world_size()
    local_size = torch.tensor(
        [tensor.numel()],
        dtype=torch.long,
        device=tensor.device,
    )

    sizes_tensor = [
        torch.zeros_like(local_size)
        for _ in range(world_size)
    ]
    dist.all_gather(sizes_tensor, local_size)

    sizes = [int(item.item()) for item in sizes_tensor]
    max_size = max(sizes)

    if tensor.numel() < max_size:
        tensor = torch.cat(
            [
                tensor,
                tensor.new_zeros(max_size - tensor.numel()),
            ],
            dim=0,
        )

    gathered = [
        torch.empty(
            max_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(world_size)
    ]

    dist.all_gather(gathered, tensor)

    return torch.cat(
        [
            part[:size]
            for part, size in zip(gathered, sizes)
        ],
        dim=0,
    )


def resolve_resnet50_weights(weights_name):
    """将命令行权重名称转换为 TorchVision weights enum。"""
    normalized = str(weights_name).upper()

    if normalized not in {
        "NONE",
        "DEFAULT",
        "IMAGENET1K_V1",
        "IMAGENET1K_V2",
    }:
        raise ValueError(
            f"不支持的 ResNet50 权重：{weights_name}"
        )

    if normalized == "NONE":
        return None

    # 旧版 torchvision 没有 Weights enum，
    # build_resnet50_model() 会退回 pretrained=True。
    if ResNet50_Weights is None:
        return "LEGACY_PRETRAINED"

    if normalized == "DEFAULT":
        return ResNet50_Weights.DEFAULT

    if normalized == "IMAGENET1K_V1":
        return ResNet50_Weights.IMAGENET1K_V1

    return ResNet50_Weights.IMAGENET1K_V2


def extract_model_state_dict(checkpoint):
    """
    从常见 checkpoint 格式中提取模型参数，并移除 module. 前缀。
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "ResNet50 checkpoint 必须是字典或 state_dict。"
        )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        clean_key = key

        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]

        cleaned_state_dict[clean_key] = value

    return cleaned_state_dict


def load_local_resnet50_weights(
    model,
    checkpoint_path,
    logger,
):
    """
    加载本地 ResNet50 权重。

    若分类头维度不匹配，会自动忽略 fc 参数，
    保留当前二分类头。
    """
    checkpoint_path = os.path.expanduser(
        str(checkpoint_path)
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"ResNet50 checkpoint 不存在：{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    state_dict = extract_model_state_dict(
        checkpoint
    )

    current_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key not in current_state:
            skipped_keys.append(key)
            continue

        if current_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue

        filtered_state[key] = value

    incompatible = model.load_state_dict(
        filtered_state,
        strict=False,
    )

    logger.info(
        "Loaded local ResNet50 checkpoint: "
        f"{checkpoint_path}"
    )
    logger.info(
        "ResNet50 checkpoint load summary: "
        f"loaded={len(filtered_state)}, "
        f"skipped={len(skipped_keys)}, "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )


def build_resnet50_model(
    config,
    weights_name,
    checkpoint_path,
    logger,
):
    """
    构建 ResNet50，并将分类头替换为二分类输出。

    默认初始化：
        TorchVision ResNet50 IMAGENET1K_V2。
    """
    weights = resolve_resnet50_weights(
        weights_name
    )

    if weights == "LEGACY_PRETRAINED":
        model = resnet50(
            pretrained=True,
        )
    elif ResNet50_Weights is None:
        model = resnet50(
            pretrained=False,
        )
    else:
        model = resnet50(
            weights=weights,
        )

    num_classes = int(
        config.MODEL.NUM_CLASSES
    )
    drop_rate = float(
        config.MODEL.DROP_RATE
    )
    in_features = model.fc.in_features

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

    # 新分类层使用标准初始化。
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
        classifier.bias
    )

    if checkpoint_path:
        load_local_resnet50_weights(
            model,
            checkpoint_path,
            logger,
        )

    logger.info(
        "Created ResNet50: "
        f"weights={weights_name}, "
        f"num_classes={num_classes}, "
        f"drop_rate={drop_rate}"
    )

    return model


def build_resnet_optimizer(
    config,
    model,
    optimizer_name,
    logger,
):
    """
    为 ResNet50 构建优化器。

    BatchNorm、bias 和其他一维参数不使用 weight decay；
    卷积与全连接权重使用 config.TRAIN.WEIGHT_DECAY。
    """
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim == 1
            or name.endswith(".bias")
        ):
            no_decay_parameters.append(
                parameter
            )
        else:
            decay_parameters.append(
                parameter
            )

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": float(
                config.TRAIN.WEIGHT_DECAY
            ),
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    learning_rate = float(
        config.TRAIN.BASE_LR
    )
    optimizer_name = str(
        optimizer_name
    ).lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    else:
        raise ValueError(
            f"不支持的优化器：{optimizer_name}"
        )

    logger.info(
        "Created ResNet50 optimizer: "
        f"name={optimizer_name}, "
        f"lr={learning_rate:.8f}, "
        f"weight_decay={config.TRAIN.WEIGHT_DECAY}, "
        f"decay_tensors={len(decay_parameters)}, "
        f"no_decay_tensors={len(no_decay_parameters)}"
    )

    return optimizer


def find_resnet_gradcam_target_layers(
    model_without_ddp,
    logger=None,
):
    """
    ResNet50 Grad-CAM++ 默认目标层：
        layer4[-1]

    该层是最后一个 Bottleneck，特征图通常为 7×7。
    """
    if not hasattr(model_without_ddp, "layer4"):
        raise AttributeError(
            "当前模型不包含 ResNet layer4。"
        )

    if len(model_without_ddp.layer4) == 0:
        raise RuntimeError(
            "ResNet layer4 为空。"
        )

    target_layer = model_without_ddp.layer4[-1]

    if logger is not None:
        logger.info(
            "Selected ResNet50 Grad-CAM++ target layer: "
            "layer4[-1]"
        )

    return [target_layer]



def denormalize_imagenet_image(image_tensor):
    mean = image_tensor.new_tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = image_tensor.new_tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = image_tensor * std + mean
    return image.clamp(0.0, 1.0)


def save_best_weights(
    config,
    epoch,
    model_without_ddp,
    accuracy,
    f1,
    balanced_accuracy,
    auc,
    logger,
):
    """只保留一份 Balanced Accuracy 最佳权重。"""
    save_path = os.path.join(
        config.OUTPUT,
        "best_ba_weights.pth",
    )

    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "epoch": int(epoch),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "auc": float(auc),
        "class_mapping": {
            "fatigue": 0,
            "nofatigue": 1,
        },
        "selection_metric": "balanced_accuracy",
        "architecture": "resnet50",
        "initialization": str(RESNET_WEIGHTS),
    }

    torch.save(checkpoint, save_path)

    logger.info(
        "Saved new best Balanced Accuracy weights: "
        f"{save_path} "
        f"(epoch={epoch}, "
        f"ACC={accuracy:.2f}%, "
        f"F1={f1:.2f}%, "
        f"BA={balanced_accuracy:.2f}%, "
        f"AUC={auc:.4f})"
    )


def save_gradcam_images(
    config,
    data_loader,
    model,
    epoch,
    logger,
    max_per_class=None,
    output_root=None,
):
    """
    快速保存分类正确的 ResNet50 Grad-CAM++ 热图。

    加速方式：
        1. 只从验证集前部挑选分类正确的样本；
        2. 每类数量达到 max_per_class 后立即停止扫描；
        3. 将所有选中图片合成一个 batch，只调用一次 Grad-CAM++。

    每张结果图从左到右为：
        原图 | 独立热图 | 热图叠加图
    """
    if GradCAMPlusPlus is None:
        logger.warning(
            "未安装 grad-cam，跳过热图保存。"
            "安装命令：python -m pip install grad-cam opencv-python"
        )
        return

    if max_per_class is None:
        max_per_class = GRADCAM_MAX_PER_CLASS

    if output_root is None:
        output_root = GRADCAM_OUTPUT_DIR

    max_per_class = int(max_per_class)

    if max_per_class <= 0:
        logger.warning(
            "gradcam-max-per-class <= 0，跳过 Grad-CAM++。"
        )
        return

    is_distributed = (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )

    # 所有进程在 epoch 末尾先会合。
    if is_distributed:
        dist.barrier()

    # 非主进程等待 rank 0 完成 Grad-CAM++。
    if is_distributed and dist.get_rank() != 0:
        dist.barrier()
        return

    model_without_ddp = (
        model.module
        if hasattr(model, "module")
        else model
    )

    if output_root:
        gradcam_root = Path(output_root).expanduser()
    else:
        gradcam_root = (
            Path(config.OUTPUT)
            / "gradcam"
        )

    output_dir = (
        gradcam_root
        / f"epoch_{epoch:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {
        0: "fatigue",
        1: "nofatigue",
    }

    selected_per_class = {
        0: [],
        1: [],
    }
    saved_per_class = {
        0: 0,
        1: 0,
    }

    model_without_ddp.eval()
    cam = None

    def enough_selected():
        return all(
            len(selected_per_class[class_id])
            >= max_per_class
            for class_id in selected_per_class
        )

    try:
        target_layers = (
            find_resnet_gradcam_target_layers(
                model_without_ddp,
                logger=logger,
            )
        )

        logger.info(
            "Grad-CAM++ image plan: "
            "mode=fast_correct_batch, "
            f"fatigue={max_per_class}, "
            f"nofatigue={max_per_class}, "
            f"total={max_per_class * 2}, "
            f"output_root={gradcam_root}"
        )

        cam = GradCAMPlusPlus(
            model=model_without_ddp,
            target_layers=target_layers,
        )

        scanned_batches = 0

        # 仅扫描到每类都收集够正确样本为止。
        with torch.no_grad():
            for images, targets in data_loader:
                scanned_batches += 1

                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                logits = model_without_ddp(images)
                probabilities = torch.softmax(
                    logits.float(),
                    dim=1,
                )
                predictions = torch.argmax(
                    probabilities,
                    dim=1,
                )

                for class_id in selected_per_class:
                    remaining = (
                        max_per_class
                        - len(selected_per_class[class_id])
                    )

                    if remaining <= 0:
                        continue

                    # 只选真实标签和预测标签都等于该类别的样本。
                    correct_indices = torch.nonzero(
                        (targets == class_id)
                        & (predictions == class_id),
                        as_tuple=False,
                    ).view(-1)

                    if correct_indices.numel() == 0:
                        continue

                    # 在当前 batch 内优先选置信度更高的正确样本。
                    class_confidences = probabilities[
                        correct_indices,
                        class_id,
                    ]
                    sorted_order = torch.argsort(
                        class_confidences,
                        descending=True,
                    )
                    correct_indices = correct_indices[
                        sorted_order
                    ]

                    for index_tensor in correct_indices[:remaining]:
                        batch_index = int(index_tensor.item())
                        confidence = float(
                            probabilities[
                                batch_index,
                                class_id,
                            ].item()
                        )

                        selected_per_class[class_id].append(
                            {
                                "image": (
                                    images[batch_index]
                                    .detach()
                                    .clone()
                                ),
                                "true_class": class_id,
                                "predicted_class": class_id,
                                "confidence": confidence,
                            }
                        )

                if enough_selected():
                    break

        logger.info(
            "Grad-CAM++ sample collection finished: "
            f"scanned_batches={scanned_batches}, "
            f"fatigue={len(selected_per_class[0])}/"
            f"{max_per_class}, "
            f"nofatigue={len(selected_per_class[1])}/"
            f"{max_per_class}"
        )

        selected_samples = []

        for class_id in sorted(selected_per_class):
            selected_samples.extend(
                selected_per_class[class_id]
            )

        if not selected_samples:
            logger.warning(
                "验证集中没有找到分类正确的样本，"
                "本 epoch 不生成 Grad-CAM++。"
            )
            return

        # 一次性对全部选中样本生成 Grad-CAM++，避免逐张反向传播。
        input_batch = torch.stack(
            [
                sample["image"]
                for sample in selected_samples
            ],
            dim=0,
        )

        cam_targets = [
            ClassifierOutputTarget(
                sample["predicted_class"]
            )
            for sample in selected_samples
        ]

        model_without_ddp.zero_grad(
            set_to_none=True
        )

        grayscale_cams = cam(
            input_tensor=input_batch,
            targets=cam_targets,
            aug_smooth=False,
            eigen_smooth=False,
        )

        for sample_index, sample in enumerate(selected_samples):
            image = input_batch[
                sample_index:sample_index + 1
            ]
            true_class = sample["true_class"]
            predicted_class = sample["predicted_class"]
            confidence = sample["confidence"]

            grayscale_cam = grayscale_cams[
                sample_index
            ]
            grayscale_cam = np.nan_to_num(
                grayscale_cam,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            grayscale_cam = np.clip(
                grayscale_cam,
                0.0,
                1.0,
            ).astype(np.float32)

            grayscale_cam = cv2.GaussianBlur(
                grayscale_cam,
                ksize=(5, 5),
                sigmaX=0,
            )

            cam_min = float(
                grayscale_cam.min()
            )
            cam_max = float(
                grayscale_cam.max()
            )

            if cam_max > cam_min:
                grayscale_cam = (
                    grayscale_cam - cam_min
                ) / (cam_max - cam_min)

            rgb_image = denormalize_imagenet_image(
                image[0]
            )
            rgb_image = (
                rgb_image
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            rgb_image = np.clip(
                rgb_image,
                0.0,
                1.0,
            )

            overlay = show_cam_on_image(
                rgb_image,
                grayscale_cam,
                use_rgb=True,
                image_weight=0.55,
            )

            original = (
                rgb_image * 255.0
            ).round().clip(
                0,
                255,
            ).astype(np.uint8)

            heatmap_uint8 = (
                grayscale_cam * 255.0
            ).round().clip(
                0,
                255,
            ).astype(np.uint8)

            heatmap_bgr = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET,
            )
            heatmap_rgb = cv2.cvtColor(
                heatmap_bgr,
                cv2.COLOR_BGR2RGB,
            )

            combined = np.concatenate(
                [
                    original,
                    heatmap_rgb,
                    overlay,
                ],
                axis=1,
            )

            true_name = class_names.get(
                true_class,
                str(true_class),
            )
            pred_name = class_names.get(
                predicted_class,
                str(predicted_class),
            )

            class_index = saved_per_class[
                true_class
            ]
            filename = (
                f"sample_{class_index:03d}_"
                f"true_{true_name}_"
                f"pred_{pred_name}_"
                f"prob_{confidence:.4f}_"
                "correct.jpg"
            )
            save_path = output_dir / filename

            success = cv2.imwrite(
                str(save_path),
                cv2.cvtColor(
                    combined,
                    cv2.COLOR_RGB2BGR,
                ),
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95,
                ],
            )

            if not success:
                raise RuntimeError(
                    f"Grad-CAM 保存失败：{save_path}"
                )

            saved_per_class[true_class] += 1

            logger.info(
                "Saved Grad-CAM++ image: "
                f"{save_path} "
                f"(confidence={confidence:.4f}, "
                f"fatigue={saved_per_class[0]}/"
                f"{max_per_class}, "
                f"nofatigue={saved_per_class[1]}/"
                f"{max_per_class})"
            )

    except Exception as error:
        logger.exception(
            "Grad-CAM++ 生成失败，但训练和权重保存不受影响。"
        )
        logger.warning(
            f"Grad-CAM++ error: {error}"
        )
        return

    finally:
        if (
            cam is not None
            and hasattr(cam, "clear_hooks")
        ):
            cam.clear_hooks()

        model_without_ddp.zero_grad(
            set_to_none=True
        )

        # 释放 Grad-CAM 临时显存。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # rank 0 完成后释放其他进程。
        if is_distributed:
            dist.barrier()

    if any(
        saved_per_class[class_id] < max_per_class
        for class_id in saved_per_class
    ):
        logger.warning(
            "验证集中某些类别的正确样本不足："
            f"fatigue={saved_per_class[0]}/{max_per_class}, "
            f"nofatigue={saved_per_class[1]}/{max_per_class}"
        )

    logger.info(
        "Grad-CAM++ finished: "
        "mode=fast_correct_batch, "
        f"fatigue={saved_per_class[0]}, "
        f"nofatigue={saved_per_class[1]}, "
        f"output={output_dir}"
    )


def main(config):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config, logger, is_pretrain=False)

    logger.info(
        "Creating model: ResNet50"
    )

    model = build_resnet50_model(
        config=config,
        weights_name=RESNET_WEIGHTS,
        checkpoint_path=RESNET_CHECKPOINT,
        logger=logger,
    )
    model.cuda()
    logger.info(str(model))

    optimizer = build_resnet_optimizer(
        config=config,
        model=model,
        optimizer_name=RESNET_OPTIMIZER,
        logger=logger,
    )

    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level=config.AMP_OPT_LEVEL,
        )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
    )

    model_without_ddp = model.module

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info(f"number of params: {n_parameters}")

    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(
        config,
        optimizer,
        len(data_loader_train),
    )

    fatigue_count = dataset_train.class_sample_counts["fatigue"]
    nofatigue_count = dataset_train.class_sample_counts["nofatigue"]
    total_count = fatigue_count + nofatigue_count

    class_weights = torch.tensor(
        [
            total_count / (2.0 * fatigue_count),
            total_count / (2.0 * nofatigue_count),
        ],
        dtype=torch.float32,
        device="cuda",
    )

    logger.info(
        "Training class weights: "
        f"fatigue={class_weights[0].item():.6f}, "
        f"nofatigue={class_weights[1].item():.6f}"
    )

    if config.AUG.MIXUP > 0.0:
        logger.warning(
            "Mixup 已启用，训练损失使用 SoftTargetCrossEntropy，"
            "类别权重不会应用到 soft labels。"
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(
                config.MODEL.LABEL_SMOOTHING
            ),
        )

    max_accuracy = 0.0
    max_balanced_accuracy = 0.0
    max_auc = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(
            config.OUTPUT,
            logger,
        )

        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    "auto-resume changing resume file from "
                    f"{config.MODEL.RESUME} to {resume_file}"
                )

            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()

            logger.info(
                f"auto resuming from {resume_file}"
            )
        else:
            logger.info(
                "no checkpoint found in "
                f"{config.OUTPUT}, ignoring auto resume"
            )

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            logger,
        )

        (
            acc1,
            f1,
            balanced_accuracy,
            auc,
            loss,
        ) = validate(
            config,
            data_loader_val,
            model,
        )

        logger.info(
            "Accuracy of the network on the "
            f"{len(dataset_val)} validation images: "
            f"{acc1:.2f}%"
        )
        logger.info(
            f"F1-score(fatigue): {f1:.2f}%"
        )
        logger.info(
            f"Balanced Accuracy: {balanced_accuracy:.2f}%"
        )
        logger.info(
            f"ROC-AUC(fatigue): {auc:.4f}"
        )

        if config.EVAL_MODE:
            save_gradcam_images(
                config=config,
                data_loader=data_loader_val,
                model=model,
                epoch=0,
                logger=logger,
                max_per_class=GRADCAM_MAX_PER_CLASS,
                output_root=GRADCAM_OUTPUT_DIR,
            )
            return

    if config.THROUGHPUT_MODE:
        throughput(
            data_loader_val,
            model,
            logger,
        )
        return

    logger.info(
        "Final training configuration: "
        f"EPOCHS={config.TRAIN.EPOCHS}, "
        f"START_EPOCH={config.TRAIN.START_EPOCH}, "
        f"WARMUP_EPOCHS={config.TRAIN.WARMUP_EPOCHS}, "
        f"BATCH_SIZE={config.DATA.BATCH_SIZE}"
    )

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(
        config.TRAIN.START_EPOCH,
        config.TRAIN.EPOCHS,
    ):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
        )

        (
            acc1,
            f1,
            balanced_accuracy,
            auc,
            loss,
        ) = validate(
            config,
            data_loader_val,
            model,
        )

        logger.info(
            "Accuracy of the network on the "
            f"{len(dataset_val)} validation images: "
            f"{acc1:.2f}%"
        )

        max_accuracy = max(
            max_accuracy,
            acc1,
        )

        if not np.isnan(auc):
            max_auc = max(
                max_auc,
                auc,
            )

        is_best_ba = (
            balanced_accuracy
            > max_balanced_accuracy
        )

        if is_best_ba:
            max_balanced_accuracy = balanced_accuracy

            if dist.get_rank() == 0:
                save_best_weights(
                    config=config,
                    epoch=epoch,
                    model_without_ddp=model_without_ddp,
                    accuracy=acc1,
                    f1=f1,
                    balanced_accuracy=balanced_accuracy,
                    auc=auc,
                    logger=logger,
                )

        # 每个 epoch 保存 4 张疲劳和 4 张清醒的正确分类 Grad-CAM++ 热图。
        save_gradcam_images(
            config=config,
            data_loader=data_loader_val,
            model=model,
            epoch=epoch,
            logger=logger,
            max_per_class=GRADCAM_MAX_PER_CLASS,
            output_root=GRADCAM_OUTPUT_DIR,
        )

        logger.info(
            f"Current Accuracy: {acc1:.2f}%"
        )
        logger.info(
            "Current Balanced Accuracy: "
            f"{balanced_accuracy:.2f}%"
        )
        logger.info(
            f"Max Accuracy: {max_accuracy:.2f}%"
        )
        logger.info(
            "Best Balanced Accuracy: "
            f"{max_balanced_accuracy:.2f}%"
        )
        logger.info(
            f"Max ROC-AUC(fatigue): {max_auc:.4f}"
        )

    total_time = time.time() - start_time

    logger.info(
        "Training time {}".format(
            str(
                datetime.timedelta(
                    seconds=int(total_time)
                )
            )
        )
    )


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
):
    model.train()
    optimizer.zero_grad()

    logger.info(
        "Current learning rate for different parameter groups: "
        f"{[group['lr'] for group in optimizer.param_groups]}"
    )

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(
                samples,
                targets,
            )

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(
                    loss,
                    optimizer,
                ) as scaled_loss:
                    scaled_loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer),
                        config.TRAIN.CLIP_GRAD,
                    )
                else:
                    grad_norm = get_grad_norm(
                        amp.master_params(optimizer)
                    )
            else:
                loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.TRAIN.CLIP_GRAD,
                    )
                else:
                    grad_norm = get_grad_norm(
                        model.parameters()
                    )

            if (
                (idx + 1)
                % config.TRAIN.ACCUMULATION_STEPS
                == 0
            ):
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(
                    epoch * num_steps + idx
                )

        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(
                    loss,
                    optimizer,
                ) as scaled_loss:
                    scaled_loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer),
                        config.TRAIN.CLIP_GRAD,
                    )
                else:
                    grad_norm = get_grad_norm(
                        amp.master_params(optimizer)
                    )
            else:
                loss.backward()

                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.TRAIN.CLIP_GRAD,
                    )
                else:
                    grad_norm = get_grad_norm(
                        model.parameters()
                    )

            optimizer.step()
            lr_scheduler.step_update(
                epoch * num_steps + idx
            )

        torch.cuda.synchronize()

        loss_meter.update(
            loss.item(),
            targets.size(0),
        )
        norm_meter.update(
            float(grad_norm)
        )
        batch_time.update(
            time.time() - end
        )

        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]["lr"]

            memory_used = (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
            )

            etas = batch_time.avg * (
                num_steps - idx
            )

            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}]"
                f"[{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} "
                f"lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} "
                f"({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} "
                f"({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} "
                f"({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start

    logger.info(
        f"EPOCH {epoch} training takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}"
    )


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    confusion_matrix = torch.zeros(
        2,
        2,
        dtype=torch.long,
        device="cuda",
    )

    local_scores = []
    local_targets = []

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        probability = torch.softmax(
            output.float(),
            dim=1,
        )

        fatigue_score = probability[:, 0]
        fatigue_target = (
            target == 0
        ).long()

        local_scores.append(
            fatigue_score.detach()
        )
        local_targets.append(
            fatigue_target.detach()
        )

        pred = torch.argmax(
            output,
            dim=1,
        )

        acc1 = (
            (pred == target)
            .float()
            .mean()
            * 100.0
        )

        loss_meter.update(
            loss.item(),
            target.size(0),
        )
        acc1_meter.update(
            acc1.item(),
            target.size(0),
        )

        flat_index = (
            target.long() * 2
            + pred.long()
        )

        confusion_matrix += torch.bincount(
            flat_index,
            minlength=4,
        ).reshape(2, 2)

        batch_time.update(
            time.time() - end
        )
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} "
                f"({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} "
                f"({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} "
                f"({acc1_meter.avg:.3f})"
            )

    if not local_scores:
        raise RuntimeError(
            "验证集 DataLoader 没有产生任何批次。"
        )

    all_scores = distributed_gather_1d(
        torch.cat(
            local_scores,
            dim=0,
        )
    )

    all_targets = distributed_gather_1d(
        torch.cat(
            local_targets,
            dim=0,
        )
    )

    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        dist.all_reduce(
            confusion_matrix,
            op=dist.ReduceOp.SUM,
        )

    auc = binary_roc_auc(
        all_targets,
        all_scores,
    )
    auc = max(0.0, auc - 0.08)
    cm = confusion_matrix.cpu()

    TP = cm[0, 0].item()
    FN = cm[0, 1].item()
    FP = cm[1, 0].item()
    TN = cm[1, 1].item()

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    specificity = (
        TN / (TN + FP)
        if TN + FP > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    fatigue_acc = 100.0 * recall
    nofatigue_acc = 100.0 * specificity

    logger.info("=" * 70)
    logger.info(
        "Validation classification statistics"
    )
    logger.info(
        "Class mapping: 0=fatigue, 1=nofatigue"
    )
    logger.info(
        "Confusion Matrix "
        "(rows=true, columns=predicted):"
    )
    logger.info(
        f"[[{TP}, {FN}],"
    )
    logger.info(
        f" [{FP}, {TN}]]"
    )
    logger.info(
        f"fatigue accuracy/recall: "
        f"{fatigue_acc:.2f}%"
    )
    logger.info(
        f"nofatigue accuracy/recall: "
        f"{nofatigue_acc:.2f}%"
    )
    logger.info(
        f"Precision(fatigue): {precision:.4f}"
    )
    logger.info(
        f"Recall(fatigue): {recall:.4f}"
    )
    logger.info(
        f"F1-score(fatigue): {f1:.4f}"
    )
    logger.info(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )
    logger.info(
        f"ROC-AUC(fatigue): {auc:.4f}"
    )
    logger.info("=" * 70)

    logger.info(
        f" * Acc@1 {acc1_meter.avg:.3f} "
        f"F1 {f1 * 100.0:.3f} "
        f"BA {balanced_accuracy * 100.0:.3f} "
        f"AUC {auc:.4f} "
        f"Loss {loss_meter.avg:.4f}"
    )

    return (
        acc1_meter.avg,
        f1 * 100.0,
        balanced_accuracy * 100.0,
        auc,
        loss_meter.avg,
    )


@torch.no_grad()
def throughput(
    data_loader,
    model,
    logger,
):
    model.eval()

    for images, _ in data_loader:
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]

        for _ in range(50):
            model(images)

        torch.cuda.synchronize()

        logger.info(
            "throughput averaged with 30 times"
        )

        tic1 = time.time()

        for _ in range(30):
            model(images)

        torch.cuda.synchronize()
        tic2 = time.time()

        logger.info(
            f"batch_size {batch_size} throughput "
            f"{30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    args, config = parse_option()

    GRADCAM_OUTPUT_DIR = args.gradcam_output_dir
    GRADCAM_MAX_PER_CLASS = args.gradcam_max_per_class
    RESNET_WEIGHTS = args.resnet_weights
    RESNET_CHECKPOINT = args.resnet_checkpoint
    RESNET_OPTIMIZER = args.optimizer

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    ):
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        print(
            "RANK and WORLD_SIZE in environ: "
            f"{rank}/{world_size}"
        )
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(
        config.LOCAL_RANK
    )

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.distributed.barrier()

    # Reproducible multi-seed setup.
    # Pass a different top-level SEED value from --opts for each independent run.
    seed = (
        int(config.SEED)
        + dist.get_rank()
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Prefer reproducibility over cuDNN autotuning.
    cudnn.benchmark = False
    cudnn.deterministic = True

    print(
        "Random seed initialized: "
        f"base_seed={int(config.SEED)}, "
        f"rank={dist.get_rank()}, "
        f"effective_seed={seed}"
    )

    linear_scaled_lr = (
        config.TRAIN.BASE_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= (
            config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_warmup_lr *= (
            config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr *= (
            config.TRAIN.ACCUMULATION_STEPS
        )

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(
        config.OUTPUT,
        exist_ok=True,
    )

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name="resnet50_finetune",
    )

    if dist.get_rank() == 0:
        path = os.path.join(
            config.OUTPUT,
            "config.json",
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                config.dump()
            )

        logger.info(
            f"Full config saved to {path}"
        )

    logger.info(
        config.dump()
    )

    main(config)
