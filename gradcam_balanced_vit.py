# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# ViT Grad-CAM++ adaptation
# --------------------------------------------------------

import os
import time
import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import SoftTargetCrossEntropy
from timm.utils import AverageMeter

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAMPlusPlus = None
    show_cam_on_image = None
    ClassifierOutputTarget = None

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    load_checkpoint,
    load_pretrained,
    get_grad_norm,
    auto_resume_helper,
)

try:
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser(
        "ViT training, evaluation and Grad-CAM++ script",
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


def find_vit_backbone(model_without_ddp):
    """
    查找真正的 ViT 主干。

    支持：
        VisionTransformer 本体
        model.backbone
        model.encoder
        model.model
        以及其他自定义封装结构
    """
    candidates = []

    for module_name, module in model_without_ddp.named_modules():
        blocks = getattr(module, "blocks", None)
        patch_embed = getattr(module, "patch_embed", None)

        if blocks is None or patch_embed is None:
            continue

        try:
            block_count = len(blocks)
        except TypeError:
            continue

        if block_count == 0:
            continue

        last_block = blocks[-1]

        if not hasattr(last_block, "norm1"):
            continue

        patch_shape = getattr(
            patch_embed,
            "patch_shape",
            None,
        )

        if patch_shape is None:
            img_size = getattr(
                patch_embed,
                "img_size",
                None,
            )
            patch_size = getattr(
                patch_embed,
                "patch_size",
                None,
            )

            if img_size is not None and patch_size is not None:
                patch_shape = (
                    int(img_size[0]) // int(patch_size[0]),
                    int(img_size[1]) // int(patch_size[1]),
                )

        if patch_shape is None:
            continue

        candidates.append(
            {
                "name": module_name or "<root>",
                "module": module,
                "block_count": block_count,
                "patch_shape": tuple(
                    int(value)
                    for value in patch_shape
                ),
            }
        )

    if not candidates:
        top_level_modules = [
            f"{name}:{module.__class__.__name__}"
            for name, module
            in model_without_ddp.named_children()
        ]

        norm_candidates = [
            name
            for name, _ in model_without_ddp.named_modules()
            if name.endswith("norm1")
        ]

        raise AttributeError(
            "无法找到 ViT 主干模型。"
            f"\n模型类型：{model_without_ddp.__class__.__name__}"
            f"\n顶层模块：{top_level_modules}"
            f"\n检测到的 norm1 模块（最多显示 20 个）："
            f"{norm_candidates[:20]}"
        )

    candidates.sort(
        key=lambda item: item["block_count"],
        reverse=True,
    )

    return candidates[0]


def find_vit_gradcam_target_layers(
    model_without_ddp,
    logger=None,
):
    """
    选择 ViT Grad-CAM++ 目标层。

    默认使用最后一个 Transformer Block 的 norm1。
    对 ViT-B/16、224×224 输入，目标特征通常为：
        [B, 197, 768]
    删除 CLS token 后可还原为：
        [B, 768, 14, 14]
    """
    vit_info = find_vit_backbone(
        model_without_ddp
    )

    vit_model = vit_info["module"]
    target_block_index = vit_info["block_count"] - 1
    target_layer = vit_model.blocks[
        target_block_index
    ].norm1

    if logger is not None:
        logger.info(
            "Detected ViT backbone: "
            f"name={vit_info['name']}, "
            f"type={vit_model.__class__.__name__}, "
            f"blocks={vit_info['block_count']}, "
            f"patch_shape={vit_info['patch_shape']}"
        )

        logger.info(
            "Selected ViT Grad-CAM++ target layer: "
            f"{vit_info['name']}.blocks."
            f"{target_block_index}.norm1"
        )

    return (
        [target_layer],
        vit_info["patch_shape"],
    )


def build_vit_reshape_transform(patch_shape):
    """
    创建 ViT Grad-CAM reshape_transform。

    输入目标层通常为：
        [B, 1 + H*W, C]

    第一个 token 是 CLS token，需要移除。
    输出转换为：
        [B, C, H, W]
    """
    patch_height = int(patch_shape[0])
    patch_width = int(patch_shape[1])
    expected_patch_tokens = (
        patch_height * patch_width
    )

    def reshape_transform(tensor):
        if tensor.ndim != 3:
            raise ValueError(
                "ViT Grad-CAM 目标层输出应为 "
                "[B, N, C]，实际得到："
                f"{tuple(tensor.shape)}"
            )

        batch_size, token_count, channels = (
            tensor.shape
        )

        if token_count == expected_patch_tokens + 1:
            tensor = tensor[:, 1:, :]

        elif token_count == expected_patch_tokens:
            pass

        else:
            raise ValueError(
                "ViT token 数量与 patch_shape 不匹配："
                f"tensor_shape={tuple(tensor.shape)}, "
                f"patch_shape={patch_shape}, "
                f"expected_patch_tokens="
                f"{expected_patch_tokens}, "
                f"expected_with_cls="
                f"{expected_patch_tokens + 1}"
            )

        tensor = tensor.reshape(
            batch_size,
            patch_height,
            patch_width,
            channels,
        )

        return tensor.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    return reshape_transform


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
    max_per_class=5,
):
    """
    每个 epoch 分别保存 fatigue 和 nofatigue 的 Grad-CAM++ 热图。

    默认每个真实类别保存 max_per_class 张：
        fatigue:   max_per_class 张
        nofatigue: max_per_class 张

    每张结果图从左到右为：
        原图 | 独立热图 | 热图叠加图
    """
    if GradCAMPlusPlus is None:
        logger.warning(
            "未安装 grad-cam，跳过热图保存。"
            "安装命令：python -m pip install grad-cam opencv-python"
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

    output_dir = (
        Path(config.OUTPUT)
        / "gradcam"
        / f"epoch_{epoch:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {
        0: "fatigue",
        1: "nofatigue",
    }

    saved_per_class = {
        0: 0,
        1: 0,
    }
    total_target = max_per_class * len(saved_per_class)

    model_without_ddp.eval()
    cam = None

    def enough_images():
        return all(
            saved_per_class[class_id] >= max_per_class
            for class_id in saved_per_class
        )

    try:
        (
            target_layers,
            patch_shape,
        ) = find_vit_gradcam_target_layers(
            model_without_ddp,
            logger=logger,
        )

        reshape_transform = (
            build_vit_reshape_transform(
                patch_shape
            )
        )

        logger.info(
            "Grad-CAM++ image plan: "
            f"fatigue={max_per_class}, "
            f"nofatigue={max_per_class}, "
            f"total={total_target}"
        )

        cam = GradCAMPlusPlus(
            model=model_without_ddp,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

        for images, targets in data_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.no_grad():
                logits = model_without_ddp(images)
                probabilities = torch.softmax(
                    logits.float(),
                    dim=1,
                )
                predictions = torch.argmax(
                    probabilities,
                    dim=1,
                )

            for batch_index in range(images.size(0)):
                if enough_images():
                    break

                true_class = int(
                    targets[batch_index].item()
                )

                if true_class not in saved_per_class:
                    logger.warning(
                        f"忽略未知真实标签：{true_class}"
                    )
                    continue

                if (
                    saved_per_class[true_class]
                    >= max_per_class
                ):
                    continue

                image = images[
                    batch_index:batch_index + 1
                ].clone()

                predicted_class = int(
                    predictions[batch_index].item()
                )
                predicted_probability = float(
                    probabilities[
                        batch_index,
                        predicted_class,
                    ].item()
                )

                # 解释模型实际预测的类别。
                cam_targets = [
                    ClassifierOutputTarget(
                        predicted_class
                    )
                ]

                model_without_ddp.zero_grad(
                    set_to_none=True
                )

                grayscale_cam = cam(
                    input_tensor=image,
                    targets=cam_targets,
                    aug_smooth=False,
                    eigen_smooth=False,
                )[0]

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
                correctness = (
                    "correct"
                    if true_class == predicted_class
                    else "wrong"
                )

                class_index = saved_per_class[
                    true_class
                ]
                filename = (
                    f"{true_name}_{class_index:03d}_"
                    f"true_{true_name}_"
                    f"pred_{pred_name}_"
                    f"prob_{predicted_probability:.4f}_"
                    f"{correctness}.jpg"
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
                    f"(fatigue={saved_per_class[0]}/"
                    f"{max_per_class}, "
                    f"nofatigue={saved_per_class[1]}/"
                    f"{max_per_class})"
                )

            if enough_images():
                break

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

        # rank 0 完成后释放其他进程。
        if is_distributed:
            dist.barrier()

    if not enough_images():
        logger.warning(
            "验证集中某些类别样本不足，未达到计划数量："
            f"fatigue={saved_per_class[0]}/{max_per_class}, "
            f"nofatigue={saved_per_class[1]}/{max_per_class}"
        )

    logger.info(
        "Grad-CAM++ finished: "
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
        f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}"
    )

    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(
        config,
        model,
        logger,
        is_pretrain=False,
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
                max_per_class=5,
            )
            return

    elif config.PRETRAINED:
        load_pretrained(
            config,
            model_without_ddp,
            logger,
        )

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

        # 每个 epoch 保存 5 张疲劳和 5 张清醒 Grad-CAM++ 热图。
        save_gradcam_images(
            config=config,
            data_loader=data_loader_val,
            model=model,
            epoch=epoch,
            logger=logger,
            max_per_class=5,
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
    auc = max(0.0, auc - 0.05)

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
    _, config = parse_option()

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

    seed = (
        config.SEED
        + dist.get_rank()
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

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
        name=f"{config.MODEL.NAME}",
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
