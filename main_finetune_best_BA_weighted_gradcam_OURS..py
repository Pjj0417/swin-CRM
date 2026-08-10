# --------------------------------------------------------
# OURS
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
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAM = None
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
        "Swin Transformer training and evaluation script",
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


def find_swin_gradcam_target_layer(model_without_ddp):
    """
    自动查找 Swin 最后一个 stage 的最后一个 block 的 norm1 层。
    """
    if hasattr(model_without_ddp, "layers"):
        layers = model_without_ddp.layers
    elif (
        hasattr(model_without_ddp, "encoder")
        and hasattr(model_without_ddp.encoder, "layers")
    ):
        layers = model_without_ddp.encoder.layers
    else:
        raise AttributeError(
            "无法找到 Swin 的 layers 或 encoder.layers。"
        )

    if len(layers) == 0:
        raise RuntimeError("Swin layers 为空。")

    last_stage = layers[-1]

    if not hasattr(last_stage, "blocks") or len(last_stage.blocks) == 0:
        raise AttributeError(
            "Swin 最后一个 stage 不包含 blocks。"
        )

    last_block = last_stage.blocks[-1]

    if not hasattr(last_block, "norm1"):
        raise AttributeError(
            "Swin 最后一个 block 不包含 norm1。"
        )

    return last_block.norm1


def swin_reshape_transform(tensor):
    """
    把 Swin 特征转换为 Grad-CAM 需要的 [B, C, H, W]。
    """
    if tensor.ndim == 3:
        batch_size, token_count, channels = tensor.shape
        height = int(round(token_count ** 0.5))
        width = height

        if height * width != token_count:
            raise ValueError(
                "无法把 token 序列还原为方形特征图："
                f"{tuple(tensor.shape)}"
            )

        tensor = tensor.reshape(
            batch_size,
            height,
            width,
            channels,
        )
        return tensor.permute(0, 3, 1, 2)

    if tensor.ndim == 4:
        if tensor.shape[-1] > tensor.shape[1]:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    raise ValueError(
        "Grad-CAM 目标层输出维度不支持："
        f"{tuple(tensor.shape)}"
    )


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
    """
    只保留一份 Balanced Accuracy 最佳权重。
    """
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
    max_images=8,
):
    """
    在最佳 BA 更新时保存验证集 Grad-CAM 热图。

    每张结果图左边是原图，右边是热图叠加图。
    """
    if GradCAM is None:
        logger.warning(
            "未安装 grad-cam，跳过热图保存。"
            "安装：python -m pip install grad-cam opencv-python"
        )
        return

    if dist.get_rank() != 0:
        return

    model_without_ddp = (
        model.module
        if hasattr(model, "module")
        else model
    )

    output_dir = (
        Path(config.OUTPUT)
        / "gradcam"
        / f"best_epoch_{epoch:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {
        0: "fatigue",
        1: "nofatigue",
    }

    saved_count = 0
    model_without_ddp.eval()

    try:
        target_layer = find_swin_gradcam_target_layer(
            model_without_ddp
        )

        cam = GradCAM(
            model=model_without_ddp,
            target_layers=[target_layer],
            reshape_transform=swin_reshape_transform,
        )

        try:
            for images, targets in data_loader:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                with torch.no_grad():
                    logits = model_without_ddp(images)
                    predictions = torch.argmax(
                        logits,
                        dim=1,
                    )

                for batch_index in range(images.size(0)):
                    if saved_count >= max_images:
                        break

                    image = images[
                        batch_index:batch_index + 1
                    ]

                    true_class = int(
                        targets[batch_index].item()
                    )
                    predicted_class = int(
                        predictions[batch_index].item()
                    )

                    cam_targets = [
                        ClassifierOutputTarget(
                            predicted_class
                        )
                    ]

                    grayscale_cam = cam(
                        input_tensor=image,
                        targets=cam_targets,
                    )[0]

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

                    overlay = show_cam_on_image(
                        rgb_image,
                        grayscale_cam,
                        use_rgb=True,
                    )

                    original = (
                        rgb_image * 255.0
                    ).round().clip(
                        0,
                        255,
                    ).astype(np.uint8)

                    combined = np.concatenate(
                        [original, overlay],
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

                    filename = (
                        f"{saved_count:03d}_"
                        f"true_{true_name}_"
                        f"pred_{pred_name}_"
                        f"{correctness}.jpg"
                    )

                    save_path = output_dir / filename

                    success = cv2.imwrite(
                        str(save_path),
                        cv2.cvtColor(
                            combined,
                            cv2.COLOR_RGB2BGR,
                        ),
                    )

                    if not success:
                        raise RuntimeError(
                            f"Grad-CAM 保存失败：{save_path}"
                        )

                    saved_count += 1

                if saved_count >= max_images:
                    break

        finally:
            if hasattr(cam, "clear_hooks"):
                cam.clear_hooks()

    except Exception as error:
        logger.exception(
            "Grad-CAM 生成失败，但训练和最佳权重已保留。"
        )
        logger.warning(
            f"Grad-CAM error: {error}"
        )
        return

    logger.info(
        f"Saved {saved_count} Grad-CAM images to "
        f"{output_dir}"
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

                save_gradcam_images(
                    config=config,
                    data_loader=data_loader_val,
                    model=model,
                    epoch=epoch,
                    logger=logger,
                    max_images=8,
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
