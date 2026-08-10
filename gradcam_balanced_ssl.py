# --------------------------------------------------------
# DINOv2 backbone: self-supervised pretrained weights
# Classes: 0=fatigue, 1=nofatigue
# --------------------------------------------------------

import os
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
from utils import load_checkpoint, get_grad_norm, auto_resume_helper
from models.dinov2_model import build_dinov2

try:
    from apex import amp
except ImportError:
    amp = None


GRADCAM_OUTPUT_DIR = None
GRADCAM_MAX_PER_CLASS = 4
DINOV2_MODEL_NAME = "dinov2_vits14"
FREEZE_BACKBONE = False
OPTIMIZER_NAME = "adamw"


def parse_option():
    parser = argparse.ArgumentParser(
        "DINOv2 training, evaluation and Grad-CAM++ script",
        add_help=True,
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
        default="O0",
        choices=["O0", "O1", "O2"],
    )
    parser.add_argument("--output", default="output", type=str)
    parser.add_argument("--tag")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")

    parser.add_argument(
        "--dinov2-model",
        type=str,
        default="dinov2_vits14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze DINOv2 backbone and train only the classifier.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
    )
    parser.add_argument(
        "--gradcam-output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gradcam-max-per-class",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--skip-gradcam",
        action="store_true",
        help="Skip Grad-CAM++ generation during training.",
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
    sizes_tensor = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes_tensor, local_size)

    sizes = [int(item.item()) for item in sizes_tensor]
    max_size = max(sizes)

    if tensor.numel() < max_size:
        tensor = torch.cat(
            [tensor, tensor.new_zeros(max_size - tensor.numel())],
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
        [part[:size] for part, size in zip(gathered, sizes)],
        dim=0,
    )


def build_optimizer(config, model, optimizer_name, logger):
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": float(config.TRAIN.WEIGHT_DECAY),
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    learning_rate = float(config.TRAIN.BASE_LR)
    optimizer_name = str(optimizer_name).lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    logger.info(
        "Created optimizer: "
        f"name={optimizer_name}, "
        f"lr={learning_rate:.8f}, "
        f"weight_decay={config.TRAIN.WEIGHT_DECAY}, "
        f"decay_tensors={len(decay_parameters)}, "
        f"no_decay_tensors={len(no_decay_parameters)}"
    )
    return optimizer


def dinov2_reshape_transform(tensor):
    if tensor.ndim != 3:
        return tensor

    token_count = tensor.shape[1]

    # DINOv2 block output generally contains CLS + patch tokens.
    patch_tokens = tensor[:, 1:, :]
    patch_count = patch_tokens.shape[1]
    spatial_size = int(round(patch_count ** 0.5))

    if spatial_size * spatial_size != patch_count:
        # Some hooks may already return patch tokens only.
        spatial_size = int(round(token_count ** 0.5))
        if spatial_size * spatial_size != token_count:
            raise RuntimeError(
                f"Cannot reshape {token_count} DINOv2 tokens "
                "into a square feature map."
            )
        patch_tokens = tensor

    return (
        patch_tokens
        .reshape(
            patch_tokens.shape[0],
            spatial_size,
            spatial_size,
            patch_tokens.shape[2],
        )
        .permute(0, 3, 1, 2)
    )


def find_dinov2_gradcam_target_layers(model_without_ddp, logger=None):
    if not hasattr(model_without_ddp, "backbone"):
        raise AttributeError("Model does not contain a DINOv2 backbone.")

    backbone = model_without_ddp.backbone
    if not hasattr(backbone, "blocks") or len(backbone.blocks) == 0:
        raise AttributeError("DINOv2 backbone does not contain blocks.")

    target_layer = backbone.blocks[-1].norm1

    if logger is not None:
        logger.info(
            "Selected DINOv2 Grad-CAM++ target layer: "
            "backbone.blocks[-1].norm1"
        )

    return [target_layer]


def denormalize_imagenet_image(image_tensor):
    mean = image_tensor.new_tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)
    std = image_tensor.new_tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


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
        "architecture": DINOV2_MODEL_NAME,
        "initialization": "DINOv2_self_supervised",
    }

    torch.save(checkpoint, save_path)
    logger.info(
        "Saved new best Balanced Accuracy weights: "
        f"{save_path} "
        f"(epoch={epoch}, ACC={accuracy:.2f}%, "
        f"F1={f1:.2f}%, BA={balanced_accuracy:.2f}%, "
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
    if GradCAMPlusPlus is None:
        logger.warning(
            "grad-cam is not installed; skipping Grad-CAM++. "
            "Install with: python -m pip install grad-cam opencv-python"
        )
        return

    if max_per_class is None:
        max_per_class = GRADCAM_MAX_PER_CLASS
    if output_root is None:
        output_root = GRADCAM_OUTPUT_DIR

    max_per_class = int(max_per_class)
    if max_per_class <= 0:
        return

    is_distributed = (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )

    if is_distributed:
        dist.barrier()

    if is_distributed and dist.get_rank() != 0:
        dist.barrier()
        return

    model_without_ddp = (
        model.module
        if hasattr(model, "module")
        else model
    )

    gradcam_root = (
        Path(output_root).expanduser()
        if output_root
        else Path(config.OUTPUT) / "gradcam"
    )
    output_dir = gradcam_root / f"epoch_{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = {0: "fatigue", 1: "nofatigue"}
    selected_per_class = {0: [], 1: []}
    saved_per_class = {0: 0, 1: 0}
    cam = None
    model_without_ddp.eval()

    def enough_selected():
        return all(
            len(selected_per_class[class_id]) >= max_per_class
            for class_id in selected_per_class
        )

    try:
        target_layers = find_dinov2_gradcam_target_layers(
            model_without_ddp,
            logger,
        )

        cam = GradCAMPlusPlus(
            model=model_without_ddp,
            target_layers=target_layers,
            reshape_transform=dinov2_reshape_transform,
        )

        with torch.no_grad():
            for images, targets in data_loader:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                logits = model_without_ddp(images)
                probabilities = torch.softmax(logits.float(), dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                for class_id in selected_per_class:
                    remaining = (
                        max_per_class
                        - len(selected_per_class[class_id])
                    )
                    if remaining <= 0:
                        continue

                    correct_indices = torch.nonzero(
                        (targets == class_id)
                        & (predictions == class_id),
                        as_tuple=False,
                    ).view(-1)

                    if correct_indices.numel() == 0:
                        continue

                    confidences = probabilities[
                        correct_indices,
                        class_id,
                    ]
                    order = torch.argsort(
                        confidences,
                        descending=True,
                    )
                    correct_indices = correct_indices[order]

                    for index_tensor in correct_indices[:remaining]:
                        batch_index = int(index_tensor.item())
                        selected_per_class[class_id].append(
                            {
                                "image": images[batch_index].detach().clone(),
                                "true_class": class_id,
                                "predicted_class": class_id,
                                "confidence": float(
                                    probabilities[
                                        batch_index,
                                        class_id,
                                    ].item()
                                ),
                            }
                        )

                if enough_selected():
                    break

        selected_samples = []
        for class_id in sorted(selected_per_class):
            selected_samples.extend(selected_per_class[class_id])

        if not selected_samples:
            logger.warning(
                "No correctly classified validation samples found for Grad-CAM++."
            )
            return

        input_batch = torch.stack(
            [sample["image"] for sample in selected_samples],
            dim=0,
        )
        cam_targets = [
            ClassifierOutputTarget(sample["predicted_class"])
            for sample in selected_samples
        ]

        model_without_ddp.zero_grad(set_to_none=True)
        grayscale_cams = cam(
            input_tensor=input_batch,
            targets=cam_targets,
            aug_smooth=False,
            eigen_smooth=False,
        )

        for sample_index, sample in enumerate(selected_samples):
            grayscale_cam = np.nan_to_num(
                grayscale_cams[sample_index],
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            grayscale_cam = np.clip(
                grayscale_cam,
                0.0,
                1.0,
            ).astype(np.float32)

            rgb_image = denormalize_imagenet_image(
                input_batch[sample_index]
            )
            rgb_image = (
                rgb_image
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            rgb_image = np.clip(rgb_image, 0.0, 1.0)

            overlay = show_cam_on_image(
                rgb_image,
                grayscale_cam,
                use_rgb=True,
                image_weight=0.55,
            )

            original = (
                rgb_image * 255.0
            ).round().clip(0, 255).astype(np.uint8)

            heatmap_uint8 = (
                grayscale_cam * 255.0
            ).round().clip(0, 255).astype(np.uint8)

            heatmap_bgr = cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET,
            )
            heatmap_rgb = cv2.cvtColor(
                heatmap_bgr,
                cv2.COLOR_BGR2RGB,
            )

            combined = np.concatenate(
                [original, heatmap_rgb, overlay],
                axis=1,
            )

            true_class = sample["true_class"]
            predicted_class = sample["predicted_class"]
            confidence = sample["confidence"]

            filename = (
                f"sample_{saved_per_class[true_class]:03d}_"
                f"true_{class_names[true_class]}_"
                f"pred_{class_names[predicted_class]}_"
                f"prob_{confidence:.4f}_correct.jpg"
            )
            save_path = output_dir / filename

            success = cv2.imwrite(
                str(save_path),
                cv2.cvtColor(
                    combined,
                    cv2.COLOR_RGB2BGR,
                ),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if not success:
                raise RuntimeError(f"Failed to save Grad-CAM image: {save_path}")

            saved_per_class[true_class] += 1
            logger.info(f"Saved Grad-CAM++ image: {save_path}")

    except Exception as error:
        logger.exception(
            "Grad-CAM++ failed, but training and checkpoint saving continue."
        )
        logger.warning(f"Grad-CAM++ error: {error}")

    finally:
        if cam is not None and hasattr(cam, "clear_hooks"):
            cam.clear_hooks()

        model_without_ddp.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_distributed:
            dist.barrier()


def main(config, skip_gradcam=False):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config, logger, is_pretrain=False)

    logger.info(f"Creating self-supervised model: {DINOV2_MODEL_NAME}")

    model = build_dinov2(
        num_classes=int(config.MODEL.NUM_CLASSES),
        model_name=DINOV2_MODEL_NAME,
        drop_rate=float(config.MODEL.DROP_RATE),
        freeze_backbone=FREEZE_BACKBONE,
    )
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(
        config,
        model,
        OPTIMIZER_NAME,
        logger,
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
    logger.info(f"number of trainable params: {n_parameters}")

    lr_scheduler = build_scheduler(
        config,
        optimizer,
        len(data_loader_train),
    )

    targets_tensor = torch.as_tensor(
        dataset_train.targets,
        dtype=torch.long,
    )
    class_counts = torch.bincount(
        targets_tensor,
        minlength=len(dataset_train.classes),
    )

    fatigue_count = int(
        class_counts[
            dataset_train.class_to_idx["fatigue"]
        ].item()
    )
    nofatigue_count = int(
        class_counts[
            dataset_train.class_to_idx["nofatigue"]
        ].item()
    )
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
        "Training class counts: "
        f"fatigue={fatigue_count}, "
        f"nofatigue={nofatigue_count}"
    )
    logger.info(
        "Training class weights: "
        f"fatigue={class_weights[0].item():.6f}, "
        f"nofatigue={class_weights[1].item():.6f}"
    )

    if mixup_fn is not None:
        logger.warning(
            "Mixup/CutMix is enabled. "
            "SoftTargetCrossEntropy is used and class weights "
            "are not applied to soft labels."
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(config.MODEL.LABEL_SMOOTHING),
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
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            logger,
        )

        acc1, f1, balanced_accuracy, auc, loss = validate(
            config,
            data_loader_val,
            model,
        )

        if config.EVAL_MODE:
            if not skip_gradcam:
                save_gradcam_images(
                    config,
                    data_loader_val,
                    model,
                    epoch=0,
                    logger=logger,
                )
            return

    if config.THROUGHPUT_MODE:
        throughput(
            data_loader_val,
            model,
            logger,
        )
        return

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(
        config.TRAIN.START_EPOCH,
        config.TRAIN.EPOCHS,
    ):
        if hasattr(data_loader_train.sampler, "set_epoch"):
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

        acc1, f1, balanced_accuracy, auc, loss = validate(
            config,
            data_loader_val,
            model,
        )

        max_accuracy = max(max_accuracy, acc1)
        if not np.isnan(auc):
            max_auc = max(max_auc, auc)

        is_best_ba = balanced_accuracy > max_balanced_accuracy
        if is_best_ba:
            max_balanced_accuracy = balanced_accuracy
            if dist.get_rank() == 0:
                save_best_weights(
                    config,
                    epoch,
                    model_without_ddp,
                    acc1,
                    f1,
                    balanced_accuracy,
                    auc,
                    logger,
                )

        if not skip_gradcam:
            save_gradcam_images(
                config,
                data_loader_val,
                model,
                epoch,
                logger,
            )

        logger.info(f"Current Accuracy: {acc1:.2f}%")
        logger.info(
            f"Current Balanced Accuracy: {balanced_accuracy:.2f}%"
        )
        logger.info(f"Max Accuracy: {max_accuracy:.2f}%")
        logger.info(
            f"Best Balanced Accuracy: {max_balanced_accuracy:.2f}%"
        )
        logger.info(f"Max ROC-AUC(fatigue): {max_auc:.4f}")

    total_time = time.time() - start_time
    logger.info(
        "Training time "
        + str(datetime.timedelta(seconds=int(total_time)))
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
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        accumulation_steps = max(
            1,
            int(config.TRAIN.ACCUMULATION_STEPS),
        )
        loss_for_backward = loss / accumulation_steps

        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(
                loss_for_backward,
                optimizer,
            ) as scaled_loss:
                scaled_loss.backward()

            grad_parameters = amp.master_params(optimizer)
        else:
            loss_for_backward.backward()
            grad_parameters = model.parameters()

        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                grad_parameters,
                config.TRAIN.CLIP_GRAD,
            )
        else:
            grad_norm = get_grad_norm(grad_parameters)

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step_update(
                epoch * num_steps + idx
            )

        torch.cuda.synchronize()

        batch_size = samples.size(0)
        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(float(grad_norm))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]["lr"]
            memory_used = (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
            )
            etas = batch_time.avg * (num_steps - idx)

            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}]"
                f"[{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} "
                f"lr {lr:.8f}\t"
                f"time {batch_time.val:.4f} "
                f"({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} "
                f"({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} "
                f"({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    # Flush remaining gradients when the final batch is not divisible.
    accumulation_steps = max(
        1,
        int(config.TRAIN.ACCUMULATION_STEPS),
    )
    if num_steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    logger.info(
        f"EPOCH {epoch} training takes "
        f"{datetime.timedelta(seconds=int(time.time() - start))}"
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

        probability = torch.softmax(output.float(), dim=1)
        fatigue_score = probability[:, 0]
        fatigue_target = (target == 0).long()

        local_scores.append(fatigue_score.detach())
        local_targets.append(fatigue_target.detach())

        pred = torch.argmax(output, dim=1)
        acc1 = (pred == target).float().mean() * 100.0

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        flat_index = target.long() * 2 + pred.long()
        confusion_matrix += torch.bincount(
            flat_index,
            minlength=4,
        ).reshape(2, 2)

        batch_time.update(time.time() - end)
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
            "Validation DataLoader produced no batches."
        )

    all_scores = distributed_gather_1d(
        torch.cat(local_scores, dim=0)
    )
    all_targets = distributed_gather_1d(
        torch.cat(local_targets, dim=0)
    )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(
            confusion_matrix,
            op=dist.ReduceOp.SUM,
        )
    auc = binary_roc_auc(all_targets, all_scores)
    cm = confusion_matrix.cpu()

    TP = cm[0, 0].item()
    FN = cm[0, 1].item()
    FP = cm[1, 0].item()
    TN = cm[1, 1].item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    logger.info("=" * 70)
    logger.info("Class mapping: 0=fatigue, 1=nofatigue")
    logger.info(
        f"Confusion Matrix: [[{TP}, {FN}], [{FP}, {TN}]]"
    )
    logger.info(f"Precision(fatigue): {precision:.4f}")
    logger.info(f"Recall(fatigue): {recall:.4f}")
    logger.info(f"F1-score(fatigue): {f1:.4f}")
    logger.info(
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )
    logger.info(f"ROC-AUC(fatigue): {auc:.4f}")
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
def throughput(data_loader, model, logger):
    model.eval()

    for images, _ in data_loader:
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]

        for _ in range(50):
            model(images)

        torch.cuda.synchronize()
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
    DINOV2_MODEL_NAME = args.dinov2_model
    FREEZE_BACKBONE = args.freeze_backbone
    OPTIMIZER_NAME = args.optimizer

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "Apex AMP is not installed."

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Preserve the original SimMIM learning-rate scaling behavior.
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
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name="dinov2_finetune",
    )

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w", encoding="utf-8") as file:
            file.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    main(config, skip_gradcam=args.skip_gradcam)
