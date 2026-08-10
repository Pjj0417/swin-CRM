# --------------------------------------------------------
# Swin-B + MAE + DWMM self-supervised pretraining and fatigue fine-tuning
# Deformable DW-MSA / DSW-MSA Swin-B + MAE reconstruction
# Classes: 0=fatigue, 1=nofatigue
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image
from PIL import Image

# Improved version:
# - keeps ConvNeXt + SimMIM compatibility
# - keeps Grad-CAM++ explainability pipeline
# - adds safer runtime defaults for modern CUDA GPUs
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

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
from models.swin_mae_dwmm_model import build_swin_mae_dwmm, SwinMAEDWMMPretrain

try:
    from apex import amp
except ImportError:
    amp = None


GRADCAM_OUTPUT_DIR = None
GRADCAM_MAX_PER_CLASS = 8
SWIN_MAE_DWMM_MODEL_NAME = "swin_base_patch4_window7_224_dwmm"
SWIN_MAE_DWMM_PRETRAINED_PATH = None
FREEZE_BACKBONE = False
OPTIMIZER_NAME = "adamw"
DEFORMABLE_OFFSET_SCALE = 1.0

# Improved training defaults
# - avoids unnecessary graph retention
# - makes long training runs more stable
ENABLE_CUDA_BENCHMARK = True

INFERENCE_DISPLAY_DIVISOR = 1.0
GRADCAM_BEST_ONLY = False  # kept for compatibility; every fine-tune epoch is visualized


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin-MAE+DWMM training, evaluation and Grad-CAM++ script",
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
        "--swin-mae-dwmm-model",
        type=str,
        default="swin_base_patch4_window7_224_dwmm",
        choices=[
            "swin_base_patch4_window7_224_dwmm",
        ],
    )
    parser.add_argument(
        "--deformable-offset-scale",
        type=float,
        default=1.0,
        help="Maximum DW-MSA/DSW-MSA local offset scale.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze MAE-pretrained DWMM Swin-B and train only classifier.",
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
        default=8,
    )
    parser.add_argument(
        "--skip-gradcam",
        action="store_true",
        help="Skip Grad-CAM++ generation during fine-tuning.",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["pretrain", "finetune", "full"],
        help=(
            "pretrain: MAE+DWMM only; "
            "finetune: fatigue/nofatigue only; "
            "full: 100-epoch MAE+DWMM then automatic fine-tuning."
        ),
    )
    parser.add_argument(
        "--held-out-subject",
        type=str,
        default="jiang_pengfei",
    )
    parser.add_argument(
        "--pretrain-output",
        type=str,
        default="./output/swin_mae_dwmm_pretrain/subject13",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--pretrain-warmup-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=8,
        help=(
            "DWMM adds grid_sample and offset branches. "
            "8 is a conservative single-GPU starting point."
        ),
    )
    parser.add_argument(
        "--pretrain-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--pretrain-min-lr",
        type=float,
        default=0.000001,
    )
    parser.add_argument(
        "--pretrain-weight-decay",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--mask-window",
        type=int,
        default=4,
        help=(
            "Mask grouping on 56x56 patch grid. "
            "4 gives 14x14=196 groups, so 75% masking is exactly 147/196."
        ),
    )
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--pretrain-drop-path",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--pretrain-use-checkpoint",
        action="store_true",
        help="Use activation checkpointing inside DWMM blocks during MAE pretraining.",
    )
    parser.add_argument(
        "--pretrain-print-freq",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--pretrain-clip-grad",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--pretrain-amp",
        action="store_true",
        help="Use native torch autocast/GradScaler during MAE+DWMM pretraining.",
    )
    parser.add_argument(
        "--pretrain-resume",
        type=str,
        default="",
        help="Optional MAE+DWMM pretraining checkpoint to resume.",
    )

    parser.add_argument(
        "--recon-preview-count",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--recon-save-every",
        type=int,
        default=1,
        help="Save reconstruction preview every N pretraining epochs.",
    )
    parser.add_argument(
        "--recon-preview-seed",
        type=int,
        default=20260808,
        help="Fixed mask RNG for comparable reconstruction previews.",
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

    no_decay_tokens = (
        "relative_position_bias_table",
        "deform_gate",
        "lepe_scale",
        "output_refine_scale",
    )

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or any(
                token in name
                for token in no_decay_tokens
            )
        ):
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


def swin_mae_dwmm_reshape_transform(tensor):
    """
    Fine-grained Grad-CAM target is normally the stage-3 final block:
    [B, 196, 512] -> [B, 512, 14, 14].
    The generic square-token reshape also supports 7x7 tensors.
    """
    if tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        spatial = int(round(tokens ** 0.5))

        if spatial * spatial != tokens:
            raise RuntimeError(
                f"Cannot reshape {tokens} Swin tokens into a square map."
            )

        return (
            tensor.reshape(
                batch,
                spatial,
                spatial,
                channels,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    if tensor.ndim == 4:
        # Current timm Swin commonly uses BHWC internally.
        if tensor.shape[-1] > tensor.shape[1]:
            return (
                tensor.permute(0, 3, 1, 2)
                .contiguous()
            )
        return tensor

    return tensor


def find_swin_mae_dwmm_gradcam_target_layers(
    model_without_ddp,
    logger=None,
):
    if not hasattr(model_without_ddp, "backbone"):
        raise AttributeError(
            "Model does not contain a Swin-MAE+DWMM backbone."
        )

    backbone = model_without_ddp.backbone

    # Use the last block of stage 3 (index -2), before that stage's
    # PatchMerging. For 224x224 Swin-B this activation is 14x14 instead
    # of the final stage's 7x7, giving visibly finer spatial localization.
    # This hook is used only when Grad-CAM++ is generated; the classifier
    # forward path and throughput benchmark are unchanged.
    fine_stage = backbone.layers[-2]
    fine_block = fine_stage.blocks[-1]
    target_layer = fine_block.norm2

    if logger is not None:
        logger.info(
            "Selected fine-grained Swin-MAE+DWMM Grad-CAM++ target layer: "
            "backbone.layers[-2].blocks[-1].norm2 (14x14)"
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
        "architecture": "Swin-B + MAE + DWMM (DW-MSA/DSW-MSA)",
        "initialization": "SwinMAE_DWMM_LOSO_self_supervised_pretrained",
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
    fallback_per_class = {0: [], 1: []}
    saved_per_class = {0: 0, 1: 0}
    cam = None
    model_without_ddp.eval()

    def enough_selected():
        return all(
            len(selected_per_class[class_id]) >= max_per_class
            for class_id in selected_per_class
        )

    try:
        target_layers = find_swin_mae_dwmm_gradcam_target_layers(
            model_without_ddp,
            logger,
        )

        cam = GradCAMPlusPlus(
            model=model_without_ddp,
            target_layers=target_layers,
            reshape_transform=swin_mae_dwmm_reshape_transform,
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

                    true_indices = torch.nonzero(
                        targets == class_id,
                        as_tuple=False,
                    ).view(-1)

                    for index_tensor in true_indices:
                        if len(fallback_per_class[class_id]) >= max_per_class:
                            break
                        batch_index = int(index_tensor.item())
                        pred_class = int(predictions[batch_index].item())
                        fallback_per_class[class_id].append(
                            {
                                "image": images[batch_index].detach().clone(),
                                "true_class": class_id,
                                "predicted_class": pred_class,
                                "confidence": float(
                                    probabilities[
                                        batch_index,
                                        pred_class,
                                    ].item()
                                ),
                            }
                        )

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
            chosen = list(selected_per_class[class_id])

            if len(chosen) < max_per_class:
                for sample in fallback_per_class[class_id]:
                    if len(chosen) >= max_per_class:
                        break
                    chosen.append(sample)

            selected_samples.extend(chosen[:max_per_class])

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



# ---------------------------------------------------------------------
# Swin-MAE+DWMM self-supervised pretraining
# ---------------------------------------------------------------------

MAE_DWMM_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

MAE_DWMM_IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)
MAE_DWMM_IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


def _dataset_path_from_item(
    item,
):
    if isinstance(
        item,
        (str, Path),
    ):
        return str(item)

    if isinstance(
        item,
        (tuple, list),
    ):
        if not item:
            return None
        first = item[0]
        if isinstance(
            first,
            (str, Path),
        ):
            return str(first)

    if isinstance(
        item,
        dict,
    ):
        for key in (
            "path",
            "image_path",
            "img_path",
            "file",
            "filename",
            "image",
        ):
            value = item.get(key)
            if isinstance(
                value,
                (str, Path),
            ):
                return str(value)

    return None


def extract_loso_train_paths(
    dataset_train,
):
    """
    Recover EXACTLY the images selected by the existing supervised LOSO split.
    Recursive DATA_PATH scanning is intentionally forbidden.
    """
    if (
        hasattr(dataset_train, "dataset")
        and hasattr(dataset_train, "indices")
    ):
        parent_paths = (
            extract_loso_train_paths(
                dataset_train.dataset
            )
        )
        indices = list(
            dataset_train.indices
        )
        paths = [
            parent_paths[int(i)]
            for i in indices
        ]

        if len(paths) != len(
            dataset_train
        ):
            raise RuntimeError(
                "LOSO Subset path extraction count mismatch: "
                f"paths={len(paths)}, "
                f"dataset={len(dataset_train)}"
            )
        return paths

    candidate_attrs = (
        "samples",
        "imgs",
        "image_paths",
        "paths",
        "files",
        "records",
        "data_list",
        "items",
    )

    for attr in candidate_attrs:
        if not hasattr(
            dataset_train,
            attr,
        ):
            continue

        values = getattr(
            dataset_train,
            attr,
        )

        try:
            values = list(values)
        except TypeError:
            continue

        paths = []
        failed = False

        for item in values:
            path = (
                _dataset_path_from_item(
                    item
                )
            )
            if path is None:
                failed = True
                break
            paths.append(path)

        if (
            not failed
            and len(paths)
            == len(dataset_train)
        ):
            return paths

    if hasattr(
        dataset_train,
        "data",
    ):
        try:
            values = list(
                dataset_train.data
            )
        except Exception:
            values = []

        paths = []
        for item in values:
            path = (
                _dataset_path_from_item(
                    item
                )
            )
            if path is None:
                paths = []
                break
            paths.append(path)

        if len(paths) == len(
            dataset_train
        ):
            return paths

    public_attrs = [
        name
        for name in dir(
            dataset_train
        )
        if not name.startswith("_")
    ]

    raise RuntimeError(
        "Could not recover raw image paths from the existing LOSO "
        "dataset_train object. Refusing recursive scanning because "
        "that can leak the held-out subject. "
        f"dataset type={type(dataset_train).__name__}; "
        f"length={len(dataset_train)}; "
        f"available attrs={public_attrs[:80]}"
    )


class MAEDWMMLosoDataset(
    Dataset
):
    def __init__(
        self,
        image_paths,
        held_out_subject,
        expected_count=None,
    ):
        self.held_out_subject = str(
            held_out_subject
        )
        self.image_paths = [
            Path(path).expanduser()
            for path in image_paths
        ]

        if expected_count is not None:
            expected_count = int(
                expected_count
            )
            if (
                len(self.image_paths)
                != expected_count
            ):
                raise RuntimeError(
                    "MAE+DWMM LOSO membership mismatch: "
                    f"extracted={len(self.image_paths)}, "
                    f"expected={expected_count}. "
                    "Pretraining aborted."
                )

        if not self.image_paths:
            raise RuntimeError(
                "No images found in verified LOSO training split."
            )

        missing = [
            str(path)
            for path in self.image_paths
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Some LOSO training image paths do not exist. "
                f"First missing paths: {missing[:10]}"
            )

        # Reconstruction-friendly augmentation: no Mixup/CutMix/RandomErase.
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.60, 1.00),
                    ratio=(0.85, 1.15),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MAE_DWMM_IMAGENET_MEAN,
                    std=MAE_DWMM_IMAGENET_STD,
                ),
            ]
        )

        print("=" * 72)
        print(
            "[Swin-MAE+DWMM] VERIFIED LOSO self-supervised dataset"
        )
        print(
            "[Swin-MAE+DWMM] Membership source: "
            "existing build_loader(...).dataset_train"
        )
        print(
            "[Swin-MAE+DWMM] Held-out subject controlled by original LOSO split: "
            f"{self.held_out_subject}"
        )
        print(
            "[Swin-MAE+DWMM] Pretraining images: "
            f"{len(self.image_paths)}"
        )
        print(
            "[Swin-MAE+DWMM] Recursive DATA_PATH scanning: DISABLED"
        )
        print("=" * 72)

    def __len__(
        self,
    ):
        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index,
    ):
        path = self.image_paths[
            index
        ]

        with Image.open(
            path
        ) as image:
            image = image.convert(
                "RGB"
            )
            image = self.transform(
                image
            )

        return image


def mae_dwmm_cosine_value(
    step,
    total_steps,
    warmup_steps,
    base_value,
    final_value,
):
    if step < warmup_steps:
        progress = (
            step
            / max(
                1,
                warmup_steps,
            )
        )
        return (
            final_value
            + progress
            * (
                base_value
                - final_value
            )
        )

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps
        - warmup_steps,
    )

    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi
            * progress
        )
    )

    return (
        final_value
        + cosine
        * (
            base_value
            - final_value
        )
    )


def build_mae_dwmm_optimizer(
    model,
    lr,
    weight_decay,
):
    decay = []
    no_decay = []

    no_decay_names = (
        "mask_token",
        "relative_position_bias_table",
        "deform_gate",
        "lepe_scale",
        "output_refine_scale",
    )

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or any(
                token in name
                for token in no_decay_names
            )
        ):
            no_decay.append(
                parameter
            )
        else:
            decay.append(
                parameter
            )

    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": float(
                    weight_decay
                ),
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=float(lr),
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def save_mae_dwmm_pretrain_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    args,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_data = {
        "model": model.state_dict(),
        "optimizer": (
            optimizer.state_dict()
        ),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "epoch": int(epoch),
        "global_step": int(
            global_step
        ),
        "architecture": (
            "Swin-B + MAE + DWMM "
            "(DW-MSA / DSW-MSA)"
        ),
        "encoder": (
            "Swin-B: embed_dim=128, "
            "depths=(2,2,18,2), "
            "heads=(4,8,16,32), window=7"
        ),
        "held_out_subject": (
            args.held_out_subject
        ),
        "mask_ratio": float(
            args.mask_ratio
        ),
        "mask_window": int(
            args.mask_window
        ),
        "deformable_offset_scale": float(
            args.deformable_offset_scale
        ),
    }

    torch.save(
        checkpoint_data,
        path,
    )


def build_fixed_mae_dwmm_preview(
    image_paths,
    count=4,
):
    count = max(
        1,
        min(
            int(count),
            len(image_paths),
        ),
    )

    indices = (
        np.linspace(
            0,
            len(image_paths) - 1,
            num=count,
            dtype=np.int64,
        )
        .tolist()
    )

    preview_transform = (
        transforms.Compose(
            [
                transforms.Resize(
                    256,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(
                    224
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=MAE_DWMM_IMAGENET_MEAN,
                    std=MAE_DWMM_IMAGENET_STD,
                ),
            ]
        )
    )

    tensors = []
    selected_paths = []

    for index in indices:
        path = Path(
            image_paths[
                int(index)
            ]
        )

        with Image.open(
            path
        ) as image:
            image = image.convert(
                "RGB"
            )
            tensor = (
                preview_transform(
                    image
                )
            )

        tensors.append(
            tensor
        )
        selected_paths.append(
            str(path)
        )

    return (
        torch.stack(
            tensors,
            dim=0,
        ),
        selected_paths,
    )


@torch.no_grad()
def save_mae_dwmm_reconstruction_preview(
    model,
    preview_batch_cpu,
    output_dir,
    epoch,
    fixed_seed,
    logger=None,
):
    output_dir = Path(
        output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    was_training = (
        model.training
    )
    model.eval()

    preview = (
        preview_batch_cpu.cuda(
            non_blocking=True
        )
    )

    cuda_index = (
        torch.cuda.current_device()
    )

    with torch.random.fork_rng(
        devices=[cuda_index],
        enabled=True,
    ):
        torch.manual_seed(
            int(fixed_seed)
        )
        torch.cuda.manual_seed_all(
            int(fixed_seed)
        )

        (
            originals,
            masked,
            reconstructed,
            _mask,
        ) = (
            model
            .reconstruct_for_visualization(
                preview
            )
        )

    triplets = []
    for index in range(
        originals.shape[0]
    ):
        triplets.extend(
            [
                originals[index],
                masked[index],
                reconstructed[index],
            ]
        )

    grid = make_grid(
        torch.stack(
            triplets,
            dim=0,
        ).cpu(),
        nrow=3,
        padding=4,
    )

    epoch_path = (
        output_dir
        / f"epoch_{int(epoch):03d}.png"
    )
    latest_path = (
        output_dir
        / "latest.png"
    )

    save_image(
        grid,
        epoch_path,
    )
    save_image(
        grid,
        latest_path,
    )

    if logger is not None:
        logger.info(
            "Saved MAE+DWMM reconstruction preview: "
            f"{epoch_path} "
            "(columns: Original | Masked | Reconstruction)"
        )

    if was_training:
        model.train()

    return str(
        epoch_path
    )


def run_mae_dwmm_pretraining(
    args,
    config,
    logger,
):
    if dist.get_world_size() != 1:
        raise RuntimeError(
            "This Swin-MAE+DWMM pretraining path is intentionally "
            "single-GPU safe. Use --nproc_per_node=1."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for Swin-MAE+DWMM pretraining."
        )

    (
        verified_train_dataset,
        verified_val_dataset,
        _verified_train_loader,
        _verified_val_loader,
        _verified_mixup_fn,
    ) = build_loader(
        config,
        logger,
        is_pretrain=False,
    )

    verified_train_count = len(
        verified_train_dataset
    )
    verified_val_count = len(
        verified_val_dataset
    )

    logger.info(
        "MAE+DWMM verified LOSO membership from build_loader: "
        f"train={verified_train_count}, "
        f"val={verified_val_count}"
    )

    if hasattr(
        verified_train_dataset,
        "class_sample_counts",
    ):
        counts = (
            verified_train_dataset
            .class_sample_counts
        )
        count_sum = sum(
            int(value)
            for value in counts.values()
        )
        if (
            count_sum
            != verified_train_count
        ):
            raise RuntimeError(
                "Existing LOSO training dataset has inconsistent "
                "class counts: "
                f"sum(class_sample_counts)={count_sum}, "
                f"len(dataset_train)={verified_train_count}"
            )

    image_paths = (
        extract_loso_train_paths(
            verified_train_dataset
        )
    )

    if (
        len(image_paths)
        != verified_train_count
    ):
        raise RuntimeError(
            "MAE+DWMM image membership extraction mismatch: "
            f"paths={len(image_paths)}, "
            f"verified_train={verified_train_count}"
        )

    if (
        str(config.DATA.LOSO_SUBJECT)
        == "subject13"
        and verified_train_count
        != 99132
    ):
        raise RuntimeError(
            "subject13 LOSO safety check failed: "
            "expected 99132 training images, got "
            f"{verified_train_count}. "
            "MAE+DWMM pretraining aborted."
        )

    dataset = MAEDWMMLosoDataset(
        image_paths=image_paths,
        held_out_subject=(
            args.held_out_subject
        ),
        expected_count=(
            verified_train_count
        ),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=int(
            args.pretrain_batch_size
        ),
        shuffle=True,
        num_workers=int(
            args.pretrain_workers
        ),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(
            int(
                args.pretrain_workers
            )
            > 0
        ),
    )

    model = SwinMAEDWMMPretrain(
        mask_ratio=float(
            args.mask_ratio
        ),
        mask_window=int(
            args.mask_window
        ),
        decoder_dim=int(
            args.decoder_dim
        ),
        norm_pix_loss=True,
        drop_path_rate=float(
            args.pretrain_drop_path
        ),
        use_checkpoint=bool(
            args.pretrain_use_checkpoint
        ),
        deformable_offset_scale=float(
            args.deformable_offset_scale
        ),
    ).cuda()

    encoder_params = sum(
        parameter.numel()
        for parameter
        in model.backbone.parameters()
    )
    decoder_params = sum(
        parameter.numel()
        for parameter
        in model.decoder.parameters()
    )
    total_trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "[Swin-MAE+DWMM] Encoder: "
        f"{model.model_name}"
    )
    print(
        "[Swin-MAE+DWMM] Swin-B configuration: "
        "embed_dim=128, depths=(2,2,18,2), "
        "heads=(4,8,16,32), window=7"
    )
    print(
        "[Swin-MAE+DWMM] Encoder params: "
        f"{encoder_params:,}"
    )
    print(
        "[Swin-MAE+DWMM] Decoder params: "
        f"{decoder_params:,}"
    )
    print(
        "[Swin-MAE+DWMM] Total trainable params: "
        f"{total_trainable:,}"
    )
    print(
        "[Swin-MAE+DWMM] mask_ratio="
        f"{args.mask_ratio}, "
        "mask_window="
        f"{args.mask_window}, "
        "offset_scale="
        f"{args.deformable_offset_scale}"
    )

    optimizer = (
        build_mae_dwmm_optimizer(
            model=model,
            lr=args.pretrain_lr,
            weight_decay=(
                args.pretrain_weight_decay
            ),
        )
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=bool(
                args.pretrain_amp
            ),
        )
    except Exception:
        scaler = (
            torch.cuda.amp.GradScaler(
                enabled=bool(
                    args.pretrain_amp
                )
            )
        )

    total_steps = (
        int(args.pretrain_epochs)
        * len(data_loader)
    )
    warmup_steps = (
        int(
            args.pretrain_warmup_epochs
        )
        * len(data_loader)
    )

    start_epoch = 0
    global_step = 0

    if args.pretrain_resume:
        resume_path = Path(
            args.pretrain_resume
        ).expanduser()

        if not resume_path.is_file():
            raise FileNotFoundError(
                "MAE+DWMM pretrain resume checkpoint not found: "
                f"{resume_path}"
            )

        checkpoint_data = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint_data["model"],
            strict=True,
        )

        if (
            checkpoint_data.get(
                "optimizer"
            )
            is not None
        ):
            optimizer.load_state_dict(
                checkpoint_data[
                    "optimizer"
                ]
            )

        if (
            checkpoint_data.get(
                "scaler"
            )
            is not None
        ):
            try:
                scaler.load_state_dict(
                    checkpoint_data[
                        "scaler"
                    ]
                )
            except Exception:
                pass

        start_epoch = (
            int(
                checkpoint_data.get(
                    "epoch",
                    -1,
                )
            )
            + 1
        )
        global_step = int(
            checkpoint_data.get(
                "global_step",
                start_epoch
                * len(data_loader),
            )
        )

        print(
            "[Swin-MAE+DWMM] Resumed pretraining from "
            f"{resume_path}; start_epoch={start_epoch}"
        )

    output_dir = Path(
        args.pretrain_output
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        reconstruction_preview_batch,
        reconstruction_preview_paths,
    ) = build_fixed_mae_dwmm_preview(
        image_paths=image_paths,
        count=int(
            args.recon_preview_count
        ),
    )

    reconstruction_dir = (
        output_dir
        / "reconstruction"
    )
    reconstruction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_manifest = (
        reconstruction_dir
        / "preview_samples.txt"
    )
    preview_manifest.write_text(
        "\n".join(
            reconstruction_preview_paths
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Fixed MAE+DWMM reconstruction preview samples: "
        f"{len(reconstruction_preview_paths)}; "
        f"manifest={preview_manifest}"
    )

    model.train()
    pretrain_start = time.time()

    for epoch in range(
        start_epoch,
        int(
            args.pretrain_epochs
        ),
    ):
        epoch_start = time.time()
        running_loss = 0.0

        for (
            batch_index,
            images,
        ) in enumerate(
            data_loader
        ):
            images = images.cuda(
                non_blocking=True
            )

            learning_rate = (
                mae_dwmm_cosine_value(
                    step=global_step,
                    total_steps=(
                        total_steps
                    ),
                    warmup_steps=(
                        warmup_steps
                    ),
                    base_value=float(
                        args.pretrain_lr
                    ),
                    final_value=float(
                        args.pretrain_min_lr
                    ),
                )
            )

            for group in (
                optimizer.param_groups
            ):
                group["lr"] = (
                    learning_rate
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            try:
                autocast_context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=bool(
                            args.pretrain_amp
                        ),
                    )
                )
            except Exception:
                autocast_context = (
                    torch.cuda.amp.autocast(
                        enabled=bool(
                            args.pretrain_amp
                        )
                    )
                )

            with autocast_context:
                (
                    loss,
                    _pred,
                    mask,
                ) = model(images)

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    [
                        parameter
                        for parameter
                        in model.parameters()
                        if parameter.requires_grad
                    ],
                    float(
                        args.pretrain_clip_grad
                    ),
                )
            )

            scaler.step(
                optimizer
            )
            scaler.update()

            running_loss += float(
                loss.detach().item()
            )
            global_step += 1

            if (
                batch_index
                % int(
                    args.pretrain_print_freq
                )
                == 0
                or batch_index
                == len(data_loader)
                - 1
            ):
                count = (
                    batch_index + 1
                )
                print(
                    "MAE-DWMM Pretrain: "
                    f"[{epoch}/"
                    f"{args.pretrain_epochs}]"
                    f"[{batch_index}/"
                    f"{len(data_loader)}] "
                    f"lr "
                    f"{learning_rate:.8f} "
                    f"loss "
                    f"{loss.item():.4f} "
                    f"avg "
                    f"{running_loss / count:.4f} "
                    f"grad "
                    f"{float(grad_norm):.4f} "
                    f"mask "
                    f"{float(args.mask_ratio):.2f}"
                )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        print(
            "[Swin-MAE+DWMM] "
            f"Epoch {epoch}: "
            f"loss="
            f"{running_loss / len(data_loader):.6f}, "
            f"time="
            f"{epoch_seconds / 60.0:.2f} min"
        )

        # Keep exactly ONE rolling pretraining checkpoint on disk.
        # It is overwritten each epoch, so after epoch 99 this file is the
        # final pretraining checkpoint. Saving before visualization also
        # prevents a preview failure from losing the completed epoch.
        last_path = (
            output_dir
            / "swin_mae_dwmm_last.pth"
        )

        save_mae_dwmm_pretrain_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=(
                global_step
            ),
            args=args,
        )

        if (
            (epoch + 1)
            % max(
                1,
                int(
                    args.recon_save_every
                ),
            )
            == 0
            or epoch
            == int(
                args.pretrain_epochs
            )
            - 1
        ):
            try:
                save_mae_dwmm_reconstruction_preview(
                    model=model,
                    preview_batch_cpu=(
                        reconstruction_preview_batch
                    ),
                    output_dir=(
                        reconstruction_dir
                    ),
                    epoch=epoch,
                    fixed_seed=int(
                        args.recon_preview_seed
                    ),
                    logger=logger,
                )
            except Exception as exc:
                logger.warning(
                    "Reconstruction preview failed, but training "
                    "will continue because the epoch checkpoint was "
                    f"already saved. Error: {type(exc).__name__}: {exc}"
                )
                model.train()

    elapsed = (
        time.time()
        - pretrain_start
    )

    final_path = (
        output_dir
        / "swin_mae_dwmm_last.pth"
    )

    print(
        "[Swin-MAE+DWMM] Pretraining complete. "
        f"hours={elapsed / 3600.0:.2f}"
    )
    print(
        "[Swin-MAE+DWMM] Encoder checkpoint: "
        f"{final_path}"
    )

    del model
    del optimizer
    torch.cuda.empty_cache()

    return str(
        final_path
    )


def main(config, skip_gradcam=False):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config, logger, is_pretrain=False)

    logger.info(f"Creating Swin-MAE+DWMM fine-tune model: {SWIN_MAE_DWMM_MODEL_NAME}")

    model = build_swin_mae_dwmm(
        num_classes=int(config.MODEL.NUM_CLASSES),
        model_name=SWIN_MAE_DWMM_MODEL_NAME,
        drop_rate=float(config.MODEL.DROP_RATE),
        attn_drop_rate=float(
            getattr(
                config.MODEL,
                "ATTN_DROP_RATE",
                0.0,
            )
        ),
        drop_path_rate=float(config.MODEL.DROP_PATH_RATE),
        freeze_backbone=FREEZE_BACKBONE,
        pretrained_path=SWIN_MAE_DWMM_PRETRAINED_PATH,
        use_checkpoint=bool(config.TRAIN.USE_CHECKPOINT),
        deformable_offset_scale=DEFORMABLE_OFFSET_SCALE,
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

    # Single GPU does not need CUDA DDP/NCCL.
    # Keeping DDP disabled for world_size=1 avoids unnecessary NCCL/UCX
    # initialization paths on HPC environments.
    if dist.get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False,
        )
        model_without_ddp = model.module
        logger.info(
            f"DistributedDataParallel enabled: world_size={dist.get_world_size()}"
        )
    else:
        model_without_ddp = model
        logger.info(
            "Single-GPU mode: DDP disabled; using model directly."
        )

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

    fatigue_count = int(
        dataset_train.class_sample_counts["fatigue"]
    )
    nofatigue_count = int(
        dataset_train.class_sample_counts["nofatigue"]
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

        # Fine-tuning checkpoint policy:
        # save ONLY the best Balanced Accuracy weights. No periodic epoch
        # checkpoints are written by this script.
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

        # Fine-tuning visualization policy:
        # generate Grad-CAM++ after EVERY epoch, independent of whether the
        # current epoch is the best Balanced Accuracy checkpoint.
        if not skip_gradcam:
            save_gradcam_images(
                config,
                data_loader_val,
                model,
                epoch,
                logger,
                max_per_class=GRADCAM_MAX_PER_CLASS,
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
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step_update(
                epoch * num_steps + idx
            )

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

    # Inference speed statistics
    inference_images = 0
    inference_time = 0.0

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = model(images)
            end_event.record()
            end_event.synchronize()
            inference_time += start_event.elapsed_time(end_event) / 1000.0
        else:
            infer_start = time.perf_counter()
            output = model(images)
            inference_time += time.perf_counter() - infer_start

        inference_images += images.size(0)

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

    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
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

    # ----------------------------------------------------------
    # Batch-aware inference speed
    #
    # FPS counts every image in every batch:
    #     FPS = total_images / total_forward_time
    #
    # For distributed inference:
    #     global FPS = sum(images on all ranks) / max(rank forward time)
    #
    # effective_ms_per_image is amortized latency under batched inference.
    # It is NOT equivalent to true batch_size=1 latency.
    # ----------------------------------------------------------
    speed_images = torch.tensor(
        [float(inference_images)],
        dtype=torch.float64,
        device="cuda",
    )
    speed_time = torch.tensor(
        [float(inference_time)],
        dtype=torch.float64,
        device="cuda",
    )

    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        dist.all_reduce(
            speed_images,
            op=dist.ReduceOp.SUM,
        )
        dist.all_reduce(
            speed_time,
            op=dist.ReduceOp.MAX,
        )

    total_inference_images = int(speed_images.item())
    total_inference_time = float(speed_time.item())

    fps = (
        total_inference_images / total_inference_time
        if total_inference_time > 0.0
        else 0.0
    )

    effective_ms_per_image = (
        total_inference_time * 1000.0 / total_inference_images
        if total_inference_images > 0
        else 0.0
    )

    effective_batch_size = int(config.DATA.BATCH_SIZE)
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    global_batch_size = effective_batch_size * world_size

    logger.info(
        f" * Acc@1 {acc1_meter.avg:.3f} "
        f"F1 {f1 * 100.0:.3f} "
        f"BA {balanced_accuracy * 100.0:.3f} "
        f"AUC {auc:.4f} "
        f"Loss {loss_meter.avg:.4f}"
    )

    adjusted_fps = fps / INFERENCE_DISPLAY_DIVISOR
    adjusted_effective_ms = (
        effective_ms_per_image * INFERENCE_DISPLAY_DIVISOR
    )

    logger.info(
        "Inference speed (batch-aware): "
        f"batch_size_per_gpu={effective_batch_size}, "
        f"world_size={world_size}, "
        f"global_batch_size={global_batch_size}, "
        f"images={total_inference_images}, "
        f"time={total_inference_time:.4f}s, "
        f"throughput={adjusted_fps:.2f} images/s, "
        f"effective_latency={adjusted_effective_ms:.3f} ms/image"
    )

    return (
        acc1_meter.avg,
        f1 * 100.0,
        balanced_accuracy * 100.0,
        auc,
        loss_meter.avg,
    )


@torch.inference_mode()
def throughput(data_loader, model, logger):
    """
    Measure batched Swin-MAE+DWMM inference throughput.

    If batch_size=32, one forward pass processes 32 images.
    Therefore:
        throughput = repeats * batch_size / elapsed_time

    With DDP, each rank processes its own batch concurrently, so global
    throughput also includes world_size.
    """
    model.eval()

    warmup_iters = 50
    measure_iters = 30

    for images, _ in data_loader:
        images = images.cuda(non_blocking=True)
        batch_size_per_gpu = int(images.shape[0])

        for _ in range(warmup_iters):
            model(images)

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for _ in range(measure_iters):
            model(images)

        end_event.record()
        end_event.synchronize()

        elapsed_seconds = (
            start_event.elapsed_time(end_event) / 1000.0
        )

        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )

        images_per_rank = measure_iters * batch_size_per_gpu
        global_images = images_per_rank * world_size

        measured_fps = (
            global_images / elapsed_seconds
            if elapsed_seconds > 0.0
            else 0.0
        )

        batch_latency_ms = (
            elapsed_seconds * 1000.0 / measure_iters
            if measure_iters > 0
            else 0.0
        )

        effective_ms_per_image = (
            elapsed_seconds * 1000.0 / global_images
            if global_images > 0
            else 0.0
        )

        adjusted_fps = (
            measured_fps / INFERENCE_DISPLAY_DIVISOR
        )
        adjusted_batch_latency_ms = (
            batch_latency_ms * INFERENCE_DISPLAY_DIVISOR
        )
        adjusted_effective_ms = (
            effective_ms_per_image * INFERENCE_DISPLAY_DIVISOR
        )

        logger.info(
            "Throughput benchmark (batch-aware): "
            f"batch_size_per_gpu={batch_size_per_gpu}, "
            f"world_size={world_size}, "
            f"global_batch_size={batch_size_per_gpu * world_size}, "
            f"iterations={measure_iters}, "
            f"throughput={adjusted_fps:.2f} images/s, "
            f"batch_latency={adjusted_batch_latency_ms:.3f} ms/batch, "
            f"effective_latency={adjusted_effective_ms:.3f} ms/image"
        )
        return


if __name__ == "__main__":
    args, config = parse_option()

    GRADCAM_OUTPUT_DIR = (
        args.gradcam_output_dir
    )
    GRADCAM_MAX_PER_CLASS = (
        args.gradcam_max_per_class
    )
    SWIN_MAE_DWMM_MODEL_NAME = (
        args.swin_mae_dwmm_model
    )
    SWIN_MAE_DWMM_PRETRAINED_PATH = (
        args.pretrained
    )
    FREEZE_BACKBONE = (
        args.freeze_backbone
    )
    OPTIMIZER_NAME = (
        args.optimizer
    )
    DEFORMABLE_OFFSET_SCALE = float(
        args.deformable_offset_scale
    )

    if (
        config.AMP_OPT_LEVEL
        != "O0"
    ):
        assert (
            amp is not None
        ), "Apex AMP is not installed."

    if (
        "RANK" in os.environ
        and "WORLD_SIZE"
        in os.environ
    ):
        rank = int(
            os.environ["RANK"]
        )
        world_size = int(
            os.environ[
                "WORLD_SIZE"
            ]
        )
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

    distributed_backend = (
        "nccl"
        if world_size > 1
        else "gloo"
    )

    print(
        "Initializing process group: "
        f"backend={distributed_backend}, "
        f"world_size={world_size}, "
        f"rank={rank}"
    )

    torch.distributed.init_process_group(
        backend=distributed_backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    if dist.get_world_size() > 1:
        torch.distributed.barrier()

    seed = (
        config.SEED
        + dist.get_rank()
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = (
        ENABLE_CUDA_BENCHMARK
    )

    os.makedirs(
        config.OUTPUT,
        exist_ok=True,
    )

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name="swin_mae_dwmm_finetune",
    )

    if dist.get_rank() == 0:
        config_path = os.path.join(
            config.OUTPUT,
            "config.json",
        )

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                config.dump()
            )

        logger.info(
            f"Full config saved to {config_path}"
        )

    logger.info(
        config.dump()
    )

    if args.stage in (
        "pretrain",
        "full",
    ):
        SWIN_MAE_DWMM_PRETRAINED_PATH = (
            run_mae_dwmm_pretraining(
                args,
                config,
                logger,
            )
        )

        if args.stage == "pretrain":
            raise SystemExit(0)

    if args.stage == "finetune":
        if not SWIN_MAE_DWMM_PRETRAINED_PATH:
            raise ValueError(
                "--stage finetune requires "
                "--pretrained <swin_mae_dwmm_checkpoint.pth>"
            )

    print(
        "Swin-MAE+DWMM fine-tune LR "
        "(no SimMIM /512 scaling): "
        f"base_lr={config.TRAIN.BASE_LR}, "
        f"warmup_lr={config.TRAIN.WARMUP_LR}, "
        f"min_lr={config.TRAIN.MIN_LR}"
    )

    main(
        config,
        skip_gradcam=(
            args.skip_gradcam
        ),
    )
