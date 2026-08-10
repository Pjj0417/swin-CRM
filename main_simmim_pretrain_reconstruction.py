# --------------------------------------------------------

# 1. Supports --pretrained: load model weights only for domain-adaptive pre-training.

# --------------------------------------------------------

import argparse
import datetime
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.utils import AverageMeter
from torchvision.utils import make_grid, save_image

from config import get_config
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import (
    auto_resume_helper,
    get_grad_norm,
    load_checkpoint,
    save_checkpoint,
)

try:
    # Original SimMIM uses NVIDIA Apex AMP.
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser(
        "SimMIM pre-training script",
        add_help=True,
    )

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--opts",
        help="Override config options using KEY VALUE pairs.",
        default=None,
        nargs="+",
    )

    # Common overrides
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for one GPU.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the training dataset.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume an interrupted run. This normally restores model, "
            "optimizer, scheduler and epoch."
        ),
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help=(
            "Load SimMIM model weights only, then start a new training run "
            "with a fresh optimizer and scheduler."
        ),
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="Apex AMP level. Use O0 to disable Apex AMP.",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="Root output folder.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Experiment tag.",
    )

    parser.add_argument(
        "--reconstruction-freq",
        type=int,
        default=1,
        help="Save a reconstruction panel every N epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--reconstruction-images",
        type=int,
        default=4,
        help="Number of samples included in each reconstruction panel.",
    )

    # torchrun normally provides LOCAL_RANK as an environment variable.
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank. Normally supplied by torchrun.",
    )

    args = parser.parse_args()
    config = get_config(args)
    return args, config


def unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Extract a model state_dict from common checkpoint formats.

    Supported examples:
        {"model": state_dict}
        {"state_dict": state_dict}
        raw state_dict
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dict-like object, got {type(checkpoint)!r}."
        )

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key

        # Remove common wrappers.
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_pretrained_weights(model, checkpoint_path, logger):
    """
    Load model weights only.

    This is intended for:
        ImageNet SimMIM checkpoint
            -> target-domain SimMIM continued pre-training

    Optimizer, scheduler and old epoch are deliberately not restored.
    """
    if not checkpoint_path:
        return

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint does not exist: {checkpoint_path}"
        )

    logger.info(f"Loading pretrained model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=False)

    logger.info(
        "Pretrained weights loaded. "
        f"Missing keys: {len(incompatible.missing_keys)}, "
        f"unexpected keys: {len(incompatible.unexpected_keys)}"
    )

    if incompatible.missing_keys:
        logger.warning(
            "Missing keys:\n  " + "\n  ".join(incompatible.missing_keys)
        )

    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected keys:\n  " + "\n  ".join(incompatible.unexpected_keys)
        )

    del checkpoint
    torch.cuda.empty_cache()


def main(config, args, logger):
    data_loader_train = build_loader(
        config,
        logger,
        is_pretrain=True,
    )

    logger.info(
        f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}"
    )
    model = build_model(config, is_pretrain=True)

    # --pretrained loads model weights only.
    # Do this before moving the model to CUDA and before building the optimizer.
    if args.pretrained:
        load_pretrained_weights(
            model=model,
            checkpoint_path=args.pretrained,
            logger=logger,
        )

    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(
        config,
        model,
        logger,
        is_pretrain=True,
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
        output_device=config.LOCAL_RANK,
        broadcast_buffers=False,
    )
    model_without_ddp = model.module

    n_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logger.info(f"Number of trainable parameters: {n_parameters}")

    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9:.3f}")

    lr_scheduler = build_scheduler(
        config,
        optimizer,
        len(data_loader_train),
    )

    # Auto-resume is for interrupted runs. Explicit --resume takes priority.
    if config.TRAIN.AUTO_RESUME and not config.MODEL.RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)

        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from: {resume_file}")
        else:
            logger.info(
                f"No checkpoint found in {config.OUTPUT}; "
                "starting without auto-resume."
            )

    # --resume/config.MODEL.RESUME restores full training state.
    if config.MODEL.RESUME:
        load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            logger,
        )

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(
        config.TRAIN.START_EPOCH,
        config.TRAIN.EPOCHS,
    ):
        if hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config=config,
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            logger=logger,
            args=args,
        )

        should_save = (
            epoch % config.SAVE_FREQ == 0
            or epoch == config.TRAIN.EPOCHS - 1
        )

        if dist.get_rank() == 0 and should_save:
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                0.0,
                optimizer,
                lr_scheduler,
                logger,
            )

    total_time = time.time() - start_time
    total_time_str = str(
        datetime.timedelta(seconds=int(total_time))
    )
    logger.info(f"Training time: {total_time_str}")



def _as_pair(value):
    """Convert an integer or a two-element sequence to (height, width)."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a two-element patch size, got {value}.")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _denormalize(images):
    """
    Convert ImageNet-normalized tensors to the display range [0, 1].

    SimMIM's standard ImageNet transform uses these statistics.
    """
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def save_reconstruction_panel(
    config,
    images,
    masks,
    reconstructions,
    epoch,
    max_images,
    logger,
):
    """
    Save a paper-style panel:
        original | masked input | reconstruction | blended reconstruction

    The reported MSE is measured only on masked pixels in normalized space.
    """
    count = min(max(1, int(max_images)), images.size(0))
    images = images[:count].detach()
    masks = masks[:count].detach()
    reconstructions = reconstructions[:count].detach()

    patch_size = config.MODEL.SWIN.PATCH_SIZE
    if config.MODEL.TYPE == "vit":
        patch_size = config.MODEL.VIT.PATCH_SIZE
    patch_h, patch_w = _as_pair(patch_size)

    pixel_mask = masks.repeat_interleave(
        patch_h, dim=1
    ).repeat_interleave(
        patch_w, dim=2
    ).unsqueeze(1).to(
        device=images.device,
        dtype=images.dtype,
    )

    if pixel_mask.shape[-2:] != images.shape[-2:]:
        raise ValueError(
            f"Expanded mask shape {pixel_mask.shape[-2:]} does not match "
            f"image shape {images.shape[-2:]}."
        )

    masked_mse = (
        ((reconstructions - images) ** 2 * pixel_mask).sum()
        / (pixel_mask.sum() * images.size(1)).clamp_min(1.0)
    )

    original_display = _denormalize(images)
    reconstruction_display = _denormalize(reconstructions)

    # Use neutral gray for hidden regions.
    masked_display = (
        original_display * (1.0 - pixel_mask)
        + 0.5 * pixel_mask
    )

    # Preserve visible pixels and insert model predictions only in masked areas.
    blended_display = (
        original_display * (1.0 - pixel_mask)
        + reconstruction_display * pixel_mask
    )

    rows = []
    for sample_index in range(count):
        rows.extend([
            original_display[sample_index],
            masked_display[sample_index],
            reconstruction_display[sample_index],
            blended_display[sample_index],
        ])

    panel = make_grid(
        rows,
        nrow=4,
        padding=4,
    )

    output_dir = os.path.join(config.OUTPUT, "reconstruction")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"epoch_{epoch:04d}_mse_{masked_mse.item():.6f}.png",
    )
    save_image(panel.cpu(), output_path)

    logger.info(
        f"Saved reconstruction panel to {output_path}; "
        f"masked MSE={masked_mse.item():.6f}. "
        "Columns: original | masked | reconstruction | blended."
    )

def train_one_epoch(
    config,
    model,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    logger,
    args,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps = len(data_loader)
    accumulation_steps = max(
        1,
        int(config.TRAIN.ACCUMULATION_STEPS),
    )

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (img, mask, _) in enumerate(data_loader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        should_visualize = (
            dist.get_rank() == 0
            and args.reconstruction_freq > 0
            and epoch % args.reconstruction_freq == 0
            and idx == 0
        )

        if should_visualize:
            loss, x_rec = model(img, mask, return_rec=True)
            save_reconstruction_panel(
                config=config,
                images=img,
                masks=mask,
                reconstructions=x_rec,
                epoch=epoch,
                max_images=args.reconstruction_images,
                logger=logger,
            )
        else:
            loss = model(img, mask)

        original_loss_value = loss.item()

        # Scale loss only for backward accumulation.
        backward_loss = loss / accumulation_steps

        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(
                backward_loss,
                optimizer,
            ) as scaled_loss:
                scaled_loss.backward()

            parameters_for_norm = amp.master_params(optimizer)
        else:
            backward_loss.backward()
            parameters_for_norm = model.parameters()

        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters_for_norm,
                config.TRAIN.CLIP_GRAD,
            )
        else:
            grad_norm = get_grad_norm(parameters_for_norm)

        is_accumulation_boundary = (
            (idx + 1) % accumulation_steps == 0
        )
        is_last_step = idx == num_steps - 1

        if is_accumulation_boundary or is_last_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Keep scheduler update consistent with the original iteration index.
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(
            original_loss_value,
            img.size(0),
        )

        grad_norm_value = (
            grad_norm.item()
            if torch.is_tensor(grad_norm)
            else float(grad_norm)
        )
        norm_meter.update(grad_norm_value)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = (
                torch.cuda.max_memory_allocated()
                / (1024.0 * 1024.0)
            )
            eta_seconds = batch_time.avg * (num_steps - idx - 1)

            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}]"
                f"[{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(eta_seconds))} "
                f"lr {lr:.8f}\t"
                f"time {batch_time.val:.4f} "
                f"({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.6f} "
                f"({loss_meter.avg:.6f})\t"
                f"grad_norm {norm_meter.val:.4f} "
                f"({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"Epoch {epoch} training takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}"
    )


def init_distributed_mode(config):
    """
    Initialize one-process-per-GPU distributed training for torchrun.
    """
    local_rank = int(os.environ.get(
        "LOCAL_RANK",
        config.LOCAL_RANK,
    ))

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    dist.barrier()

    # Keep config synchronized with torchrun's environment.
    config.defrost()
    config.LOCAL_RANK = local_rank
    config.freeze()


def scale_learning_rates(config):
    """
    Apply the SimMIM linear learning-rate scaling rule.

    Effective batch size:
        batch_size_per_gpu * world_size * accumulation_steps
    """
    world_size = dist.get_world_size()
    accumulation_steps = max(
        1,
        int(config.TRAIN.ACCUMULATION_STEPS),
    )
    effective_batch_size = (
        config.DATA.BATCH_SIZE
        * world_size
        * accumulation_steps
    )

    scale = effective_batch_size / 512.0

    config.defrost()
    config.TRAIN.BASE_LR *= scale
    config.TRAIN.WARMUP_LR *= scale
    config.TRAIN.MIN_LR *= scale
    config.freeze()


if __name__ == "__main__":
    args, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, (
            "Apex AMP is not installed. "
            "Use --amp-opt-level O0 or install a compatible Apex build."
        )

    init_distributed_mode(config)

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    scale_learning_rates(config)

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=config.MODEL.NAME,
    )

    if dist.get_rank() == 0:
        config_path = os.path.join(
            config.OUTPUT,
            "config.yaml",
        )
        with open(config_path, "w", encoding="utf-8") as file:
            file.write(config.dump())
        logger.info(f"Full config saved to: {config_path}")

    logger.info(config.dump())

    main(
        config=config,
        args=args,
        logger=logger,
    )
