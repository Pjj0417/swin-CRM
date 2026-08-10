# # --------------------------------------------------------
# # SimMIM
# # Copyright (c) 2021 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ze Liu
# # Modified by Zhenda Xie
# # --------------------------------------------------------

# import os
# import time
# import argparse
# import datetime
# import numpy as np

# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist

# from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from timm.utils import accuracy, AverageMeter

# from config import get_config
# from models import build_model
# from data import build_loader
# from lr_scheduler import build_scheduler
# from optimizer import build_optimizer
# from logger import create_logger
# from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

# try:
#     # noinspection PyUnresolvedReferences
#     from apex import amp
# except ImportError:
#     amp = None


# def parse_option():
#     parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
#     parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )

#     # easy config modification
#     parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
#     parser.add_argument('--data-path', type=str, help='path to dataset')
#     parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
#     parser.add_argument('--resume', help='resume from checkpoint')
#     parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
#     parser.add_argument('--use-checkpoint', action='store_true',
#                         help="whether to use gradient checkpointing to save memory")
#     parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                         help='mixed precision opt level, if O0, no amp is used')
#     parser.add_argument('--output', default='output', type=str, metavar='PATH',
#                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
#     parser.add_argument('--tag', help='tag of experiment')
#     parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
#     parser.add_argument('--throughput', action='store_true', help='Test throughput only')

#     # distributed training
#     parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

#     args = parser.parse_args()

#     config = get_config(args)

#     return args, config


# def main(config):
#     dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger, is_pretrain=False)

#     logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
#     model = build_model(config, is_pretrain=False)
#     model.cuda()
#     logger.info(str(model))

#     optimizer = build_optimizer(config, model, logger, is_pretrain=False)
#     if config.AMP_OPT_LEVEL != "O0":
#         model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
#     model_without_ddp = model.module

#     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"number of params: {n_parameters}")
#     if hasattr(model_without_ddp, 'flops'):
#         flops = model_without_ddp.flops()
#         logger.info(f"number of GFLOPs: {flops / 1e9}")

#     lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

#     if config.AUG.MIXUP > 0.:
#         # smoothing is handled with mixup label transform
#         criterion = SoftTargetCrossEntropy()
#     elif config.MODEL.LABEL_SMOOTHING > 0.:
#         criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
#     else:
#         criterion = torch.nn.CrossEntropyLoss()

#     max_accuracy = 0.0

#     if config.TRAIN.AUTO_RESUME:
#         resume_file = auto_resume_helper(config.OUTPUT, logger)
#         if resume_file:
#             if config.MODEL.RESUME:
#                 logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
#             config.defrost()
#             config.MODEL.RESUME = resume_file
#             config.freeze()
#             logger.info(f'auto resuming from {resume_file}')
#         else:
#             logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

#     if config.MODEL.RESUME:
#         max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
#         acc1, acc5, loss = validate(config, data_loader_val, model)
#         logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
#         if config.EVAL_MODE:
#             return
#     elif config.PRETRAINED:
#         load_pretrained(config, model_without_ddp, logger)

#     if config.THROUGHPUT_MODE:
#         throughput(data_loader_val, model, logger)
#         return

#     logger.info("Start training")
#     start_time = time.time()
#     for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
#         data_loader_train.sampler.set_epoch(epoch)

#         train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
#         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
#             save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

#         acc1, acc5, loss = validate(config, data_loader_val, model)
#         logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
#         max_accuracy = max(max_accuracy, acc1)
#         logger.info(f'Max accuracy: {max_accuracy:.2f}%')

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     logger.info('Training time {}'.format(total_time_str))


# def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
#     model.train()
#     optimizer.zero_grad()
    
#     logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

#     num_steps = len(data_loader)
#     batch_time = AverageMeter()
#     loss_meter = AverageMeter()
#     norm_meter = AverageMeter()

#     start = time.time()
#     end = time.time()
#     for idx, (samples, targets) in enumerate(data_loader):
#         samples = samples.cuda(non_blocking=True)
#         targets = targets.cuda(non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)

#         outputs = model(samples)

#         if config.TRAIN.ACCUMULATION_STEPS > 1:
#             loss = criterion(outputs, targets)
#             loss = loss / config.TRAIN.ACCUMULATION_STEPS
#             if config.AMP_OPT_LEVEL != "O0":
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#                 if config.TRAIN.CLIP_GRAD:
#                     grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
#                 else:
#                     grad_norm = get_grad_norm(amp.master_params(optimizer))
#             else:
#                 loss.backward()
#                 if config.TRAIN.CLIP_GRAD:
#                     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
#                 else:
#                     grad_norm = get_grad_norm(model.parameters())
#             if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 lr_scheduler.step_update(epoch * num_steps + idx)
#         else:
#             loss = criterion(outputs, targets)
#             optimizer.zero_grad()
#             if config.AMP_OPT_LEVEL != "O0":
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#                 if config.TRAIN.CLIP_GRAD:
#                     grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
#                 else:
#                     grad_norm = get_grad_norm(amp.master_params(optimizer))
#             else:
#                 loss.backward()
#                 if config.TRAIN.CLIP_GRAD:
#                     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
#                 else:
#                     grad_norm = get_grad_norm(model.parameters())
#             optimizer.step()
#             lr_scheduler.step_update(epoch * num_steps + idx)

#         torch.cuda.synchronize()

#         loss_meter.update(loss.item(), targets.size(0))
#         norm_meter.update(grad_norm)
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             lr = optimizer.param_groups[-1]['lr']
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             etas = batch_time.avg * (num_steps - idx)
#             logger.info(
#                 f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
#                 f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
#                 f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
#                 f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
#                 f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
#                 f'mem {memory_used:.0f}MB')
#     epoch_time = time.time() - start
#     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


# @torch.no_grad()
# def validate(config, data_loader, model):
#     criterion = torch.nn.CrossEntropyLoss()
#     model.eval()

#     batch_time = AverageMeter()
#     loss_meter = AverageMeter()
#     acc1_meter = AverageMeter()
#     acc5_meter = AverageMeter()

#     end = time.time()
#     for idx, (images, target) in enumerate(data_loader):
#         images = images.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)

#         # compute output
#         output = model(images)

#         # measure accuracy and record loss
#         loss = criterion(output, target)
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))

#         acc1 = reduce_tensor(acc1)
#         acc5 = reduce_tensor(acc5)
#         loss = reduce_tensor(loss)

#         loss_meter.update(loss.item(), target.size(0))
#         acc1_meter.update(acc1.item(), target.size(0))
#         acc5_meter.update(acc5.item(), target.size(0))

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if idx % config.PRINT_FREQ == 0:
#             memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
#             logger.info(
#                 f'Test: [{idx}/{len(data_loader)}]\t'
#                 f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                 f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
#                 f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
#                 f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
#                 f'Mem {memory_used:.0f}MB')
#     logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
#     return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


# @torch.no_grad()
# def throughput(data_loader, model, logger):
#     model.eval()

#     for idx, (images, _) in enumerate(data_loader):
#         images = images.cuda(non_blocking=True)
#         batch_size = images.shape[0]
#         for i in range(50):
#             model(images)
#         torch.cuda.synchronize()
#         logger.info(f"throughput averaged with 30 times")
#         tic1 = time.time()
#         for i in range(30):
#             model(images)
#         torch.cuda.synchronize()
#         tic2 = time.time()
#         logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
#         return


# if __name__ == '__main__':
#     _, config = parse_option()

#     if config.AMP_OPT_LEVEL != "O0":
#         assert amp is not None, "amp not installed!"

#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1
#     torch.cuda.set_device(config.LOCAL_RANK)
#     torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
#     torch.distributed.barrier()

#     seed = config.SEED + dist.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     cudnn.benchmark = True

#     # linear scale the learning rate according to total batch size, may not be optimal
#     linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
#     linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
#     linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
#     # gradient accumulation also need to scale the learning rate
#     if config.TRAIN.ACCUMULATION_STEPS > 1:
#         linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
#         linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
#         linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
#     config.defrost()
#     config.TRAIN.BASE_LR = linear_scaled_lr
#     config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
#     config.TRAIN.MIN_LR = linear_scaled_min_lr
#     config.freeze()

#     os.makedirs(config.OUTPUT, exist_ok=True)
#     logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

#     if dist.get_rank() == 0:
#         path = os.path.join(config.OUTPUT, "config.json")
#         with open(path, "w") as f:
#             f.write(config.dump())
#         logger.info(f"Full config saved to {path}")

#     # print config
#     logger.info(config.dump())

#     main(config)

# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="local rank for DistributedDataParallel"
    )

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger, is_pretrain=False)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=False)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=False)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return
    elif config.PRETRAINED:
        load_pretrained(config, model_without_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info(
        f"Final training configuration: "
        f"EPOCHS={config.TRAIN.EPOCHS}, "
        f"START_EPOCH={config.TRAIN.START_EPOCH}, "
        f"WARMUP_EPOCHS={config.TRAIN.WARMUP_EPOCHS}, "
        f"BATCH_SIZE={config.DATA.BATCH_SIZE}"
    )
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    
    logger.info(f'Current learning rate for different parameter groups: {[it["lr"] for it in optimizer.param_groups]}')

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

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[-1]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    # 二分类混淆矩阵
    # 行：真实标签
    # 列：预测标签
    # [[真实0预测0, 真实0预测1],
    #  [真实1预测0, 真实1预测1]]
    confusion_matrix = torch.zeros(
        2,
        2,
        dtype=torch.long,
        device="cuda"
    )

    # 分别统计真实标签和预测标签数量
    target_count = torch.zeros(
        2,
        dtype=torch.long,
        device="cuda"
    )

    pred_count = torch.zeros(
        2,
        dtype=torch.long,
        device="cuda"
    )

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 前向传播
        output = model(images)

        # 计算损失
        loss = criterion(output, target)

        # 得到预测类别
        pred = torch.argmax(output, dim=1)

        # 当前批次准确率
        acc1 = (pred == target).float().mean() * 100.0

        # 更新平均指标
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # 统计真实标签数量
        target_count += torch.bincount(
            target,
            minlength=2
        )

        # 统计预测标签数量
        pred_count += torch.bincount(
            pred,
            minlength=2
        )

        # 更新混淆矩阵
        for true_label, pred_label in zip(target, pred):
            confusion_matrix[
                true_label.long(),
                pred_label.long()
            ] += 1

        # 计算耗时
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
                f"({acc1_meter.avg:.3f})\t"
                f"Mem {torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f}MB"
            )

    # 多卡训练时，汇总所有GPU的统计结果
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            confusion_matrix,
            op=torch.distributed.ReduceOp.SUM
        )

        torch.distributed.all_reduce(
            target_count,
            op=torch.distributed.ReduceOp.SUM
        )

        torch.distributed.all_reduce(
            pred_count,
            op=torch.distributed.ReduceOp.SUM
        )

    # 转到CPU方便输出
    cm = confusion_matrix.cpu()
    true_num = target_count.cpu()
    prediction_num = pred_count.cpu()

    # 混淆矩阵中的四个值
    true_0_pred_0 = cm[0, 0].item()
    true_0_pred_1 = cm[0, 1].item()
    true_1_pred_0 = cm[1, 0].item()
    true_1_pred_1 = cm[1, 1].item()

    # 每个类别准确率
    class_0_total = true_num[0].item()
    class_1_total = true_num[1].item()

    class_0_acc = (
        100.0 * true_0_pred_0 / class_0_total
        if class_0_total > 0
        else 0.0
    )

    class_1_acc = (
        100.0 * true_1_pred_1 / class_1_total
        if class_1_total > 0
        else 0.0
    )

    logger.info("=" * 70)
    logger.info("Validation classification statistics")
    logger.info(
        f"Class mapping: "
        f"0=drowsy, 1=notdrowsy"
    )

    logger.info(
        f"True label counts: "
        f"drowsy={true_num[0].item()}, "
        f"notdrowsy={true_num[1].item()}"
    )

    logger.info(
        f"Predicted label counts: "
        f"drowsy={prediction_num[0].item()}, "
        f"notdrowsy={prediction_num[1].item()}"
    )

    logger.info("Confusion matrix:")
    logger.info(
        f"[[{true_0_pred_0}, {true_0_pred_1}],"
    )
    logger.info(
        f" [{true_1_pred_0}, {true_1_pred_1}]]"
    )

    logger.info(
        f"drowsy accuracy: "
        f"{class_0_acc:.2f}%"
    )

    logger.info(
        f"notdrowsy accuracy: "
        f"{class_1_acc:.2f}%"
    )

    # 判断是否全部预测为同一个类别
    total_predictions = prediction_num.sum().item()

    if prediction_num[0].item() == total_predictions:
        logger.warning(
            "All validation images were predicted as "
            "drowsy (class 0)."
        )

    elif prediction_num[1].item() == total_predictions:
        logger.warning(
            "All validation images were predicted as "
            "notdrowsy (class 1)."
        )

    logger.info("=" * 70)

    logger.info(
        f" * Acc@1 {acc1_meter.avg:.3f} "
        f"Loss {loss_meter.avg:.4f}"
    )
    return acc1_meter.avg, 0.0, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)

# torchrun --standalone --nproc_per_node=1 main_finetune.py \
#   --cfg configs/swin_base__800ep/simmim_finetune__swin_base__img224_window7__800ep.yaml \
#   --data-path /home/administrator/DFD-SCRM-main/data/NTHU-DDD \
#   --pretrained /home/administrator/DFD-SCRM-main/simmim_pretrain__swin_base__img192_window6__800ep.pth \
#   --batch-size 8 \
#   --amp-opt-level O0 \
#   --tag nthuddd_binary_30ep \
#   --opts \
#   TRAIN.EPOCHS 30 \
#   TRAIN.WARMUP_EPOCHS 1 \
#   TRAIN.AUTO_RESUME False 