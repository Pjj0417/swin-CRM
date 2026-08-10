# # --------------------------------------------------------
# # SimMIM
# # Copyright (c) 2021 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Zhenda Xie
# # --------------------------------------------------------

# import os
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from torchvision import datasets, transforms
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import Mixup
# from timm.data import create_transform
# try:
#     from timm.data.transforms import str_to_interp_mode

#     def _pil_interp(method):
#         return str_to_interp_mode(method)

# except ImportError:
#     from timm.data.transforms import _pil_interp


# def build_loader_finetune(config, logger):
#     config.defrost()
#     dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
#     config.freeze()
#     dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
#     logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

#     num_tasks = dist.get_world_size()
#     global_rank = dist.get_rank()
#     sampler_train = DistributedSampler(
#         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
#     )
#     sampler_val = DistributedSampler(
#         dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
#     )

#     data_loader_train = DataLoader(
#         dataset_train, sampler=sampler_train,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=True,
#     )

#     data_loader_val = DataLoader(
#         dataset_val, sampler=sampler_val,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=False,
#     )

#     # setup mixup / cutmix
#     mixup_fn = None
#     mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
#     if mixup_active:
#         mixup_fn = Mixup(
#             mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
#             prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
#             label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

#     return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


# def build_dataset(is_train, config, logger):
#     transform = build_transform(is_train, config)
#     logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
    
#     if config.DATA.DATASET == 'imagenet':
#         prefix = 'train' if is_train else 'val'
#         root = os.path.join(config.DATA.DATA_PATH, prefix)
#         dataset = datasets.ImageFolder(root, transform=transform)
#         nb_classes = 1000
#     else:
#         raise NotImplementedError("We only support ImageNet Now.")

#     return dataset, nb_classes


# def build_transform(is_train, config):
#     resize_im = config.DATA.IMG_SIZE > 32
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=config.DATA.IMG_SIZE,
#             is_training=True,
#             color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
#             auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATA.INTERPOLATION,
#         )
#         if not resize_im:
#             # replace RandomResizedCropAndInterpolation with
#             # RandomCrop
#             transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
#         return transform

#     t = []
#     if resize_im:
#         if config.TEST.CROP:
#             size = int((256 / 224) * config.DATA.IMG_SIZE)
#             t.append(
#                 transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
#                 # to maintain same ratio w.r.t. 224 images
#             )
#             t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
#         else:
#             t.append(
#                 transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
#                                   interpolation=_pil_interp(config.DATA.INTERPOLATION))
#             )

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)

# #新的v1，可以的
# # --------------------------------------------------------
# # SimMIM
# # Copyright (c) 2021 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Zhenda Xie
# # --------------------------------------------------------

# import os

# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from torchvision import datasets, transforms

# from timm.data.constants import (
#     IMAGENET_DEFAULT_MEAN,
#     IMAGENET_DEFAULT_STD,
# )
# from timm.data import Mixup, create_transform

# try:
#     from timm.data.transforms import str_to_interp_mode

#     def _pil_interp(method):
#         return str_to_interp_mode(method)

# except ImportError:
#     from timm.data.transforms import _pil_interp


# def build_loader_finetune(config, logger):
#     # 构建训练集，并根据实际文件夹数量设置类别数
#     dataset_train, num_classes = build_dataset(
#         is_train=True,
#         config=config,
#         logger=logger,
#     )

#     config.defrost()
#     config.MODEL.NUM_CLASSES = num_classes
#     config.freeze()

#     # 构建验证集
#     dataset_val, val_num_classes = build_dataset(
#         is_train=False,
#         config=config,
#         logger=logger,
#     )

#     # 检查训练集和验证集的类别数量是否一致
#     if num_classes != val_num_classes:
#         raise ValueError(
#             f"训练集和验证集类别数不一致："
#             f"train={num_classes}, val={val_num_classes}"
#         )

#     # 检查类别名称及标签映射是否完全一致
#     if dataset_train.class_to_idx != dataset_val.class_to_idx:
#         raise ValueError(
#             "训练集与验证集的类别映射不一致。\n"
#             f"train class_to_idx: {dataset_train.class_to_idx}\n"
#             f"val class_to_idx: {dataset_val.class_to_idx}"
#         )

#     logger.info(
#         f"Build dataset: "
#         f"train images = {len(dataset_train)}, "
#         f"val images = {len(dataset_val)}"
#     )
#     logger.info(
#         f"Detected classes ({num_classes}): "
#         f"{dataset_train.classes}"
#     )
#     logger.info(
#         f"Class mapping: {dataset_train.class_to_idx}"
#     )

#     num_tasks = dist.get_world_size()
#     global_rank = dist.get_rank()

#     # 训练集必须打乱
#     sampler_train = DistributedSampler(
#         dataset_train,
#         num_replicas=num_tasks,
#         rank=global_rank,
#         shuffle=True,
#     )

#     # 验证集不打乱是正常做法
#     sampler_val = DistributedSampler(
#         dataset_val,
#         num_replicas=num_tasks,
#         rank=global_rank,
#         shuffle=False,
#     )

#     data_loader_train = DataLoader(
#         dataset_train,
#         sampler=sampler_train,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=True,
#     )

#     data_loader_val = DataLoader(
#         dataset_val,
#         sampler=sampler_val,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=False,
#     )

#     # 设置 Mixup / CutMix
#     mixup_fn = None

#     mixup_active = (
#         config.AUG.MIXUP > 0
#         or config.AUG.CUTMIX > 0
#         or config.AUG.CUTMIX_MINMAX is not None
#     )

#     if mixup_active:
#         mixup_fn = Mixup(
#             mixup_alpha=config.AUG.MIXUP,
#             cutmix_alpha=config.AUG.CUTMIX,
#             cutmix_minmax=config.AUG.CUTMIX_MINMAX,
#             prob=config.AUG.MIXUP_PROB,
#             switch_prob=config.AUG.MIXUP_SWITCH_PROB,
#             mode=config.AUG.MIXUP_MODE,
#             label_smoothing=config.MODEL.LABEL_SMOOTHING,
#             num_classes=config.MODEL.NUM_CLASSES,
#         )

#     return (
#         dataset_train,
#         dataset_val,
#         data_loader_train,
#         data_loader_val,
#         mixup_fn,
#     )


# def build_dataset(is_train, config, logger):
#     transform = build_transform(is_train, config)

#     logger.info(
#         f"Fine-tune data transform, is_train={is_train}:\n"
#         f"{transform}"
#     )

#     if config.DATA.DATASET != "imagenet":
#         raise NotImplementedError(
#             f"Unsupported dataset type: {config.DATA.DATASET}"
#         )

#     prefix = "train" if is_train else "val"
#     root = os.path.join(config.DATA.DATA_PATH, prefix)

#     if not os.path.isdir(root):
#         raise FileNotFoundError(
#             f"数据目录不存在：{root}\n"
#             "要求目录结构为：\n"
#             "NTHU-DDD/train/类别名/图片\n"
#             "NTHU-DDD/val/类别名/图片"
#         )

#     dataset = datasets.ImageFolder(
#         root=root,
#         transform=transform,
#     )

#     # 关键修改：根据实际类别文件夹自动确定类别数量
#     num_classes = len(dataset.classes)

#     if num_classes < 2:
#         raise ValueError(
#             f"{root} 中只检测到 {num_classes} 个类别，"
#             "分类任务至少需要两个类别文件夹。"
#         )

#     logger.info(
#         f"Dataset root: {root}, "
#         f"images: {len(dataset)}, "
#         f"classes: {dataset.classes}, "
#         f"class_to_idx: {dataset.class_to_idx}"
#     )

#     return dataset, num_classes


# def build_transform(is_train, config):
#     resize_im = config.DATA.IMG_SIZE > 32

#     if is_train:
#         transform = create_transform(
#             input_size=config.DATA.IMG_SIZE,
#             is_training=True,
#             color_jitter=(
#                 config.AUG.COLOR_JITTER
#                 if config.AUG.COLOR_JITTER > 0
#                 else None
#             ),
#             auto_augment=(
#                 config.AUG.AUTO_AUGMENT
#                 if config.AUG.AUTO_AUGMENT != "none"
#                 else None
#             ),
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATA.INTERPOLATION,
#         )

#         if not resize_im:
#             transform.transforms[0] = transforms.RandomCrop(
#                 config.DATA.IMG_SIZE,
#                 padding=4,
#             )

#         return transform

#     transform_list = []

#     if resize_im:
#         if config.TEST.CROP:
#             size = int(
#                 (256 / 224) * config.DATA.IMG_SIZE
#             )

#             transform_list.append(
#                 transforms.Resize(
#                     size,
#                     interpolation=_pil_interp(
#                         config.DATA.INTERPOLATION
#                     ),
#                 )
#             )

#             transform_list.append(
#                 transforms.CenterCrop(
#                     config.DATA.IMG_SIZE
#                 )
#             )

#         else:
#             transform_list.append(
#                 transforms.Resize(
#                     (
#                         config.DATA.IMG_SIZE,
#                         config.DATA.IMG_SIZE,
#                     ),
#                     interpolation=_pil_interp(
#                         config.DATA.INTERPOLATION
#                     ),
#                 )
#             )

#     transform_list.append(transforms.ToTensor())

#     transform_list.append(
#         transforms.Normalize(
#             IMAGENET_DEFAULT_MEAN,
#             IMAGENET_DEFAULT_STD,
#         )
#     )

#     return transforms.Compose(transform_list)

# #v2
# import os
# from typing import List, Tuple

# from PIL import Image

# import torch.distributed as dist
# from torch.utils.data import DataLoader, Dataset, DistributedSampler
# from torchvision import transforms

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import Mixup, create_transform

# try:
#     from timm.data.transforms import str_to_interp_mode

#     def _pil_interp(method):
#         return str_to_interp_mode(method)
# except ImportError:
#     from timm.data.transforms import _pil_interp


# IMG_EXTENSIONS = (
#     ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
# )


# class LOSOImageDataset(Dataset):
#     def __init__(
#         self,
#         root,
#         held_out_subject,
#         is_train,
#         transform=None,
#         class_names=("fatigue", "nofatigue"),
#         # class_names=("drowsy", "notdrowsy"),
#     ):
#         self.root = os.path.abspath(root)
#         self.held_out_subject = held_out_subject
#         self.is_train = is_train
#         self.transform = transform
#         self.classes = list(class_names)
#         self.class_to_idx = {
#             name: idx for idx, name in enumerate(self.classes)
#         }

#         if not os.path.isdir(self.root):
#             raise FileNotFoundError(
#                 f"LOSO 数据根目录不存在：{self.root}"
#             )

#         subject_dirs = sorted(
#             entry.name
#             for entry in os.scandir(self.root)
#             if entry.is_dir()
#         )

#         if held_out_subject not in subject_dirs:
#             raise ValueError(
#                 f"LOSO_SUBJECT={held_out_subject} 不存在。"
#                 f"可用被试：{subject_dirs}"
#             )

#         if is_train:
#             self.subjects = [
#                 subject for subject in subject_dirs
#                 if subject != held_out_subject
#             ]
#         else:
#             self.subjects = [held_out_subject]

#         if not self.subjects:
#             raise RuntimeError(
#                 "训练集没有可用被试，LOSO 至少需要两个被试。"
#             )

#         self.samples: List[Tuple[str, int]] = []
#         self.class_sample_counts = {
#             name: 0 for name in self.classes
#         }

#         for subject in self.subjects:
#             for class_name in self.classes:
#                 class_dir = os.path.join(
#                     self.root,
#                     subject,
#                     class_name,
#                 )
#                 if not os.path.isdir(class_dir):
#                     raise FileNotFoundError(
#                         f"缺少类别目录：{class_dir}"
#                     )

#                 label = self.class_to_idx[class_name]

#                 for current_root, _, filenames in os.walk(class_dir):
#                     for filename in sorted(filenames):
#                         if filename.lower().endswith(IMG_EXTENSIONS):
#                             image_path = os.path.join(
#                                 current_root,
#                                 filename,
#                             )
#                             self.samples.append((image_path, label))
#                             self.class_sample_counts[class_name] += 1

#         if not self.samples:
#             raise RuntimeError("当前 LOSO 划分没有检测到图片。")

#         for class_name, count in self.class_sample_counts.items():
#             if count == 0:
#                 raise RuntimeError(
#                     f"类别 {class_name} 在当前划分中没有图片。"
#                 )

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, index):
#         image_path, target = self.samples[index]

#         with Image.open(image_path) as image:
#             image = image.convert("RGB")

#         if self.transform is not None:
#             image = self.transform(image)

#         return image, target


# def build_loader_finetune(config, logger):
#     dataset_train, num_classes = build_dataset(
#         is_train=True,
#         config=config,
#         logger=logger,
#     )

#     config.defrost()
#     config.MODEL.NUM_CLASSES = num_classes
#     config.freeze()

#     dataset_val, val_num_classes = build_dataset(
#         is_train=False,
#         config=config,
#         logger=logger,
#     )

#     if num_classes != val_num_classes:
#         raise ValueError(
#             f"训练集和验证集类别数不一致："
#             f"train={num_classes}, val={val_num_classes}"
#         )

#     if dataset_train.class_to_idx != dataset_val.class_to_idx:
#         raise ValueError(
#             "训练集与验证集类别映射不一致。\n"
#             f"train={dataset_train.class_to_idx}\n"
#             f"val={dataset_val.class_to_idx}"
#         )

#     logger.info("=" * 70)
#     logger.info(
#         f"LOSO held-out subject: {config.DATA.LOSO_SUBJECT}"
#     )
#     logger.info(
#         f"Train subjects ({len(dataset_train.subjects)}): "
#         f"{dataset_train.subjects}"
#     )
#     logger.info(
#         f"Validation subject: {dataset_val.subjects}"
#     )
#     logger.info(
#         f"Train images: {len(dataset_train)}, "
#         f"Validation images: {len(dataset_val)}"
#     )
#     logger.info(
#         f"Class mapping: {dataset_train.class_to_idx}"
#     )
#     logger.info(
#         f"Train class counts: {dataset_train.class_sample_counts}"
#     )
#     logger.info(
#         f"Validation class counts: {dataset_val.class_sample_counts}"
#     )
#     logger.info("=" * 70)

#     num_tasks = dist.get_world_size()
#     global_rank = dist.get_rank()

#     sampler_train = DistributedSampler(
#         dataset_train,
#         num_replicas=num_tasks,
#         rank=global_rank,
#         shuffle=True,
#     )

#     sampler_val = DistributedSampler(
#         dataset_val,
#         num_replicas=num_tasks,
#         rank=global_rank,
#         shuffle=False,
#     )

#     data_loader_train = DataLoader(
#         dataset_train,
#         sampler=sampler_train,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=True,
#     )

#     data_loader_val = DataLoader(
#         dataset_val,
#         sampler=sampler_val,
#         batch_size=config.DATA.BATCH_SIZE,
#         num_workers=config.DATA.NUM_WORKERS,
#         pin_memory=config.DATA.PIN_MEMORY,
#         drop_last=False,
#     )

#     mixup_fn = None
#     mixup_active = (
#         config.AUG.MIXUP > 0
#         or config.AUG.CUTMIX > 0
#         or config.AUG.CUTMIX_MINMAX is not None
#     )

#     if mixup_active:
#         mixup_fn = Mixup(
#             mixup_alpha=config.AUG.MIXUP,
#             cutmix_alpha=config.AUG.CUTMIX,
#             cutmix_minmax=config.AUG.CUTMIX_MINMAX,
#             prob=config.AUG.MIXUP_PROB,
#             switch_prob=config.AUG.MIXUP_SWITCH_PROB,
#             mode=config.AUG.MIXUP_MODE,
#             label_smoothing=config.MODEL.LABEL_SMOOTHING,
#             num_classes=config.MODEL.NUM_CLASSES,
#         )

#     return (
#         dataset_train,
#         dataset_val,
#         data_loader_train,
#         data_loader_val,
#         mixup_fn,
#     )


# def build_dataset(is_train, config, logger):
#     transform = build_transform(is_train, config)

#     logger.info(
#         f"Fine-tune data transform, is_train={is_train}:\n"
#         f"{transform}"
#     )

#     if config.DATA.DATASET != "imagenet":
#         raise NotImplementedError(
#             f"Unsupported dataset type: {config.DATA.DATASET}"
#         )

#     held_out_subject = str(
#         config.DATA.LOSO_SUBJECT
#     ).strip()

#     if not held_out_subject:
#         raise ValueError(
#             "DATA.LOSO_SUBJECT 不能为空，例如 subject_01。"
#         )

#     dataset = LOSOImageDataset(
#         root=config.DATA.DATA_PATH,
#         held_out_subject=held_out_subject,
#         is_train=is_train,
#         transform=transform,
#     )

#     return dataset, len(dataset.classes)


# def build_transform(is_train, config):
#     resize_im = config.DATA.IMG_SIZE > 32

#     if is_train:
#         transform = create_transform(
#             input_size=config.DATA.IMG_SIZE,
#             is_training=True,
#             color_jitter=(
#                 config.AUG.COLOR_JITTER
#                 if config.AUG.COLOR_JITTER > 0
#                 else None
#             ),
#             auto_augment=(
#                 config.AUG.AUTO_AUGMENT
#                 if config.AUG.AUTO_AUGMENT != "none"
#                 else None
#             ),
#             re_prob=config.AUG.REPROB,
#             re_mode=config.AUG.REMODE,
#             re_count=config.AUG.RECOUNT,
#             interpolation=config.DATA.INTERPOLATION,
#         )

#         if not resize_im:
#             transform.transforms[0] = transforms.RandomCrop(
#                 config.DATA.IMG_SIZE,
#                 padding=4,
#             )

#         return transform

#     transform_list = []

#     if resize_im:
#         if config.TEST.CROP:
#             size = int(
#                 (256 / 224) * config.DATA.IMG_SIZE
#             )
#             transform_list.append(
#                 transforms.Resize(
#                     size,
#                     interpolation=_pil_interp(
#                         config.DATA.INTERPOLATION
#                     ),
#                 )
#             )
#             transform_list.append(
#                 transforms.CenterCrop(
#                     config.DATA.IMG_SIZE
#                 )
#             )
#         else:
#             transform_list.append(
#                 transforms.Resize(
#                     (
#                         config.DATA.IMG_SIZE,
#                         config.DATA.IMG_SIZE,
#                     ),
#                     interpolation=_pil_interp(
#                         config.DATA.INTERPOLATION
#                     ),
#                 )
#             )

#     transform_list.append(transforms.ToTensor())
#     transform_list.append(
#         transforms.Normalize(
#             IMAGENET_DEFAULT_MEAN,
#             IMAGENET_DEFAULT_STD,
#         )
#     )

#     return transforms.Compose(transform_list)


# v2
import os
from typing import List, Tuple

from PIL import Image

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from timm.data import Mixup, create_transform


try:
    from timm.data.transforms import str_to_interp_mode

    def _pil_interp(method):
        return str_to_interp_mode(method)

except ImportError:
    from timm.data.transforms import _pil_interp


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)


# ============================================================
# 被试目录名与姓名拼音的对应关系
#
# 注意：
# 1. 数据目录仍然保持 subject01、subject02 等名称。
# 2. 这里仅用于日志显示，不会影响数据读取。
# 3. 如有姓名拼音不准确，可以直接修改右侧字符串。
# ============================================================
SUBJECT_NAME_MAP = {
    "subject01": "zhou_xinghong",
    "subject02": "wang_chaoting",
    "subject03": "wang_hao",
    "subject04": "tang_xingguang",
    "subject05": "gong_wanyin",
    "subject06": "peng_jiajun",
    "subject07": "duan_yunxin",
    "subject08": "chen_naiyu",
    "subject09": "li_hailong",
    "subject10": "li_zekun",
    "subject11": "zhang_yingzhi",
    "subject12": "zhang_guigeng",
    "subject13": "jiang_pengfei",
    "subject14": "cao_zilong",
    "subject15": "hu_qinlin",
}


def display_subject_name(subject_name):
    """
    将内部目录名转换为用于日志显示的姓名拼音。

    示例：
        subject02 -> wang_chaoting

    如果映射表中不存在，则返回原名称。
    """
    subject_name = str(subject_name).strip()
    return SUBJECT_NAME_MAP.get(subject_name, subject_name)


def display_subject_list(subjects):
    """
    将一组被试目录名转换成姓名拼音列表。
    """
    return [
        display_subject_name(subject)
        for subject in subjects
    ]


class LOSOImageDataset(Dataset):
    def __init__(
        self,
        root,
        held_out_subject,
        is_train,
        transform=None,
        class_names=("fatigue", "nofatigue"),
    ):
        self.root = os.path.abspath(root)
        self.held_out_subject = held_out_subject
        self.is_train = is_train
        self.transform = transform

        self.classes = list(class_names)

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"LOSO 数据根目录不存在：{self.root}"
            )

        # 只读取真正的被试目录。
        #
        # 过滤以下目录：
        #   .ipynb_checkpoints
        #   __pycache__
        #   其他不是 subject 开头的目录
        subject_dirs = sorted(
            entry.name
            for entry in os.scandir(self.root)
            if (
                entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name.startswith("subject")
            )
        )

        if held_out_subject not in subject_dirs:
            available_subjects = display_subject_list(
                subject_dirs
            )

            raise ValueError(
                f"LOSO_SUBJECT={held_out_subject} 不存在。"
                f"可用被试：{available_subjects}"
            )

        if is_train:
            self.subjects = [
                subject
                for subject in subject_dirs
                if subject != held_out_subject
            ]
        else:
            self.subjects = [held_out_subject]

        if not self.subjects:
            raise RuntimeError(
                "训练集没有可用被试，"
                "LOSO 至少需要两个被试。"
            )

        self.samples: List[Tuple[str, int]] = []

        self.class_sample_counts = {
            name: 0
            for name in self.classes
        }

        for subject in self.subjects:
            for class_name in self.classes:
                class_dir = os.path.join(
                    self.root,
                    subject,
                    class_name,
                )

                if not os.path.isdir(class_dir):
                    raise FileNotFoundError(
                        "缺少类别目录："
                        f"{class_dir}，"
                        "对应被试："
                        f"{display_subject_name(subject)}"
                    )

                label = self.class_to_idx[class_name]

                for current_root, _, filenames in os.walk(
                    class_dir
                ):
                    for filename in sorted(filenames):
                        if filename.lower().endswith(
                            IMG_EXTENSIONS
                        ):
                            image_path = os.path.join(
                                current_root,
                                filename,
                            )

                            self.samples.append(
                                (
                                    image_path,
                                    label,
                                )
                            )

                            self.class_sample_counts[
                                class_name
                            ] += 1

        if not self.samples:
            raise RuntimeError(
                "当前 LOSO 划分没有检测到图片。"
            )

        for class_name, count in (
            self.class_sample_counts.items()
        ):
            if count == 0:
                raise RuntimeError(
                    f"类别 {class_name} "
                    "在当前划分中没有图片。"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def build_loader_finetune(config, logger):
    dataset_train, num_classes = build_dataset(
        is_train=True,
        config=config,
        logger=logger,
    )

    config.defrost()
    config.MODEL.NUM_CLASSES = num_classes
    config.freeze()

    dataset_val, val_num_classes = build_dataset(
        is_train=False,
        config=config,
        logger=logger,
    )

    if num_classes != val_num_classes:
        raise ValueError(
            "训练集和验证集类别数不一致："
            f"train={num_classes}, "
            f"val={val_num_classes}"
        )

    if (
        dataset_train.class_to_idx
        != dataset_val.class_to_idx
    ):
        raise ValueError(
            "训练集与验证集类别映射不一致。\n"
            f"train={dataset_train.class_to_idx}\n"
            f"val={dataset_val.class_to_idx}"
        )

    held_out_display_name = display_subject_name(
        config.DATA.LOSO_SUBJECT
    )

    train_display_names = display_subject_list(
        dataset_train.subjects
    )

    val_display_names = display_subject_list(
        dataset_val.subjects
    )

    logger.info("=" * 70)

    logger.info(
        "LOSO held-out subject: "
        f"{held_out_display_name}"
    )

    logger.info(
        f"Train subjects "
        f"({len(dataset_train.subjects)}): "
        f"{train_display_names}"
    )

    logger.info(
        "Validation subject: "
        f"{val_display_names}"
    )

    logger.info(
        f"Train images: {len(dataset_train)}, "
        f"Validation images: {len(dataset_val)}"
    )

    logger.info(
        "Class mapping: "
        f"{dataset_train.class_to_idx}"
    )

    logger.info(
        "Train class counts: "
        f"{dataset_train.class_sample_counts}"
    )

    logger.info(
        "Validation class counts: "
        f"{dataset_val.class_sample_counts}"
    )

    logger.info("=" * 70)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

    sampler_val = DistributedSampler(
        dataset_val,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=False,
    )

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    mixup_fn = None

    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0
        or config.AUG.CUTMIX_MINMAX is not None
    )

    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=(
                config.AUG.CUTMIX_MINMAX
            ),
            prob=config.AUG.MIXUP_PROB,
            switch_prob=(
                config.AUG.MIXUP_SWITCH_PROB
            ),
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=(
                config.MODEL.LABEL_SMOOTHING
            ),
            num_classes=(
                config.MODEL.NUM_CLASSES
            ),
        )

    return (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    )


def build_dataset(is_train, config, logger):
    transform = build_transform(
        is_train,
        config,
    )

    logger.info(
        "Fine-tune data transform, "
        f"is_train={is_train}:\n"
        f"{transform}"
    )

    if config.DATA.DATASET != "imagenet":
        raise NotImplementedError(
            "Unsupported dataset type: "
            f"{config.DATA.DATASET}"
        )

    held_out_subject = str(
        config.DATA.LOSO_SUBJECT
    ).strip()

    if not held_out_subject:
        raise ValueError(
            "DATA.LOSO_SUBJECT 不能为空，"
            "例如 subject02。"
        )

    dataset = LOSOImageDataset(
        root=config.DATA.DATA_PATH,
        held_out_subject=held_out_subject,
        is_train=is_train,
        transform=transform,
    )

    return dataset, len(dataset.classes)


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32

    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=(
                config.AUG.COLOR_JITTER
                if config.AUG.COLOR_JITTER > 0
                else None
            ),
            auto_augment=(
                config.AUG.AUTO_AUGMENT
                if (
                    config.AUG.AUTO_AUGMENT
                    != "none"
                )
                else None
            ),
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=(
                config.DATA.INTERPOLATION
            ),
        )

        if not resize_im:
            transform.transforms[0] = (
                transforms.RandomCrop(
                    config.DATA.IMG_SIZE,
                    padding=4,
                )
            )

        return transform

    transform_list = []

    if resize_im:
        if config.TEST.CROP:
            size = int(
                (256 / 224)
                * config.DATA.IMG_SIZE
            )

            transform_list.append(
                transforms.Resize(
                    size,
                    interpolation=_pil_interp(
                        config.DATA.INTERPOLATION
                    ),
                )
            )

            transform_list.append(
                transforms.CenterCrop(
                    config.DATA.IMG_SIZE
                )
            )

        else:
            transform_list.append(
                transforms.Resize(
                    (
                        config.DATA.IMG_SIZE,
                        config.DATA.IMG_SIZE,
                    ),
                    interpolation=_pil_interp(
                        config.DATA.INTERPOLATION
                    ),
                )
            )

    transform_list.append(
        transforms.ToTensor()
    )

    transform_list.append(
        transforms.Normalize(
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD,
        )
    )

    return transforms.Compose(
        transform_list
    )

