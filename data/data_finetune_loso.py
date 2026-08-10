import os
from typing import List, Tuple

from PIL import Image

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup, create_transform

try:
    from timm.data.transforms import str_to_interp_mode

    def _pil_interp(method):
        return str_to_interp_mode(method)
except ImportError:
    from timm.data.transforms import _pil_interp


IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
)


class LOSOImageDataset(Dataset):
    def __init__(
        self,
        root,
        held_out_subject,
        is_train,
        transform=None,
        class_names=("drowsy", "notdrowsy"),
    ):
        self.root = os.path.abspath(root)
        self.held_out_subject = held_out_subject
        self.is_train = is_train
        self.transform = transform
        self.classes = list(class_names)
        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.classes)
        }

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"LOSO 数据根目录不存在：{self.root}"
            )

        subject_dirs = sorted(
            entry.name
            for entry in os.scandir(self.root)
            if entry.is_dir()
        )

        if held_out_subject not in subject_dirs:
            raise ValueError(
                f"LOSO_SUBJECT={held_out_subject} 不存在。"
                f"可用被试：{subject_dirs}"
            )

        if is_train:
            self.subjects = [
                subject for subject in subject_dirs
                if subject != held_out_subject
            ]
        else:
            self.subjects = [held_out_subject]

        if not self.subjects:
            raise RuntimeError(
                "训练集没有可用被试，LOSO 至少需要两个被试。"
            )

        self.samples: List[Tuple[str, int]] = []
        self.class_sample_counts = {
            name: 0 for name in self.classes
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
                        f"缺少类别目录：{class_dir}"
                    )

                label = self.class_to_idx[class_name]

                for current_root, _, filenames in os.walk(class_dir):
                    for filename in sorted(filenames):
                        if filename.lower().endswith(IMG_EXTENSIONS):
                            image_path = os.path.join(
                                current_root,
                                filename,
                            )
                            self.samples.append((image_path, label))
                            self.class_sample_counts[class_name] += 1

        if not self.samples:
            raise RuntimeError("当前 LOSO 划分没有检测到图片。")

        for class_name, count in self.class_sample_counts.items():
            if count == 0:
                raise RuntimeError(
                    f"类别 {class_name} 在当前划分中没有图片。"
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
            f"训练集和验证集类别数不一致："
            f"train={num_classes}, val={val_num_classes}"
        )

    if dataset_train.class_to_idx != dataset_val.class_to_idx:
        raise ValueError(
            "训练集与验证集类别映射不一致。\n"
            f"train={dataset_train.class_to_idx}\n"
            f"val={dataset_val.class_to_idx}"
        )

    logger.info("=" * 70)
    logger.info(
        f"LOSO held-out subject: {config.DATA.LOSO_SUBJECT}"
    )
    logger.info(
        f"Train subjects ({len(dataset_train.subjects)}): "
        f"{dataset_train.subjects}"
    )
    logger.info(
        f"Validation subject: {dataset_val.subjects}"
    )
    logger.info(
        f"Train images: {len(dataset_train)}, "
        f"Validation images: {len(dataset_val)}"
    )
    logger.info(
        f"Class mapping: {dataset_train.class_to_idx}"
    )
    logger.info(
        f"Train class counts: {dataset_train.class_sample_counts}"
    )
    logger.info(
        f"Validation class counts: {dataset_val.class_sample_counts}"
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
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    )


def build_dataset(is_train, config, logger):
    transform = build_transform(is_train, config)

    logger.info(
        f"Fine-tune data transform, is_train={is_train}:\n"
        f"{transform}"
    )

    if config.DATA.DATASET != "imagenet":
        raise NotImplementedError(
            f"Unsupported dataset type: {config.DATA.DATASET}"
        )

    held_out_subject = str(
        config.DATA.LOSO_SUBJECT
    ).strip()

    if not held_out_subject:
        raise ValueError(
            "DATA.LOSO_SUBJECT 不能为空，例如 subject_01。"
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
                if config.AUG.AUTO_AUGMENT != "none"
                else None
            ),
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE,
                padding=4,
            )

        return transform

    transform_list = []

    if resize_im:
        if config.TEST.CROP:
            size = int(
                (256 / 224) * config.DATA.IMG_SIZE
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

    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD,
        )
    )

    return transforms.Compose(transform_list)
