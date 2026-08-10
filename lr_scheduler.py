# # --------------------------------------------------------
# # SimMIM
# # Copyright (c) 2021 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
# # Written by Ze Liu
# # Modified by Zhenda Xie
# # --------------------------------------------------------

# from collections import Counter
# from bisect import bisect_right

# import torch
# from timm.scheduler.cosine_lr import CosineLRScheduler
# from timm.scheduler.step_lr import StepLRScheduler
# from timm.scheduler.scheduler import Scheduler


# def build_scheduler(config, optimizer, n_iter_per_epoch):
#     num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
#     warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
#     decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
#     multi_steps = [i * n_iter_per_epoch for i in config.TRAIN.LR_SCHEDULER.MULTISTEPS]

#     lr_scheduler = None
#     if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
#         lr_scheduler = CosineLRScheduler(
#             optimizer,
#             t_initial=num_steps,
#             t_mul=1.,
#             lr_min=config.TRAIN.MIN_LR,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             cycle_limit=1,
#             t_in_epochs=False,
#         )
#     elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
#         lr_scheduler = LinearLRScheduler(
#             optimizer,
#             t_initial=num_steps,
#             lr_min_rate=0.01,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             t_in_epochs=False,
#         )
#     elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
#         lr_scheduler = StepLRScheduler(
#             optimizer,
#             decay_t=decay_steps,
#             decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             t_in_epochs=False,
#         )
#     elif config.TRAIN.LR_SCHEDULER.NAME == 'multistep':
#         lr_scheduler = MultiStepLRScheduler(
#             optimizer,
#             milestones=multi_steps,
#             gamma=config.TRAIN.LR_SCHEDULER.GAMMA,
#             warmup_lr_init=config.TRAIN.WARMUP_LR,
#             warmup_t=warmup_steps,
#             t_in_epochs=False,
#         )

#     return lr_scheduler


# class LinearLRScheduler(Scheduler):
#     def __init__(self,
#                  optimizer: torch.optim.Optimizer,
#                  t_initial: int,
#                  lr_min_rate: float,
#                  warmup_t=0,
#                  warmup_lr_init=0.,
#                  t_in_epochs=True,
#                  noise_range_t=None,
#                  noise_pct=0.67,
#                  noise_std=1.0,
#                  noise_seed=42,
#                  initialize=True,
#                  ) -> None:
#         super().__init__(
#             optimizer, param_group_field="lr",
#             noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
#             initialize=initialize)

#         self.t_initial = t_initial
#         self.lr_min_rate = lr_min_rate
#         self.warmup_t = warmup_t
#         self.warmup_lr_init = warmup_lr_init
#         self.t_in_epochs = t_in_epochs
#         if self.warmup_t:
#             self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
#             super().update_groups(self.warmup_lr_init)
#         else:
#             self.warmup_steps = [1 for _ in self.base_values]

#     def _get_lr(self, t):
#         if t < self.warmup_t:
#             lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
#         else:
#             t = t - self.warmup_t
#             total_t = self.t_initial - self.warmup_t
#             lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
#         return lrs

#     def get_epoch_values(self, epoch: int):
#         if self.t_in_epochs:
#             return self._get_lr(epoch)
#         else:
#             return None

#     def get_update_values(self, num_updates: int):
#         if not self.t_in_epochs:
#             return self._get_lr(num_updates)
#         else:
#             return None


# class MultiStepLRScheduler(Scheduler):
#     def __init__(self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True) -> None:
#         super().__init__(optimizer, param_group_field="lr")
        
#         self.milestones = milestones
#         self.gamma = gamma
#         self.warmup_t = warmup_t
#         self.warmup_lr_init = warmup_lr_init
#         self.t_in_epochs = t_in_epochs
#         if self.warmup_t:
#             self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
#             super().update_groups(self.warmup_lr_init)
#         else:
#             self.warmup_steps = [1 for _ in self.base_values]
        
#         assert self.warmup_t <= min(self.milestones)
    
#     def _get_lr(self, t):
#         if t < self.warmup_t:
#             lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
#         else:
#             lrs = [v * (self.gamma ** bisect_right(self.milestones, t)) for v in self.base_values]
#         return lrs

#     def get_epoch_values(self, epoch: int):
#         if self.t_in_epochs:
#             return self._get_lr(epoch)
#         else:
#             return None

#     def get_update_values(self, num_updates: int):
#         if not self.t_in_epochs:
#             return self._get_lr(num_updates)
#         else:
#             return None


# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    multi_steps = [
        int(epoch * n_iter_per_epoch)
        for epoch in config.TRAIN.LR_SCHEDULER.MULTISTEPS
    ]

    scheduler_name = config.TRAIN.LR_SCHEDULER.NAME.lower()

    if scheduler_name == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.0,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    elif scheduler_name == "linear":
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    elif scheduler_name == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    elif scheduler_name == "multistep":
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=multi_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.GAMMA,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    else:
        raise NotImplementedError(
            f"Unsupported LR scheduler: {config.TRAIN.LR_SCHEDULER.NAME}"
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer,
        t_initial,
        lr_min_rate,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ):
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t
                for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [
                self.warmup_lr_init + t * step
                for step in self.warmup_steps
            ]

        t = t - self.warmup_t
        total_t = max(1, self.t_initial - self.warmup_t)

        return [
            v - ((v - v * self.lr_min_rate) * (t / total_t))
            for v in self.base_values
        ]

    def get_epoch_values(self, epoch):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None


class MultiStepLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer,
        decay_t,
        decay_rate=1.0,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ):
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t
                for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            return [
                self.warmup_lr_init + t * step
                for step in self.warmup_steps
            ]

        decay_count = sum(t >= milestone for milestone in self.decay_t)

        return [
            v * (self.decay_rate ** decay_count)
            for v in self.base_values
        ]

    def get_epoch_values(self, epoch):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        return None

    def get_update_values(self, num_updates):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        return None