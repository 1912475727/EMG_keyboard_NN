# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


def instantiate_optimizer_and_scheduler(
    params: Iterator[nn.Parameter],
    optimizer_config: DictConfig,
    lr_scheduler_config: DictConfig,
    trainer: Any = None,
) -> dict[str, Any]:
    optimizer = instantiate(optimizer_config, params)
    sched_cfg = lr_scheduler_config.scheduler
    if trainer is not None and OmegaConf.select(sched_cfg, "_total_steps_from_trainer", default=False):
        total_steps = trainer.max_epochs * len(trainer.datamodule.train_dataloader())
        sched_dict = {k: v for k, v in OmegaConf.to_container(sched_cfg, resolve=True).items() if k != "_total_steps_from_trainer"}
        sched_dict["total_steps"] = total_steps
        sched_cfg = OmegaConf.create(sched_dict)
    scheduler = instantiate(sched_cfg, optimizer)
    lr_scheduler = instantiate(lr_scheduler_config, scheduler=scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": OmegaConf.to_container(lr_scheduler),
    }


def get_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def cpus_per_task(gpus_per_node: int, tasks_per_node: int, num_workers: int) -> int:
    """Number of CPUs to request per task per node taking into account
    the number of GPUs and dataloading workers."""
    gpus_per_task = gpus_per_node // tasks_per_node
    if gpus_per_task <= 0:
        return num_workers + 1
    else:
        return (num_workers + 1) * gpus_per_task
