# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
import shutil
import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        # Build module init kwargs from current config so checkpoints with missing/different
        # hparams (e.g. external user7.ckpt) still instantiate correctly.
        module_cfg = OmegaConf.to_container(config.module, resolve=True)
        assert isinstance(module_cfg, dict)
        module_kw = {k: v for k, v in module_cfg.items() if k != "_target_"}
        checkpoint_path = Path(config.checkpoint).expanduser().resolve()
        load_path = str(checkpoint_path)
        temp_ckpt = None
        # PyTorch 2.6+ defaults to weights_only=True; Lightning checkpoints contain OmegaConf/typing.
        _original_torch_load = torch.load
        try:
            def _torch_load_allow_pickle(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _original_torch_load(*args, **kwargs)
            torch.load = _torch_load_allow_pickle
            try:
                module = module.load_from_checkpoint(
                    load_path,
                    optimizer=config.optimizer,
                    lr_scheduler=config.lr_scheduler,
                    decoder=config.decoder,
                    **module_kw,
                )
            except PermissionError:
                # File locked (e.g. by another process or antivirus on Windows). Try loading from a copy.
                if not checkpoint_path.is_file():
                    raise
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".ckpt", delete=False, dir=checkpoint_path.parent
                )
                tmp.close()
                temp_ckpt = tmp.name
                shutil.copy2(checkpoint_path, temp_ckpt)
                log.info(f"Loaded from temp copy (original was locked): {temp_ckpt}")
                module = module.load_from_checkpoint(
                    temp_ckpt,
                    optimizer=config.optimizer,
                    lr_scheduler=config.lr_scheduler,
                    decoder=config.decoder,
                    **module_kw,
                )
        finally:
            torch.load = _original_torch_load
            if temp_ckpt is not None and os.path.isfile(temp_ckpt):
                try:
                    os.unlink(temp_ckpt)
                except OSError:
                    pass

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Instantiate logger(s) so Trainer receives Logger instances, not config dicts
    trainer_kw = dict(config.trainer)
    if "logger" in trainer_kw and trainer_kw["logger"] is not None:
        logger_configs = trainer_kw["logger"]
        trainer_kw["logger"] = [instantiate(cfg) for cfg in logger_configs]

    # Initialize trainer
    trainer = pl.Trainer(
        **trainer_kw,
        callbacks=callbacks,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        module = module.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Fixed path in project root so results are always findable (e.g. when running test from notebook)
    results_file = Path(working_dir) / "latest_test_results.txt"

    try:
        # Validate and test on the best checkpoint (if training), or on the loaded config.checkpoint (otherwise)
        val_metrics = trainer.validate(module, datamodule)
        test_metrics = trainer.test(module, datamodule)

        best_ckpt = (
            trainer.checkpoint_callback.best_model_path
            if config.train and getattr(trainer.checkpoint_callback, "best_model_path", None)
            else getattr(config, "checkpoint", None)
        )
        results = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": best_ckpt,
        }
        results_text = pprint.pformat(results, sort_dicts=False)
        log.info("Validation and test results:\n%s", results_text)
        pprint.pprint(results, sort_dicts=False)
        results_file.write_text(results_text, encoding="utf-8")
        log.info("Results written to %s", results_file.resolve())
        # Also write to run dir if different (Hydra may set cwd to run dir)
        run_dir_file = Path.cwd() / "test_results.txt"
        if run_dir_file.resolve() != results_file.resolve():
            run_dir_file.write_text(results_text, encoding="utf-8")
    except Exception as e:
        err_msg = f"Error during validate/test:\n{traceback.format_exc()}"
        log.exception("Error during validate/test")
        results_file.write_text(err_msg, encoding="utf-8")
        log.info("Error details written to %s", results_file.resolve())
        raise


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
