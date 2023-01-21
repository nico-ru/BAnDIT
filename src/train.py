import os
import hydra
import logging
import pyrootutils
from typing import List
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.trainer import Trainer

from src.utils.app_utils import (
    instantiate_callbacks,
    instantiate_loggers,
    load_dynamic_cfg,
    print_config_tree,
)

import pandas

pandas.options.mode.chained_assignment = None


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

# LOAD ENV VARIABLES & INSTANTIATE LOGGER
CONFIG_DIR = os.path.join(root, "config")

logger = logging.getLogger(__name__)


# REGISTER CONFIG RESOLVERS
OmegaConf.register_new_resolver("start", lambda v: [v - 2], replace=True)
OmegaConf.register_new_resolver("end", lambda v: [v - 4], replace=True)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train.yaml")
def main(cfg: DictConfig):
    load_dynamic_cfg(cfg)
    print_config_tree(cfg)

    logger.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info(f"Instantiating loggers...")
    train_logger: List[LightningLoggerBase] = instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=train_logger
    )

    trainer.fit(model=model, datamodule=datamodule)
    train_metrics = trainer.callback_metrics

    trainer.test(model=model, datamodule=datamodule)
    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}
    for name, value in metric_dict.items():
        logger.info(f"{name}: {value.item():0.4f}")

    ckpt_path = trainer.checkpoint_callback.best_model_path  # type: ignore
    logger.info(f"Best ckpt path: {ckpt_path}")


if __name__ == "__main__":
    main()
