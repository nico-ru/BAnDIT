import os
from typing import List
import hydra
import logging
import pyrootutils
import json
from rich.progress import track
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.trainer import Trainer
from src.datamodules.components.dataset_base import DatasetBase
from src.datamodules.components.sample_transformer import SampleTransformer

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


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="eval.yaml")
def main(cfg: DictConfig):
    load_dynamic_cfg(cfg)
    print_config_tree(cfg)

    assert cfg.ckpt_path

    logger.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info(f"Instantiating loggers...")
    train_logger: List[LightningLoggerBase] = instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=train_logger
    )

    predictions = trainer.predict(
        model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
    )

    logger.info(f"Instantiating dataset <{cfg.datamodule._target_}>")
    dataset: DatasetBase = hydra.utils.instantiate(cfg.dataset)

    assert dataset.transformer is not None
    transformer: SampleTransformer = dataset.transformer

    assert predictions is not None
    diff_count = 0
    results = []
    for i, (reconstruct, input, loss) in track(
        enumerate(predictions),
        description="Writing predictions",
        total=len(predictions),
    ):
        target = transformer.retransform_sample(input)
        prediction = transformer.retransform_sample(reconstruct)
        pair = dict(target=target, prediction=prediction)

        predictions_dir = os.path.join(cfg.paths.output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        json.dump(pair, open(os.path.join(predictions_dir, str(i)), "w"))

        is_diff = target != prediction
        results.append(
            [i, dataset.annotations["MESSAGE"].iloc[i], loss.item(), is_diff]
        )

        if is_diff:
            diff_count += 1

    logger.info("Writing losses to csv")
    results_df = pandas.DataFrame(
        results, columns=["DS_INDEX", "MESSAGE", "LOSS", "IS_DIFFERENT"]
    )
    results_df.to_csv(os.path.join(cfg.paths.output_dir, "results.csv"), index=False)

    logger.info(f"found {diff_count} deviations")


if __name__ == "__main__":
    main()
