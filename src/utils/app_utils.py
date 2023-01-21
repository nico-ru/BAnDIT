import logging
import hydra
from pandas import DataFrame
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
import rich
import rich.syntax
import rich.tree
from pathlib import Path
from typing import List, Sequence
from omegaconf import DictConfig, OmegaConf

from src.datamodules.components.dataset_base import DatasetBase

logger = logging.getLogger(__name__)


def load_dynamic_cfg(cfg: DictConfig):
    logger.info("loading dynamic properties of dataset")
    dataset: DatasetBase = hydra.utils.instantiate(cfg.dataset)

    if cfg.max_message_length is None and hasattr(dataset, "size"):
        cfg.max_message_length = dataset.size + 1  # type: ignore

    if cfg.vocab_size is None and hasattr(dataset, "get_vocab"):
        vocab = dataset.get_vocab()  # type: ignore
        assert isinstance(vocab, DataFrame)
        cfg.vocab_size = len(vocab)

    if cfg.embedded_size is None:
        cfg.embedded_size = 512


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.

    this function is taken from: https://github.com/ashleve/lightning-hydra-template
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    lit_logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        logger.warning("Logger config is empty.")
        return lit_logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            lit_logger.append(hydra.utils.instantiate(lg_conf))

    return lit_logger
