import os
import hydra
import logging
import pyrootutils
import shutil
from pytorch_lightning import LightningModule
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from src.datamodules.components.embedded_sequence_dataset import EmbeddedSequenceDataset
from src.utils.app_utils import (
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

    dataset: EmbeddedSequenceDataset = hydra.utils.instantiate(cfg.dataset)

    model_class = hydra.utils.get_class(cfg.model._target_)
    model: LightningModule = model_class.load_from_checkpoint(  # type: ignore
        checkpoint_path=cfg.ckpt_path
    )

    keys = dataset.group_keys
    for i in track(
        range(len(dataset)), description="transform samples", total=len(dataset)
    ):
        key = keys[i]
        group = dataset.groups.get_group(key)  # type: ignore
        sequence = dataset[i]

        out_dir = os.path.join(dataset.files_dir, "cases", str(key))  # type: ignore
        src_dir = os.path.join(dataset.files_dir, "documents")
        os.makedirs(out_dir, exist_ok=True)
        for k, sample in enumerate(sequence):  # type: ignore
            filename = group.iloc[k]["MESSAGE"]
            representation = model.encode(sample)  # type: ignore
            dest = os.path.join(out_dir, filename + ".pt")
            shutil.copy(
                os.path.join(src_dir, filename), os.path.join(out_dir, filename)
            )
            torch.save(representation.data, dest)


if __name__ == "__main__":
    main()
