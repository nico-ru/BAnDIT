import os
import random
import hydra
import logging
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from rich.progress import track

from src.datamodules.components.dataset_base import DatasetBase

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

OmegaConf.register_new_resolver(
    "dotsplit", lambda name: name.split(".")[-1], replace=True
)

REPLACE_TOKEN_RATE = 0.002
SKIP_TOKEN_RATE = 0.002
ADD_TOKEN_RATE = 0.002

SWAP_EVENT_RATE = 0.00
SKIP_EVENT_RATE = 0.00
REPEAT_EVENT_RATE = 0.00


REPLACE_TOKEN_CODE = 64
SKIP_TOKEN_CODE = 128
ADD_TOKEN_CODE = 256

SWAP_EVENT_CODE = 8
SKIP_EVENT_CODE = 16
REPEAT_EVENT_CODE = 32


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train.yaml")
def main(cfg: DictConfig):
    dataset: DatasetBase = hydra.utils.instantiate(cfg.dataset)

    annotations = dataset.annotations
    vocab = dataset.vocab

    for i, row in track(
        annotations.iterrows(), description="inducing anomalies", total=len(annotations)
    ):
        req_file = row["MESSAGE"]

        anomaly_code: int = 0
        dir = os.path.join(cfg.dataset.files_dir)
        with open(os.path.join(dir, "documents", req_file), "r") as file:  # type: ignore
            tokens = file.read().split(" ")

        if random.random() < REPLACE_TOKEN_RATE:
            tokens = replace_token(tokens, vocab)
            anomaly_code += REPLACE_TOKEN_CODE

        if random.random() < SKIP_TOKEN_RATE:
            tokens = skip_token(tokens)
            anomaly_code += SKIP_TOKEN_CODE

        if random.random() < ADD_TOKEN_RATE:
            tokens = add_token(tokens, vocab)
            anomaly_code += ADD_TOKEN_CODE

        if random.random() < SWAP_EVENT_RATE:
            replacement = annotations.iloc[i + 1, 1]  # type: ignore
            mem = row["TIMESTAMP"]
            annotations.at[i, "TIMESTAMP"] = replacement
            annotations.at[i + 1, "TIMESTAMP"] = mem  # type: ignore
            anomaly_code += SWAP_EVENT_CODE

        if random.random() < SKIP_EVENT_RATE:
            annotations.drop(i, inplace=True)
            annotations.sort_index(inplace=True)
            annotations.reset_index(drop=True, inplace=True)
            anomaly_code += SKIP_EVENT_CODE  # assuming the code will be assigned to the succeeding event

        if random.random() < REPEAT_EVENT_RATE:
            annotations.loc[i + 0.5] = annotations.loc[i]  # type: ignore
            annotations.sort_index(inplace=True)
            annotations.reset_index(drop=True, inplace=True)
            anomaly_code += REPEAT_EVENT_CODE

        mut_dir = os.path.join(dir, "documents_mut")
        if not os.path.exists(mut_dir):
            os.mkdir(mut_dir)

        with open(os.path.join(mut_dir, req_file), "w") as file:  # type: ignore
            file.write(" ".join(tokens))

        annotations.at[i, "CODE"] = anomaly_code

    annotations["CODE"] = annotations["CODE"].astype(int, errors="ignore")  # type: ignore
    path_splits = list(os.path.split(cfg.dataset.annotations))
    location = os.path.join(*path_splits[:-1], "annotations_mut.csv")
    annotations.to_csv(location, index=False)


#######################################
# Anomalies in the content of messages
#######################################


def replace_token(tokens, vocab):
    idx = random.choice(range(len(tokens)))
    replacement = vocab.sample()
    tokens[idx] = replacement.iloc[0, 0]
    return tokens


def skip_token(tokens):
    idx = random.choice(range(len(tokens)))
    tokens.pop(idx)
    return tokens


def add_token(tokens, vocab):
    idx = random.choice(range(len(tokens)))
    replacement = vocab.sample()
    tokens.insert(idx, replacement.iloc[0, 0])
    return tokens


if __name__ == "__main__":
    main()
