import datetime
import torch
import os
import pandas as pd
from io import TextIOWrapper
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Callable

from src.datamodules.components.sample_transformer import Padding, SampleTransformer


TRANSFROM_TYPE = Optional[
    Callable[
        [
            Union[List[str], TextIOWrapper],
            Dict,
            Optional[Callable],
        ],
        torch.Tensor,
    ]
]


class DatasetBase(Dataset):
    def __init__(
        self,
        annotations: Union[pd.DataFrame, str],
        files_dir: str,
        services: Optional[List[str]] = None,
        vocab_file: Optional[str] = None,
        transform: TRANSFROM_TYPE = None,
        target_transform: Optional[Callable] = None,
        input_mapping_type: str = "index",
        start_date: Optional[Union[datetime.datetime, str]] = None,
        end_date: Optional[Union[datetime.datetime, str]] = None,
        padding: Optional[Union[Padding, str]] = Padding.EMPTYS,
        *args,
        **kwargs,
    ):

        if isinstance(annotations, str):
            df = pd.read_csv(annotations)
            assert isinstance(
                df, pd.DataFrame
            ), f"path: {annotations} needs to point to valid csv"
            annotations = df

        self.services = services
        if self.services is not None:
            annotations = annotations.loc[annotations["SERVICE"].isin(self.services)]  # type: ignore

        assert isinstance(
            annotations, pd.DataFrame
        ), f"annotations has to be a dataframe"

        self.annotations: pd.DataFrame = annotations
        self.files_dir = files_dir
        self.vocab_file = vocab_file
        self.transform = transform
        self.target_transform = target_transform
        self.input_mapping_type = input_mapping_type
        self.start_date = start_date
        self.end_date = end_date

        if isinstance(padding, str):
            padding = Padding[padding.upper()]
        self.padding = padding
        self.vocab = None

        self._apply_date_range()

        self.size = 0
        self._init_dimension()

        self.transformer = None
        if self.transform is not None:
            self.load_vocab()
            assert self.vocab is not None
            self.transformer = SampleTransformer(
                self.annotations,
                self.vocab,
                self.size,
                self.transform,
                self.padding,
                self.input_mapping_type,
            )

    def __len__(self):
        return len(self.annotations)

    def _apply_date_range(self):
        self.annotations.loc["TIMESTAMP"] = pd.to_datetime(
            self.annotations["TIMESTAMP"]
        )
        if self.start_date:
            if isinstance(self.start_date, str):
                self.start_date = datetime.datetime.fromisoformat(self.start_date)
                self.start_date = self.start_date.replace(tzinfo=None)

            self.annotations = self.annotations.loc[
                self.annotations["TIMESTAMP"] >= self.start_date
            ]
        if self.end_date:
            if isinstance(self.end_date, str):
                self.end_date = datetime.datetime.fromisoformat(self.end_date)
                self.end_date = self.end_date.replace(tzinfo=None)

            self.annotations = self.annotations.loc[
                self.annotations["TIMESTAMP"] <= self.end_date
            ]

    def _init_dimension(self):
        file_path = os.path.join(self.files_dir, "size.txt")
        file = open(file_path, "r")
        self.size = int(file.read())

    def load_vocab(self):
        file_path = (
            self.vocab_file
            if self.vocab_file
            else os.path.join(self.files_dir, "vocab.txt")
        )
        self.vocab = pd.read_csv(file_path, delim_whitespace=True, names=["term", "frequency"])  # type: ignore

    def get_vocab(self):
        return self.vocab

    def load_and_get_vocab(self):
        self.load_vocab()
        return self.get_vocab()
