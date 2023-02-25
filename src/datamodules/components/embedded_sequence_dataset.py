import datetime
import os
from typing import Callable, List, Optional, Union
import pandas as pd
import torch
from src.datamodules.components.dataset_base import TRANSFROM_TYPE, DatasetBase
from src.datamodules.components.sample_transformer import Padding


class EmbeddedSequenceDataset(DatasetBase):
    def __init__(
        self,
        annotations: pd.DataFrame,
        files_dir: str,
        services: Optional[List[str]] = None,
        vocab_file: Optional[str] = None,
        transform: TRANSFROM_TYPE = None,
        target_transform: Optional[Callable] = None,
        input_mapping_type: str = "index",
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        padding: Optional[Union[Padding, str]] = Padding.EMPTYS,
        load_tensors: bool = False,
    ):
        super().__init__(
            annotations,
            files_dir,
            services,
            vocab_file,
            transform,
            target_transform,
            input_mapping_type,
            start_date,
            end_date,
            padding,
        )

        self.load_tensors = load_tensors

        self.annotations["TIMESTAMP"] = pd.to_datetime(self.annotations["TIMESTAMP"])  # type: ignore
        self.annotations.sort_values(["CORRELATION_ID", "TIMESTAMP"], inplace=True)
        self.annotations.reset_index(drop=True, inplace=True)

        self._group_annotations()

    def _group_annotations(self):
        groups = self.annotations.groupby("CORRELATION_ID")  # type: ignore
        self.groups = groups
        self.group_keys = list(groups.indices.keys())

    def __getitem__(self, index: int):
        key = self.group_keys[index]
        sequence = self.groups.get_group(key)

        if self.load_tensors:
            location = os.path.join(self.files_dir, "cases", str(key))  # type: ignore
            req_tensors = [
                torch.load(os.path.join(location, f"{req_file}.pt"))
                for req_file in sequence["MESSAGE"].tolist()
            ]
            dimensions = tuple(req_tensors[0].shape)
            req_tensors.insert(0, torch.full(dimensions, 100))
            req_tensors.append(torch.full(dimensions, -100))
            return torch.stack(req_tensors).view(len(req_tensors), -1)

        if self.transform and self.transformer is not None:
            docs_path = os.path.join(self.files_dir, "documents")
            req_tensors = [
                os.path.join(docs_path, req_file)
                for req_file in sequence["MESSAGE"].tolist()
            ]
            sequence = []
            for location in req_tensors:
                file = open(location, "r")
                sample = file

                sequence.append(self.transformer(sample))

        return sequence

    def __len__(self) -> int:
        return len(self.groups)
