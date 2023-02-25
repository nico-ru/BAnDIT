import datetime
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable, List, Optional


class SequenceDataset(Dataset):
    def __init__(
        self,
        annotations,
        events: str,
        services: Optional[List[str]] = None,
        concept_name: str = "ACTIVITY",
        transform: Optional[Callable] = lambda x: torch.unsqueeze(torch.tensor(x), 1),
        encode: Optional[Callable] = lambda x, _: x,
        filter: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if isinstance(annotations, str):
            df = pd.read_csv(annotations)
            assert isinstance(
                df, pd.DataFrame
            ), f"path: {annotations} needs to point to valid csv"
            annotations = df

        assert isinstance(
            annotations, pd.DataFrame
        ), f"annotations has to be a dataframe"

        self.services = services
        if self.services is not None:
            annotations = annotations.loc[annotations["SERVICE"].isin(self.services)]  # type: ignore

        self.annotations = annotations
        self.concept_name = concept_name
        self.transform = transform
        self.encode = encode
        self.filter = filter

        self.annotations["TIMESTAMP"] = pd.to_datetime(self.annotations["TIMESTAMP"])  # type: ignore
        self.annotations.sort_values(["CORRELATION_ID", "TIMESTAMP"], inplace=True)
        self.annotations.reset_index(drop=True, inplace=True)

        self.start = "START"
        self.end = "END"

        self._types = json.load(open(events, "r"))
        self.types = {v: k for k, v in self._types.items()}

        self._group_annotations()

    def _group_annotations(self):
        groups = self.annotations.groupby("CORRELATION_ID")
        self.groups = groups
        self.group_keys = list(groups.indices.keys())

    def key_to_iso(self, key) -> str:
        return datetime.date(*key).isoformat()

    def __getitem__(self, idx: int):
        key = self.group_keys[idx]
        group = self.groups.get_group(key)
        sample = group[self.concept_name].to_list()  # type: ignore

        sample.append(self.end)
        sample = self._encode_sample(sample, len(self.types))
        return self._transform(sample)

    def __len__(self) -> int:
        return len(self.groups)

    def _transform(self, sample):
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def _encode(self, sample, n: int):
        if self.encode is not None:
            return [self.encode(i, n) for i in sample]
        return sample

    def _encode_sample(self, sample, n: int):
        if self.encode is not None:
            return [self.encode(self._types[i], n) for i in sample]
        return sample

    def _filter(self, column: str, condition):
        filtered = self.annotations[self.annotations[column].map(condition)]  # type: ignore
        if filtered is not None:
            self.annotations = filtered
