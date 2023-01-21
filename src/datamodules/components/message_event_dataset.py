import os
from typing import Any

from src.datamodules.components.dataset_base import DatasetBase


class MessageEventDataset(DatasetBase):
    def __getitem__(self, idx) -> Any:
        file_path = os.path.join(
            self.files_dir, "documents", self.annotations["MESSAGE"].iloc[idx]
        )
        file = open(file_path, "r")
        sample = file

        if self.transformer:
            sample = self.transformer(sample)

        return sample
