import math
from typing import Any, Dict, Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

from src.datamodules.components.message_event_dataset import MessageEventDataset

ANNOTATION_FILE_NAME: str = "annotations.csv"
SERVICE_IDX: str = "SERVICE"


class MessageEventModule(LightningDataModule):
    def __init__(
        self,
        dataset: MessageEventDataset,
        train_val_test_split: Tuple[float, float, float] = (0.75, 0.05, 0.20),
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[MessageEventDataset] = None
        self.data_val: Optional[MessageEventDataset] = None
        self.data_test: Optional[MessageEventDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train and not self.data_test:
            dataset: MessageEventDataset = self.dataset

            train_len = math.ceil(len(dataset) * self.train_val_test_split[0])
            test_len = math.ceil(len(dataset) * self.train_val_test_split[2])
            val_len = len(dataset) - (train_len + test_len)
            self.data_train, self.data_test, self.data_val = random_split(  # type: ignore
                dataset=dataset,
                lengths=[
                    train_len,
                    test_len,
                    val_len,
                ],
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
