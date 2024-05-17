from torchgeo.datasets import DigitalTyphoonAnalysis
from torch import Tensor
from typing import Any

# from torchgeo.datasets import DigitalTyphoonAnalysis
from torchgeo.datamodules import NonGeoDataModule
from torch import Tensor
from typing import Any, Dict

import torch
from torchgeo.datamodules.utils import group_shuffle_split
from sklearn.model_selection import train_test_split
from .dataset import DigitalTyphoonAnalysis


class MyDigitalTyphoonAnalysis(DigitalTyphoonAnalysis):
    def __getitem__(self, index: int):
        sample = super().__getitem__(index)

        # Rename 'image' and 'mask' keys
        sample["input"] = sample.pop("image")
        sample["target"] = sample.pop("label")
        return sample


class MyDigitalTyphoonAnalysisDataModule(NonGeoDataModule):
    valid_split_types = ["time", "typhoon_id"]

    mean = torch.Tensor([296.23836020479655])
    std = torch.Tensor([24.303783180826187])

    def __init__(
        self,
        split_by: str = "time",
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DigitalTyphoonAnalysisDataModule instance.

        Args:
            split_by: Either 'time' or 'typhoon_id', which decides how to split
                the dataset for train, val, test
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DigitalTyphoonAnalysis`.

        """
        super().__init__(MyDigitalTyphoonAnalysis, batch_size, num_workers, **kwargs)

        assert (
            split_by in self.valid_split_types
        ), f"Please choose from {self.valid_split_types}"
        self.split_by = split_by

        # Rename 'image' and 'mask' keys
        self.aug.data_keys = ["input"]

    def split_dataset(self, dataset: MyDigitalTyphoonAnalysis):
        """Split dataset into two parts.

        Args:
            dataset: Dataset to be split into train/test or train/val subsets

        Returns:
            a tuple of the subset datasets
        """
        if self.split_by == "time":
            sequences = list(enumerate(dataset.sample_sequences))

            sorted_sequences = sorted(sequences, key=lambda x: x[1]["seq_id"])
            selected_indices = [x[0] for x in sorted_sequences]

            split_idx = int(len(sorted_sequences) * 0.8)
            train_indices = selected_indices[:split_idx]
            val_indices = selected_indices[split_idx:]

        else:
            sequences = list(enumerate(dataset.sample_sequences))
            train_indices, val_indices = group_shuffle_split(
                [x[1]["id"] for x in sequences], train_size=0.8, random_state=0
            )

        # select train and val sequences and remove enumeration
        train_sequences = [sequences[i][1] for i in train_indices]
        val_sequences = [sequences[i][1] for i in val_indices]

        return train_sequences, val_sequences

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = MyDigitalTyphoonAnalysis(**self.kwargs)

        self.target_mean = torch.tensor(
            self.dataset.aux_df[self.dataset.target[0]].mean()
        )
        self.target_std = torch.tensor(
            self.dataset.aux_df[self.dataset.target[0]].std()
        )

        self.task = self.dataset.task

        train_sequences, test_sequences = self.split_dataset(self.dataset)

        if stage in ["fit", "validate"]:
            # resplit the train indices into train and val
            self.dataset.sample_sequences = train_sequences
            train_sequences, val_sequences = self.split_dataset(self.dataset)

            # create training dataset
            self.train_dataset = MyDigitalTyphoonAnalysis(**self.kwargs)
            self.train_dataset.sample_sequences = train_sequences

            # create validation dataseqt
            self.val_dataset = MyDigitalTyphoonAnalysis(**self.kwargs)
            self.val_dataset.sample_sequences = val_sequences

        if stage in ["test"]:
            self.test_dataset = MyDigitalTyphoonAnalysis(**self.kwargs)
            self.test_dataset.sample_sequences = test_sequences

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.target_mean.device != batch["input"].device:
            if self.target_mean.device.type == "cpu":
                self.target_mean = self.target_mean.to(batch["input"].device)
                self.target_std = self.target_std.to(batch["input"].device)
            elif self.target_mean.device.type == "cuda":
                batch["input"] = batch["input"].to(self.target_mean.device)
                batch["target"] = batch["target"].to(self.target_mean.device)

        new_batch = {
            "input": self.aug({"input": batch["input"].float()})["input"],
            "target": (batch["target"].float() - self.target_mean) / self.target_std,
        }
        return new_batch
