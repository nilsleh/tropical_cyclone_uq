"""Tropical Cyclone Wind Speed Estimation."""

from typing import Any, Dict

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split
from sklearn.model_selection import train_test_split
from torchgeo.transforms import AugmentationSequential

from tropical_cyclone_ds import TropicalCycloneSequence


class TropicalCycloneSequenceDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.
    """

    input_mean = torch.Tensor([0.28154722, 0.28071895, 0.27990073])
    input_std = torch.Tensor([0.23435517, 0.23392765, 0.23351675])
    # target_mean = torch.Tensor([50.54925])
    # target_std = torch.Tensor([26.836512])

    valid_tasks = ["regression", "classification"]

    def __init__(
        self, task: str = "regression", batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new TropicalCycloneDataModule instance.

        Args:
            task: One of "regression" or "classification"
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~tropical_cyclone_uq.datasets.TropicalCyclone`.
        """
        super().__init__(TropicalCycloneSequence, batch_size, num_workers, **kwargs)

        assert task in self.valid_tasks, f"invalid task '{task}', please choose one of {self.valid_tasks}"
        self.task = task

        self.dataset = TropicalCycloneSequence(split="train", **self.kwargs)
        # mean and std can change based on setup because min wind speed is a variable
        self.target_mean = torch.Tensor([self.dataset.target_mean])
        self.target_std = torch.Tensor([self.dataset.target_std])
        
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(mean=self.input_mean, std=self.input_std),
            K.Resize(224),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=(90, 91), p=0.5),
            K.RandomRotation(degrees=(270, 271), p=0.5),
            data_keys=["image"],
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Normalize(mean=self.input_mean, std=self.input_std),
            K.Resize(224),
            data_keys=["image"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = TropicalCycloneSequence(split="train", task=self.task, **self.kwargs)
            train_indices, val_indices = group_shuffle_split(
                self.dataset.sequence_df.storm_id, test_size=0.20, random_state=0
            )

            validation_indices, calibration_indices = train_test_split(val_indices, test_size=0.20, random_state=0)

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, validation_indices)
            self.calibration_dataset = Subset(self.dataset, calibration_indices)
        if stage in ["test"]:
            self.test_dataset = TropicalCycloneSequence(split="test", task=self.task, **self.kwargs)

    def calibration_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a dataloader for the calibration dataset."""
        return DataLoader(self.calibration_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=False)

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

        if self.task == "regression":
            new_batch = {
                "input": self.aug({"image": batch["input"].float()})["image"],
                "target": (batch["target"].float() - self.target_mean) / self.target_std,
            }
        else:
            new_batch = {
                "input": self.aug({"image": batch["input"].float()})["image"],
                "target": batch["target"].long(),
            }
        return new_batch
