"""Tropical Cyclone Wind Speed Estimation."""

from typing import Any, Dict

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import Subset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split
from torchgeo.transforms import AugmentationSequential

from tropical_cyclone_ds import TropicalCycloneTriplet


class TropicalCycloneTripletDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.
    """

    input_mean = torch.Tensor([0.28154722, 0.28071895, 0.27990073])
    input_std = torch.Tensor([0.23435517, 0.23392765, 0.23351675])
    target_mean = torch.Tensor([50.54925])
    target_std = torch.Tensor([26.836512])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new TropicalCycloneDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~tropical_cyclone_uq.datasets.TropicalCyclone`.
        """
        super().__init__(TropicalCycloneTriplet, batch_size, num_workers, **kwargs)

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
            self.dataset = TropicalCycloneTriplet(split="train", **self.kwargs)
            train_indices, val_indices = group_shuffle_split(
                self.dataset.triplet_df.storm_id, test_size=0.15, random_state=0
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
        if stage in ["test"]:
            self.test_dataset = TropicalCycloneTriplet(split="test", **self.kwargs)

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
            "input": self.aug({"image": batch["input"].float()})["image"],
            "target": (batch["target"].float() - self.target_mean) / self.target_std,
        }
        return new_batch
