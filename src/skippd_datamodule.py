from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from torchgeo.datamodules import SKIPPDDataModule
from torchgeo.datasets import SKIPPD


class MySKIPPDDataModule(SKIPPDDataModule):
    """LightningDataModule implementation for the SKIPP'D dataset.

    Adds calibration dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SKIPPDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SKIPPD`.
        """
        super().__init__(batch_size, num_workers, val_split_pct, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = SKIPPD(split="trainval", **self.kwargs)

            # take 10 % of val_split data as calib_split
            calib_split_pct, val_split_pct = (
                0.1 * self.val_split_pct,
                0.89 * self.val_split_pct,
            )
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset, self.calibration_dataset = (
                random_split(
                    self.dataset,
                    [1 - self.val_split_pct, val_split_pct, calib_split_pct],
                    generator,
                )
            )
        if stage in ["test"]:
            self.test_dataset = SKIPPD(split="test", **self.kwargs)

    def calibration_dataloader(self) -> torch.utils.data.DataLoader:
        """Return a dataloader for the calibration dataset."""
        return DataLoader(
            self.calibration_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )
