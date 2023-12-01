"""Tropical Cyclone Triplet Dataset."""

import os
from functools import lru_cache
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchgeo.datasets import TropicalCyclone


class TropicalCycloneTriplet(TropicalCyclone):
    """Tropical Cyclone Dataset adopted for loading triplets."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Tropical Cyclone Wind Estimation Competition Dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        super().__init__(root, split, None, download, api_key, checksum)
        self.triplet_df = self.construct_triplets(self.collection)

    def construct_triplets(self, collection: list[dict[str, str]]) -> list[list[str]]:
        """Construct triplet collection for data loading.

        Args:
            collection: dataset collection for the spli

        Returns:
            collection as triplets
        """
        df = pd.DataFrame(collection)
        df["storm_id"] = (
            df["href"].str.split("/", expand=True)[0].str.split("_", expand=True)[6]
        )
        df["seq_id"] = (
            df["href"]
            .str.split("/", expand=True)[0]
            .str.split("_", expand=True)[7]
            .astype(int)
        )

        def find_consecutive_triplets(group):
            triplets = []
            for i in range(len(group) - 2):
                if (
                    group.iloc[i + 1, 3] == group.iloc[i, 3] + 1
                    and group.iloc[i + 2, 3] == group.iloc[i, 3] + 2
                ):
                    triplets.append(
                        (group.iloc[i, 3], group.iloc[i + 1, 3], group.iloc[i + 2, 3])
                    )
            return {"storm_id": group.iloc[0, 2], "triplet": triplets}

        # Group by 'object_id' and find consecutive triplets for each group
        cons_trip = df.groupby("storm_id").apply(find_consecutive_triplets).tolist()
        triplet_df = pd.DataFrame(cons_trip).explode("triplet").reset_index(drop=True)
        return triplet_df

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field import pdb
        """
        storm_id = self.triplet_df.iloc[index].storm_id
        triplet = self.triplet_df.iloc[index].triplet

        imgs: list[Tensor] = []
        for time_idx in triplet:
            directory = os.path.join(
                self.root,
                "_".join([self.collection_id, self.split, "{0}"]),
                "_".join(
                    [
                        self.collection_id,
                        self.split,
                        "{0}",
                        storm_id,
                        str(time_idx).zfill(3),
                    ]
                ),
            )
            imgs.append(self._load_image(directory))

        sample: dict[str, Any] = {"input": torch.stack(imgs, 0)}
        sample.update(self._load_features(directory))
        sample["target"] = sample["label"].unsqueeze(-1)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @lru_cache
    def _load_image(self, directory: str) -> Tensor:
        """Load a single image.

        Args:
            directory: directory containing image

        Returns:
            the image
        """
        filename = os.path.join(directory.format("source"), "image.jpg")
        with Image.open(filename) as img:
            if img.height != self.size or img.width != self.size:
                # Moved in PIL 9.1.0
                try:
                    resample = Image.Resampling.BILINEAR
                except AttributeError:
                    resample = Image.BILINEAR
                img = img.resize(size=(self.size, self.size), resample=resample)
            array: "np.typing.NDArray[np.int_]" = np.array(img)

            tensor = torch.from_numpy(array).float()
            # investigate why not all images have the same shape
            if tensor.dim() != 2:
                tensor = tensor[:, :, 0]
            return tensor

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.triplet_df)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image, label = sample["inputs"] / 255, sample["wind_speed"]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0))
        ax.axis("off")

        if show_titles:
            title = f"Label: {label}"
            if showing_predictions:
                title += f"\nPrediction: {prediction}"
            ax.set_title(title, fontsize=20)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
