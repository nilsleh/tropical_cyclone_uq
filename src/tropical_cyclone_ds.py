"""Tropical Cyclone Triplet Dataset."""

import os
import json
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
        min_wind_speed: float = 0.0,
        seq_len: int = 3,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Tropical Cyclone Wind Estimation Competition Dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            min_wind_speed: minimum wind speed to include in dataset
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
        self.min_wind_speed = min_wind_speed
        self.seq_len = seq_len

        self.triplet_df = self.construct_triplets(self.collection)

    def construct_triplets(self, collection: list[dict[str, str]]) -> list[list[str]]:
        """Construct triplet collection for data loading.

        Args:
            collection: dataset collection for the spli

        Returns:
            collection as triplets
        """
        df = pd.read_csv(os.path.join(self.root, f"{self.split}_info.csv"))
        df_two = pd.DataFrame(collection)
        df["seq_id"] = (
            df["path"]
            .str.split("/", expand=True)[0]
            .str.split("_", expand=True)[7]
            .astype(int)
        )

        df = df[df["wind_speed"] >= self.min_wind_speed]

        self.target_mean = df["wind_speed"].mean()
        self.target_std = df["wind_speed"].std()

        def get_subsequences(df: pd.DataFrame, k: int) -> list[dict[str, list[int]]]:
            """Generate all possible subsequences of length k for a given group.

            Args:
                df: grouped dataframe of a single typhoon
                k: length of the subsequences to generate

            Returns:
                list of all possible subsequences of length k for a given typhoon id
            """
            # generate possible subsquences of length k for each group
            subsequences = [
                list(range(i, i + k))
                for i in range(len(df) - k + 1)
            ]
            filtered_subsequences = [
                subseq
                for subseq in subsequences
                if set(subseq).issubset(df["seq_id"])
            ]
            return {
                "storm_id": df["storm_id"].iloc[0],
                "subsequences": filtered_subsequences,
            }

        # Group by 'object_id' and find consecutive triplets for each group
        cons_trip = df.groupby("storm_id").apply(get_subsequences, k=self.seq_len).tolist()
        triplet_df = pd.DataFrame(cons_trip).explode("subsequences")
        return triplet_df

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field import pdb
        """
        storm_id = self.triplet_df.iloc[index].storm_id
        subsequence = self.triplet_df.iloc[index].subsequences

        imgs: list[Tensor] = []
        for time_idx in subsequence:
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

        # already stored under "target"
        del sample["label"]
        del sample["wind_speed"]

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

ds = TropicalCycloneTriplet(root="/p/project/hai_uqmethodbox/data/tropical_cyclone", split="train", min_wind_speed=0.0, seq_len=3)