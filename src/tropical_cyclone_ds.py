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


class TropicalCycloneSequence(TropicalCyclone):
    """Tropical Cyclone Dataset adopted for loading sequences."""

    valid_tasks = ["regression", "classification"]

    # based on https://www.nhc.noaa.gov/climo/?text
    class_bins = {
        "tropical_depression": (0, 33),
        "tropical_storm": (34, 63),
        "hurr_1": (64, 82),
        "hurr_2": (83, 95),
        "hurr_3": (96, 112),
        "hurr_4": (113, 136),
        "hurr_5": (137, np.inf),
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        min_wind_speed: float = 0.0,
        task: str = "regression",
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
            task: one of "regression" or "classification"
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

        assert (
            task in self.valid_tasks
        ), f"invalid task '{task}', please choose one of {self.valid_tasks}"
        self.task = task
        self.min_wind_speed = min_wind_speed
        self.seq_len = seq_len
        self.sequence_df = self.construct_sequences()

    def construct_sequences(self) -> list[list[str]]:
        """Construct sequence collection for data loading.

        Returns:
            collection as sequences
        """
        df = pd.read_csv(os.path.join(self.root, f"{self.split}_info.csv"))

        df = df[df["wind_speed"] >= self.min_wind_speed]

        # setup df for possible classification task
        filtered_class_bins = {
            k: v for k, v in self.class_bins.items() if v[1] > self.min_wind_speed
        }
        filtered_class_bins = dict(
            sorted(filtered_class_bins.items(), key=lambda item: item[1][0])
        )

        def assign_class(wind_speed):
            """Assign class index to wind speed."""
            for i, (class_name, (min_speed, max_speed)) in enumerate(
                filtered_class_bins.items()
            ):
                # if wind_speed is within the range of the class, return the class index
                if min_speed <= wind_speed <= max_speed:
                    return i
            return len(filtered_class_bins) - 1

        df["class_index"] = df["wind_speed"].apply(assign_class)

        self.class_to_name = {
            i: class_name for i, class_name in enumerate(filtered_class_bins.keys())
        }

        df["seq_id"] = (
            df["path"]
            .str.split("/", expand=True)[0]
            .str.split("_", expand=True)[7]
            .astype(int)
        )
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
            min_seq_id = df["seq_id"].min()
            max_seq_id = df["seq_id"].max()
            # generate possible subsquences of length k for the group
            subsequences = [
                list(range(i, i + k)) for i in range(min_seq_id, max_seq_id - k + 2)
            ]
            filtered_subsequences: list[list[int]] = [
                subseq for subseq in subsequences if set(subseq).issubset(df["seq_id"])
            ]

            wind_speeds = [
                df.loc[df["seq_id"] == subseq[-1], "wind_speed"].values[0]
                for subseq in filtered_subsequences
                if subseq
            ]

            class_labels = [
                df.loc[df["seq_id"] == subseq[-1], "class_index"].values[0]
                for subseq in filtered_subsequences
                if subseq
            ]

            return {
                "storm_id": df["storm_id"].iloc[0],
                "subsequences": filtered_subsequences,
                "wind_speed": wind_speeds,
                "class_label": class_labels,
            }

        # Group by 'object_id' and find consecutive triplets for each group
        cons_sequences = (
            df.groupby("storm_id").apply(get_subsequences, k=self.seq_len).tolist()
        )
        # dropna the empty sequences
        sequence_df = (
            pd.DataFrame(cons_sequences)
            .explode(["subsequences", "wind_speed", "class_label"])
            .reset_index(drop=True)
            .dropna()
        )

        return sequence_df

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels
        """
        storm_id = self.sequence_df.iloc[index].storm_id
        subsequence = self.sequence_df.iloc[index].subsequences

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

        if self.task == "classification":
            sample["target"] = (
                torch.tensor(int(self.sequence_df.iloc[index].class_label))
                .squeeze()
                .long()
            )
        else:
            sample["target"] = (
                torch.tensor(int(self.sequence_df.iloc[index].wind_speed))
                .float()
                .unsqueeze(-1)
            )

        sample["index"] = index
        sample["storm_id"] = storm_id
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
        return len(self.sequence_df)

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
