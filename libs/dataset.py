import json, os, sys
from typing import Any, Optional

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

"""
Copyright (c) 2019 Yutaro Ogawa
"""


def get_dataloader(
    csv_file: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:

    data = COCODataset(
        csv_file,
        transform=transform,
    )
    
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


class GANImgDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        assert os.path.exists(csv_file)

        csv_path = os.path.join(csv_file)
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:
        img = self.pull_item(idx)
        return img

    def pull_item(self, idx: int) -> Any:
        image_file_path = self.df.iloc[idx]["image_path"]

        img = Image.open(image_file_path)
        img_transformed = self.transform(img)

        return img


def data_test():
    # TODO テストを追加


if __name__ == "__main__":
    data_test()
