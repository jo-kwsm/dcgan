import json, os, sys
from typing import Any, Optional

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = ["get_dataloader"]

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

    data = GANImgDataset(
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

        return img_transformed


def data_test():
    from transformer import ImageTransform
    mean = (0.5,)
    std = (0.5,)

    # DataLoaderを作成
    batch_size = 64
    
    train_dataloader = get_dataloader(
        csv_file="csv/data.csv",
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        transform=ImageTransform(mean, std),
    )

    # 動作の確認
    batch_iterator = iter(train_dataloader)  # イテレータに変換
    imges = next(batch_iterator)  # 1番目の要素を取り出す
    print(imges.size())  # torch.Size([64, 1, 64, 64])


if __name__ == "__main__":
    data_test()
