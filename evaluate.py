import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.graph import make_image
from libs.mean_std import get_mean, get_std
from libs.models import get_model
from libs.transformer import ImageTransform

random_seed = 1234


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        train GAN for object detection with Mnist Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    config = get_config(args.config)

    result_path = os.path.dirname(args.config)

    device = get_device(allow_only_gpu=True)

    model = get_model(name=config.model, z_dim=config.z_dim, image_size=config.size)
    for k, v in model.items():
        state_dict = torch.load(os.path.join(result_path, "final_model_%s.prm" % k))

        v.load_state_dict(state_dict)
        v.to(device)
        v.eval()

    train_loader = get_dataloader(
        csv_file=config.train_csv,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=ImageTransform(mean=get_mean(), std=get_std()),
    )

    make_image(
        train_loader,
        model,
        config.model,
        result_path,
        config.z_dim,
        device,
    )

    print("Done")


if __name__ == "__main__":
    main()
