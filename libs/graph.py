import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
                    make graphs for training logs.
                    """
    )
    parser.add_argument("log", type=str, default=None, help="path of a log file")

    return parser.parse_args()


def make_line(data_name:str, data:pd.DataFrame, save_dir:str) -> None:
    plt.figure()
    plt.plot(data["train_" + data_name], label="sum")
    plt.plot(data["train_d_" + data_name], label="discriminator")
    plt.plot(data["train_g_" + data_name], label="generator")
    plt.xlabel("epoch")
    plt.ylabel(data_name)
    plt.legend()
    save_path = os.path.join(save_dir, data_name + ".png")
    plt.savefig(save_path)


def make_graphs(log_path:str) -> None:
    logs = pd.read_csv(log_path)
    save_dir = os.path.dirname(log_path)
    make_line("loss", logs, save_dir)
    # TODO 評価指標


def make_image(
    loader: DataLoader,
    model: nn.Module,
    model_name: str,
    save_dir: str,
    z_dim: int,
    device: str,
    batch_size: int = 8,
    ) -> None:

    fixed_z = torch.randn(batch_size, z_dim)
    fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

    model["G"].eval()
    if model_name == "DCGAN":
        fake_images = model["G"](fixed_z.to(device))
    elif model_name == "SAGAN":
        fake_images, am1, am2 = model["G"](fixed_z.to(device))


    batch_iterator = iter(loader)
    imges = next(batch_iterator)

    fig = plt.figure(figsize=(15, 6))
    for i in range(0, 5):
        plt.subplot(2, 5, i+1)
        plt.imshow(imges[i][0].cpu().detach().numpy(), 'gray')

        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

    plt.savefig(os.path.join(save_dir, "result.png"))
    plt.close()

    if model_name == "SAGAN":
        fig = plt.figure(figsize=(15, 6))
        for i in range(0, 5):
            plt.subplot(2, 5, i+1)
            plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')

            plt.subplot(2, 5, 5+i+1)
            am = am1[i].view(16, 16, 16, 16)
            am = am[7][7]  # 中央に着目
            plt.imshow(am.cpu().detach().numpy(), 'Reds')

        plt.savefig(os.path.join(save_dir, "attention.png"))


def main() -> None:
    args = get_arguments()
    make_graphs(args.log)


if __name__ == "__main__":
    main()
