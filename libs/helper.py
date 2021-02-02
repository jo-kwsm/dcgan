import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter

__all__ = ["train", "evaluate"]


def do_one_iteration(
    sample: Dict[str, Any],
    model: Dict[str, nn.Module],
    criterion: Any,
    z_dim: int,
    device: str,
    iter_type: str,
    optimizer: Dict[str, Optional[optim.Optimizer]] = None,
) -> Tuple[int, float]:

    if iter_type not in ["train", "evaluate"]:
        raise ValueError("iter_type must be either 'train' or 'evaluate'.")

    if iter_type == "train" and optimizer is None:
        raise ValueError("optimizer must be set during training.")

    imgs = sample["img"].to(device)
    batch_size = imgs.shape[0]

    label_real = torch.full((batch_size,), 1).to(device)
    label_fake = torch.full((batch_size,), 0).to(device)

    d_input_z = torch.randn(batch_size, z_dim).to(device)
    d_input_z = d_input_z.view(d_input_z.size(0), d_input_z.size(1), 1, 1)

    d_fake_imgs = model["G"](d_input_z)
    d_out_fake = model["D"](d_fake_imgs)
    d_out_real = model["D"](imgs)

    loss_real = criterion(d_out_real.view(-1), label_real)
    loss_fake = criterion(d_out_fake.view(-1), label_fake)
    d_loss = loss_real + loss_fake

    if iter_type == "train" and optimizer is not None:
        optimizer["G"].zero_grad()
        optimizer["D"].zero_grad()
        d_loss.backward()
        optimizer["D"].step()

    g_input_z = torch.randn(batch_size, z_dim).to(device)
    g_input_z = g_input_z.view(g_input_z.size(0), g_input_z.size(1), 1, 1)

    g_fake_imgs = model["G"](g_input_z)
    g_out_fake = model["D"](g_fake_imgs)

    g_loss = criterion(g_out_fake.view(-1), label_real)

    if iter_type == "train" and optimizer is not None:
        optimizer["G"].zero_grad()
        optimizer["D"].zero_grad()
        g_loss.backward()
        optimizer["G"].step()

    loss = d_loss + g_loss

    return batch_size, loss.item()


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    z_dim: int,
    device: str,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    # TODO semantic segmentation の評価指標に変更
    # top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses], # TODO semantic segmentation の評価指標追加
        prefix="Epoch: [{}]".format(epoch),
    )

    for v in model.values():
        v.train()

    end=time.time()
    for i, imgs in enumerate(loader):
        sample = {
            "img": imgs,
        }

        data_time.update(time.time() - end)

        batch_size, loss = do_one_iteration(
            sample,
            model,
            criterion,
            z_dim,
            device,
            "train",
            optimizer,
        )

        losses.update(loss, batch_size)

        batch_time.update(time.time()-end)
        end = time.time()

        if i!=0 and i % interval_of_progress == 0:
            progress.display(i)

    return losses.get_average()


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    z_dim: int,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    for v in model.values():
        v.eval()

    with torch.no_grad():
        for img in loader:
            sample = {
            "img": img,
        }

            batch_size, loss, = do_one_iteration(
                sample,
                model,
                criterion,
                z_dim,
                device,
                "evaluate",
            )

            losses.update(loss, batch_size)

    return losses.get_average()
