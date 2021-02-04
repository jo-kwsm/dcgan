import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from libs.checkpoint import resume, save_checkpoint
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.graph import make_graphs
from libs.helper import train
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
    experiment_name = os.path.basename(result_path)

    if os.path.exists(os.path.join(result_path, "final_model_G.prm")):
        print("Already done.")
        return

    device = get_device(allow_only_gpu=True)

    train_loader = get_dataloader(
        csv_file=config.train_csv,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=ImageTransform(mean=get_mean(), std=get_std()),
    )

    model = get_model(config.model, z_dim=config.z_dim, image_size=config.size)
    for v in model.values():
        v.to(device)

    g_optimizer = torch.optim.Adam(
        model["G"].parameters(),
        config.g_lr,
        [config.beta1, config.beta2],
    )
    d_optimizer = torch.optim.Adam(
        model["D"].parameters(),
        config.d_lr,
        [config.beta1, config.beta2],
    )
    optimizer = {
        "G": g_optimizer,
        "D": d_optimizer,
    }

    begin_epoch = 0
    best_loss = float("inf")
    # TODO 評価指標の検討
    log = pd.DataFrame(
        columns=[
            "epoch",
            "d_lr",
            "g_lr",
            "train_time[sec]",
            "train_loss",
            "train_d_loss",
            "train_g_loss",
        ]
    )

    if args.resume:
        resume_path = os.path.join(result_path, "checkpoint_%s.pth")
        begin_epoch, model, optimizer, best_loss = resume(resume_path, model, optimizer)

        log_path = os.path.join(result_path, "log.csv")
        assert os.path.exists(log_path), "there is no checkpoint at the result folder"
        log = pd.read_csv(log_path)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    print("---------- Start training ----------")

    for epoch in range(begin_epoch, config.max_epoch):
        start = time.time()
        train_d_loss, train_g_loss,  = train(
            train_loader,
            model,
            config.model,
            criterion,
            optimizer,
            epoch,
            config.z_dim,
            device,
            interval_of_progress=1,
        )
        train_time = int(time.time() - start)

        if best_loss > train_d_loss + train_g_loss:
            best_loss = train_d_loss + train_g_loss
            for k in model.keys():
                torch.save(
                    model[k].state_dict(),
                    os.path.join(result_path, "best_model_%s.prm" % k),
                )

        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        tmp = pd.Series(
            [
                epoch,
                optimizer["D"].param_groups[0]["lr"],
                optimizer["G"].param_groups[0]["lr"],
                train_time,
                train_d_loss + train_g_loss,
                train_d_loss,
                train_g_loss,
            ],
            index=log.columns,
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"), index=False)
        make_graphs(os.path.join(result_path, "log.csv"))

        print(
            "epoch: {}\tepoch time[sec]: {}\tD_lr: {}\tG_lr: {}\ttrain loss: {:.4f}\ttrain d_loss: {:.4f}\ttrain g_loss: {:.4f}".format(
                epoch,
                train_time,
                optimizer["D"].param_groups[0]["lr"],
                optimizer["G"].param_groups[0]["lr"],
                train_d_loss + train_g_loss,
                train_d_loss,
                train_g_loss,
            )
        )

    for k in model.keys():
        torch.save(
            model[k].state_dict(),
            os.path.join(result_path, "final_model_%s.prm" % k),
        )

    for k in model.keys():
        os.remove(os.path.join(result_path, "checkpoint_%s.pth" % k))

    print("Done")


if __name__ == "__main__":
    main()
