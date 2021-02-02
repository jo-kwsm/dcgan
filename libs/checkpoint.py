import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

"""
Copyright (c) 2020 yiskw713
"""


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: Dict[str, nn.Module],
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:

    for k in model.keys():
        save_states = {
            "epoch": epoch,
            "state_dict": model[k].state_dict(),
            "optimizer": optimizer[k].state_dict(),
            "best_loss": best_loss,
        }
        torch.save(
            save_states,
            os.path.join(result_path, "checkpoint_%s.pth" % k),
        )


def resume(
    resume_tmp_path: Dict[str, str],
    model: Dict[str, nn.Module],
    optimizer: Dict[str, optim.Optimizer],
) -> Tuple[int, nn.Module, optim.Optimizer, float]:

    for k in model.keys():
        resume_path = resume_tmp_path % k
        assert os.path.exists(resume_path), "there is no checkpoint at the result folder"

        print("loading checkpoint {}".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)

        begin_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        model[k].load_state_dict(checkpoint["state_dict"])

        optimizer[k].load_state_dict(checkpoint["optimizer"])

    print("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer, best_loss
