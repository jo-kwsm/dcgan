import os
from typing import Dict

import torch
import torch.nn as nn

from .modules import Generator, Discriminator

__all__ = ["get_model"]


def get_model(pretrained: bool = True) -> Dict[str, nn.Module]:
    G = Generator()
    D = Discriminator()

    G.apply(weights_init)
    D.apply(weights_init)

    model = {
        "G": G,
        "D": D,
    }

    return model


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
