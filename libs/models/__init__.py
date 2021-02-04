import os
from typing import Dict

import torch
import torch.nn as nn

from .DCGAN import DCGenerator, DCDiscriminator
from .SAGAN import SAGenerator, SADiscriminator

__all__ = ["get_model"]

model_names = ["DCGAN", "SAGAN"]


def get_model(name: str, z_dim: int = 20, image_size: int = 64) -> Dict[str, nn.Module]:
    name = name.upper()
    if name not in model_names:
        raise ValueError(
            """There is no model appropriate to your choice.
            You have to choose DCGAN or SAGAN as a model.
        """
        )

    print("{} will be used as a model.".format(name))
    
    if name == "DCGAN":
        G = DCGenerator(z_dim, image_size)
        D = DCDiscriminator(z_dim, image_size)
    elif name == "SAGAN":
        G = SAGenerator(z_dim, image_size)
        D = SADiscriminator(z_dim, image_size)

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
