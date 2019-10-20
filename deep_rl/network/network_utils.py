#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import *
from typing import Optional

Tensor = Optional[torch.Tensor]


class BaseNet:
    def __init__(self):
        pass

    def forward(self, x) -> Tensor:
        pass

    def feature(self, obs) -> Tensor:
        pass

    def actor(self, phi) -> Tensor:
        pass

    def critic(self, phi, a) -> Tensor:
        pass

    def q(self, obs, a) -> Tensor:
        pass


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
