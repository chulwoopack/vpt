#!/usr/bin/env python3
"""
Modified from: mlp.py
"""
import math
import torch

from torch import nn
from typing import List, Type

from ..utils import logging
logger = logging.get_logger("visual_prompt")

class TRANSCONV(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,

    ):
        super(TRANSCONV, self).__init__()
        self.transpose = nn.ConvTranspose2d(input_dim,out_dim,16,16)
        self.activate = nn.GELU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.transpose(x)
        x = self.activate(x)
        x = self.softmax(x) # [b,class,h,w]

        return x
