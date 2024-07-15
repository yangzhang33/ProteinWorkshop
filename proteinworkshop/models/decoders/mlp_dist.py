"""Linear Decoders"""
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from loguru import logger

from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import ActivationType


class MLP_Pred_Dist(nn.Module):
    def __init__(self, hidden_channels: int, 
                 input: Optional[str] = None,):
        super(MLP_Pred_Dist, self).__init__()
        self.hidden_channels = hidden_channels
        self.linear1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_channels, 1)
        self.input = input

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.squeeze()
        return x
    

