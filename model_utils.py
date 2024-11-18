import math

import numpy as np

import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        
        nn.init.orthogonal_(layer.weight.data, gain)
        
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0.0)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer.bias, "data"):
            torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class BetaHead(nn.Module):
    def __init__(self, in_features, action_size):
        super(BetaHead, self).__init__()
        
        self.fcc_c0 = nn.Linear(in_features, action_size)
        nn.init.orthogonal_(self.fcc_c0.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c0.bias)
        
        self.fcc_c1 = nn.Linear(in_features, action_size)
        nn.init.orthogonal_(self.fcc_c1.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c1.bias)
    
    def forward(self, x):
        c0 = torch.nn.functional.softplus(self.fcc_c0(x)) + 1.
        c1 = torch.nn.functional.softplus(self.fcc_c1(x)) + 1.
        return torch.distributions.Independent(
            torch.distributions.Beta(c1, c0), 1
        )


class LayerNormGELU(nn.Module):
    def __init__(self, dim):
        super(LayerNormGELU, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.act = NewGELU()
    
    def forward(self, x):
        return self.act(self.ln(x))
