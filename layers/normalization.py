# Contains Layer Normalization
import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        self.gamma = nn.Parameters(torch.ones(d_model))
        self.beta = nn.Parameters(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta