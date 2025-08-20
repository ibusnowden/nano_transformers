# The base Transformer class
import torch.nn as nn
import os, sys
sys.path.append('..')
from layers.attention import MultiHeadAttention
from layers.feed_forward import FeedForward
from layers.normalization import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads,  d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x, mask))
        x = x + self.ffn(self.norm2(x))
        return x

