# GPT-specific model definition (decoder-only)
import torch
import os, sys
sys.path.append('..')
import torch.nn as nn
from models.base import TransformerBlock
from layers.embeddings import TokenEmbedding, PositionalEncoding



class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, max_len=512):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff)] for _ in range(num_layers))
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.to_logits(x)
