
import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        #self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output
    
# Multi-Head Attention



# Grouped-Query-Attention