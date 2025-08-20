# Contains the simple FFN block
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.ffn = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.ffn(self.w_1(x)))
    
# considr adding ffn with silu for other moe model like qween/mistral