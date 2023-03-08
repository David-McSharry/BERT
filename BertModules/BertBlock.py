import torch as t
import torch.nn as nn
from BertModules.BertConfig import BertConfig
from BertModules.BertAttention import BertAttention
from BertModules.BertMLP import BertMLP
from typing import Optional

class BertBlock(nn.Module):
    attention: BertAttention
    mlp: BertMLP

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.mlp = BertMLP(config)


    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        if additive_attention_mask is not None:
            x = self.attention(x, additive_attention_mask)
        x = self.attention(x)
        x = self.mlp(x)
        return x
