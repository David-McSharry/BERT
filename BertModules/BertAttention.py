import torch as t
import torch.nn as nn
from BertModules.LayerNorm import LayerNorm
from BertModules.BertConfig import BertConfig
from BertModules.BertSelfAttention import BertSelfAttention
from typing import Optional

class BertAttention(nn.Module):
    self_attn: BertSelfAttention
    layer_norm: LayerNorm

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attn = BertSelfAttention(config)
        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        original_x = x
        x = self.self_attn(x, additive_attention_mask)
        x = self.dropout(x)
        x = self.layer_norm(x + original_x)
        return x
