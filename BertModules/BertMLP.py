import torch as t
import torch.nn as nn
import torch.nn.functional as F
from BertModules.LayerNorm import LayerNorm
from BertModules.BertConfig import BertConfig

class BertMLP(nn.Module):
    first_linear: nn.Linear
    second_linear: nn.Linear
    layer_norm: LayerNorm

    def __init__(self, config: BertConfig):
        super().__init__()
        self.first_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.second_linear = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        original_x = x
        x = self.first_linear(x)
        x = F.gelu(x)
        x = self.second_linear(x)
        x = self.dropout(x)
        x = self.layer_norm(x + original_x)
        return x
