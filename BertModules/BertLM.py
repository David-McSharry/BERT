import torch as t
from torch import nn
from torch.nn import functional as F
from BertModules.BertConfig import BertConfig
from BertModules.BertCommon import BertCommon
from BertModules.LayerNorm import LayerNorm
from typing import Optional

class BertLanguageModel(nn.Module):
    common: BertCommon
    lm_linear: nn.Linear
    lm_layer_norm: LayerNorm
    unembed_bias: nn.Parameter

    def __init__(self, config: BertConfig):
        super().__init__()
        self.common = BertCommon(config)
        self.lm_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.unembed_bias = nn.Parameter(t.zeros(config.vocab_size))

    def forward(
        self,
        input_ids: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
        one_zero_attention_mask: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """
        x = self.common(input_ids, token_type_ids, one_zero_attention_mask)
        x = self.lm_linear(x)
        x = F.gelu(x)
        x = self.lm_layer_norm(x)
        # unembed with original embedding matrix
        x = x @ self.common.token_embedding.weight.t() + self.unembed_bias
        return x