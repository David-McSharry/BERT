from torch import nn
from einops import rearrange
from BertModules.BertConfig import BertConfig
import torch as t
import torch.nn.functional as F
from typing import Optional

class BertSelfAttention(nn.Module):
    project_query: nn.Linear
    project_key: nn.Linear
    project_value: nn.Linear
    project_output: nn.Linear

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.head_size = config.head_size
        self.num_heads = config.num_heads
        self.project_query = nn.Linear(config.hidden_size, config.num_heads * config.head_size)
        self.project_key = nn.Linear(config.hidden_size, config.num_heads * config.head_size)
        self.project_value = nn.Linear(config.hidden_size, config.num_heads * config.head_size)
        self.project_output = nn.Linear(config.num_heads * config.head_size, config.hidden_size)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)
        Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """
        # output QK^T/sqrt(d_k)        
               
        Q = rearrange(self.project_query(x), 'bs q (nh hs) -> bs nh q hs', nh = self.num_heads)
        K = rearrange(self.project_key(x), 'bs q (nh hs) -> bs nh hs q', nh = self.num_heads)
        pre_softmax_attention_scores = t.einsum(
            'ab ij, ab jk -> ab ik',
            Q,K
        )
        return pre_softmax_attention_scores / self.head_size ** 0.5


    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        """
        additive_attention_mask: shape (batch, head=1, seq_q=1, seq_k) - used in training to prevent copying data from padding tokens. Contains 0 for a real input token and a large negative number for a padding token. If provided, add this to the attention pattern (pre softmax).

        Return: (batch, seq, hidden_size)
        """
        pre_softmax_attention_scores = self.attention_pattern_pre_softmax(x)
        if additive_attention_mask is not None:
            pre_softmax_attention_scores += additive_attention_mask
        attention_scores = F.softmax(pre_softmax_attention_scores, dim=-1)

        V = rearrange(
            self.project_value(x),
            'bs q (nh hs) -> bs nh q hs',
            nh = self.num_heads
        )

        attention_weighted_value = t.einsum(
            'ab ij, ab jk -> ab ik',
            attention_scores, V
        )

        attention_weighted_value = rearrange(
            attention_weighted_value,
            'bs nh q hs -> bs q (nh hs)'
        )
        
        return self.project_output(attention_weighted_value)
