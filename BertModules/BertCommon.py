import torch as t
import torch.nn as nn
from BertModules.BertConfig import BertConfig
from BertModules.BertBlock import BertBlock
from BertModules.Embedding import Embedding
from BertModules.LayerNorm import LayerNorm
from einops import rearrange
from typing import Optional

class BertCommon(nn.Module):
    token_embedding: Embedding
    pos_embedding: Embedding
    token_type_embedding: Embedding
    layer_norm: LayerNorm
    blocks: nn.ModuleList

    def __init__(self, config: BertConfig):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embedding = Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = LayerNorm(config.hidden_size, eps = config.layer_norm_epsilon)
        self.blocks = nn.ModuleList([BertBlock(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def _make_additive_attention_mask(
        self,
        one_zero_attention_mask: t.Tensor,
        big_negative_number: float = -10000
    ) -> t.Tensor:

        """
        one_zero_attention_mask: shape (batch, seq). Contains 1 if this is a valid token and 0 if it is a padding token.
        big_negative_number: Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

        Out: shape (batch, heads, seq, seq). Contains 0 if attention is allowed, and big_negative_number if it is not allowed.
        """
        return rearrange((1 - one_zero_attention_mask), 'b s -> b 1 1 s') * big_negative_number

    
    def forward(
        self,
        input_ids: t.Tensor,
        token_type_ids: Optional[t.Tensor] = None,
        one_zero_attention_mask: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """
        input_ids: (batch, seq) - the token ids
        token_type_ids: (batch, seq) - only used for next sentence prediction.
        one_zero_attention_mask: (batch, seq) - only used in training. See make_additive_attention_mask.
        """
        token_embeddings = self.token_embedding(input_ids)
        pos_embeddings = self.pos_embedding(t.arange(input_ids.shape[1], device=input_ids.device))
        
        if not token_type_ids:
            token_type_ids = t.zeros_like(input_ids)

        token_type_embeddings = self.token_type_embedding(token_type_ids)
        x = token_embeddings + pos_embeddings + token_type_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)

        if one_zero_attention_mask:
            additive_attention_mask = self._make_additive_attention_mask(one_zero_attention_mask)
        else:
            additive_attention_mask = None

        for block in self.blocks:
            x = block(x, additive_attention_mask)
        return x





        
        
                

