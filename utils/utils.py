from transformers.models.bert.modeling_bert import BertForMaskedLM
import transformers
from typing import cast
import torch as t
from BertModules.BertConfig import BertConfig
from BertModules.BertLM import BertLanguageModel
from typing import List


def load_pretrained_bert() -> BertForMaskedLM:
    """Load the HuggingFace BERT.

    Supresses the spurious warning about some weights not being used.
    """
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    return cast(BertForMaskedLM, bert)


def load_pretrained_weights(config: BertConfig) -> BertLanguageModel:
    hf_bert = load_pretrained_bert()
    my_bert = BertLanguageModel(config)
    # print(hf_bert)
    # print(my_bert)
    
    def _copy(ours, src):
        ours.weight.detach().copy_(src.weight)
        if getattr(ours, "bias", None) is not None:
            ours.bias.detach().copy_(src.bias)

    # init the my_bert weights as Nan
    for p in my_bert.parameters():
        p = p.detach().fill_(float("nan"))
    
    _copy(my_bert.lm_linear, hf_bert.cls.predictions.transform.dense)  
    _copy(my_bert.lm_layer_norm, hf_bert.cls.predictions.transform.LayerNorm)
    my_bert.unembed_bias.detach().copy_(hf_bert.cls.predictions.decoder.bias)
    _copy(my_bert.common.token_embedding, hf_bert.bert.embeddings.word_embeddings)
    _copy(my_bert.common.pos_embedding, hf_bert.bert.embeddings.position_embeddings)
    _copy(my_bert.common.token_type_embedding, hf_bert.bert.embeddings.token_type_embeddings)
    _copy(my_bert.common.layer_norm, hf_bert.bert.embeddings.LayerNorm)

    for i, block in enumerate(my_bert.common.blocks):
        _copy(block.mlp.first_linear, hf_bert.bert.encoder.layer[i].intermediate.dense)
        _copy(block.mlp.second_linear, hf_bert.bert.encoder.layer[i].output.dense)
        _copy(block.mlp.layer_norm, hf_bert.bert.encoder.layer[i].output.LayerNorm)
        _copy(block.attention.layer_norm, hf_bert.bert.encoder.layer[i].attention.output.LayerNorm)
        _copy(block.attention.self_attn.project_query, hf_bert.bert.encoder.layer[i].attention.self.query)
        _copy(block.attention.self_attn.project_key, hf_bert.bert.encoder.layer[i].attention.self.key)
        _copy(block.attention.self_attn.project_value, hf_bert.bert.encoder.layer[i].attention.self.value)
        _copy(block.attention.self_attn.project_output, hf_bert.bert.encoder.layer[i].attention.output.dense)


    for p in my_bert.parameters():
        assert not t.isnan(p).any(), f"Parameter {p} is NaN"
    
    return my_bert


def predict(model: BertLanguageModel, tokenizer, text: str, k=15) -> List[List[str]]:
    """
    Return a list of k strings for each [MASK] in the input.
    """
    model.eval()
    #return_tensors="pt" jsut returns a tensor of the corrext shape to be fed into the model
    input_ids = tokenizer(text, return_tensors="pt")['input_ids']
    out = model(input_ids)
    log_likelihoods = out[input_ids == tokenizer.mask_token_id]
    top_k_likely_indices = t.topk(log_likelihoods, k, dim=-1).indices
    top_k_likely_tokens = [[tokenizer.decode([i]) for i in top_k_likely_indices_for_each_mask] for top_k_likely_indices_for_each_mask in top_k_likely_indices]
    return top_k_likely_tokens

