import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import Optional
from collections import OrderedDict
from safetensors.torch import load_file
import os
import logging
from huggingface_hub import try_to_load_from_cache

class SelectionP(LlamaForCausalLM): 
    def __init__(self, config): 
        super().__init__(config) 
        self.config = config
        self.token_weights = nn.Linear(self.config.hidden_size, 1)
    def forward(
            self, 
            input_ids: torch.LongTensor = None, 
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states = True,
            **kwargs): 
        output = super().forward(input_ids, output_hidden_states=True, labels=labels, **kwargs)
        rloss = output.loss
        lhid = output['hidden_states'][-1].detach()
        output['p'] = torch.sigmoid(self.token_weights(lhid).view(lhid.shape[:2]))
        return output

def load(checkpoint):
    original_log_level = logging.getLogger().getEffectiveLevel()
    logging.disable(logging.WARNING)
    model = SelectionP.from_pretrained(checkpoint, cache_dir='./')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='./')
    cached_file = try_to_load_from_cache(checkpoint, "adapter_model.safetensors")
    save_dict = load_file(cached_file)
    new_state_dict = OrderedDict()
    new_state_dict['token_weights.original_module.weight'] = save_dict['base_model.model.token_weights.weight']
    new_state_dict['token_weights.original_module.bias'] = save_dict['base_model.model.token_weights.bias']
    model.load_state_dict(new_state_dict, strict=False)
    logging.disable(original_log_level)
    return model, tokenizer
