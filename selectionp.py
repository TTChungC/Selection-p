import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer
from typing import Optional
from collections import OrderedDict
from safetensors.torch import load_file
import os

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
    model = SelectionP.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer
