class SelectionModel(LlamaForCausalLM): 
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
        # 1st forward to get the guide (last hidden state) for token selection
        output = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            labels=labels, 
            **kwargs
        )
        last_hidden = output['hidden_states'][-1].detach()
        # compute the token selection
        p = torch.sigmoid(self.token_weights(last_hidden).view(last_hidden.shape[:2]))
        topk = torch.topk(p,int(0.1*input_ids.shape[1]))
        attention_mask = torch.zeros(p.shape).to(p.device)
        for pdx, pos in enumerate(topk.indices):
            attention_mask[pdx][pos]=1
        # compute the lm loss after token selection
        output = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            labels=labels, 
            **kwargs
        )
        output['p']=p
        return output
    
    def compute_Lloss(self, output, p=1):
        loss = torch.norm(output['p'], p=p).to(torch.float32)
        return 100*loss/output['p'].shape[1]