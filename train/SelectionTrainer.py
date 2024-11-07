class SelectionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        l1_loss = model.base_model.compute_Lloss(outputs)
        alpha = 1.0/(l1_loss+1e-8)
        return (alpha*loss)+l1_loss
