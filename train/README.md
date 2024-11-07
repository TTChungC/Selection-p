# Fine-tuning
#### LLaMA-Factory is used for training Selection-p.

To replicate the training procedure:
- replace the `CustomTrainer` with `SelectionTrainer` in *LLaMA-Factory/src/llmtuner/train/pt/workflow.py*
- replace the `AutoModelForCausalLM` with `SelectionModel` in *LLaMA-Factory/src/llmtuner/model/loader.py*

To train selection-p:
```
python src/train_bash.py  \
    --stage pt \
    --do_train \
    --model_name_or_path PATH_TO_LLAMA_2_7B \ 
    --dataset redpjmv2_500M \
    --finetuning_type lora  \
    --lora_target q_proj,v_proj,k_proj,o_proj  \
    --additional_target token_weights \
    --output_dir selectionp \
    --overwrite_cache  \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10  \
    --save_steps 100 \
    --learning_rate 3e-5 \
    --num_train_epochs 1.0  \
    --plot_loss  \
    --overwrite_output_dir \
    --bf16
```
