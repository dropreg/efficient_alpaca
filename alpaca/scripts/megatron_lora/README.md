# LoRA

Megatron-LM: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
Efficient-Finetuning Method: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 


## Training Step

```
bash alpaca/scripts/megatron_lora/run_train_megatron_lora.sh
```

## Inference Step

+ (Batch-Level) Please prepare the test file.

    ```
    bash alpaca/scripts/megatron_lora/inference/run_inf_megatron_lora.sh
    ```
