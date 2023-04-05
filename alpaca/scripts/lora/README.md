# LoRA

Efficient-Finetuning Method: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 


## Training Step

```
bash alpaca/scripts/lora/run_train.sh
```

## Inference Step

+ (Batch-Level) Please prepare the test file.

    ```
    bash alpaca/scripts/lora/inference/run_inf.sh
    ```

+ (Instance-Level) Using alpaca/src/inference.py line 17 to set prompts.

    ```
    bash alpaca/scripts/lora/inference/run_inf_hub.sh
    ```
