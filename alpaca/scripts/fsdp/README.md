# Fully Sharded Data Parallel

FSDP: [Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)

## Training Step

```
bash alpaca/scripts/fsdp/run_train.sh
```

```
bash alpaca/scripts/fsdp/run_train_cpu_offload.sh
```

## Inference Step

+ (Batch-Level) Please prepare the test file.

    ```
    bash alpaca/scripts/fsdp/inference/run_inf.sh
    ```

+ (Instance-Level) Using alpaca/src/inference.py line 17 to set prompts.

    ```
    bash alpaca/scripts/fsdp/inference/run_inf_hub.sh
    ```