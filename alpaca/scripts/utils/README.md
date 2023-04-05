
# Data Process

Prepare chatbot data contains 52K instruction-following data we used for fine-tuning the Alpaca model.

```
bash prepare_llama_training_data.sh
```

parameter:

+ `DATA` init dataset dir, download the alpaca data [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).
+ `SPM` sentencepiece project spm dir: "sentencepiece/build/src/spm_encode".
+ `MODEL` LLaMA tokenizer model "tokenizer.model".

# Model Process


## Build Model Checkpoint

Process the LLaMA model based on your equipment (GPU devices).

1. single model checkpoint:
    ```
    python alpaca_lora/scripts/utils/process_llama_ckpt.py --llama-model-dir $llama_dir --llama-model-file $llama_file
    ```

2. Megatron-LM model checkpoint:
    ```
    python alpaca_lora/scripts/utils/process_llama_megatron_ckpt.py --llama-model-dir $llama_dir --llama-model-file $llama_file --parallel-size 2
    ```
    
    parameter:
    + `--parallel-size` The number of GPUs to use.

after that, we can get new checkpoint file ``model.pt``.


## Merge Model Checkpoint

We can merge multiple `Megatron-LM Checkpoints` into a single Checkpoint to support the `hub` or `web interface` mode.

```
python merge_llama_megatron_ckpt.py
```
