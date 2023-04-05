# 模型和数据处理

1. 下载 LLaMA 的模型：

    + [LLaMA](https://github.com/facebookresearch/llama)
    + 或者 [unofficial repo](https://github.com/shawwn/llama-dl)

2. 下载 Alpaca 的数据：

    下载 Alpaca 的训练数据 [alpaca_data.json](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)，其中包括了 52K instruction data.

    ```
    bash prepare_llama_training_data.sh
    ```

3. 模型处理：

    + 普通的模型处理：
        ```
        python process_llama_ckpt.py --llama-model-dir [模型地址] --llama-model-file [模型文件]
        ```
    + Megatron-LM 的模型处理 (需要分割模型和填充词典)：
        ```
        python process_llama_megatron_ckpt.py --llama-model-dir [模型地址] --llama-model-file [模型文件] --parallel-size [分割大小]
        ```

4. 模型合并：

    当使用完 Megatron-LM 进行 Fine-tuning 之后，我们可以将多个不同 checkpoint.pt 合并为单个：
    ```
    python merge_llama_megatron_ckpt.py --llama-model-dir [模型地址] --prefix [模型前缀] --parallel-size [分割大小]
    ```
