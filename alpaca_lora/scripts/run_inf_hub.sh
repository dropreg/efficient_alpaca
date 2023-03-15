export CUDA_VISIBLE_DEVICES=0


torchrun --master_port 29001 --nproc_per_node 1 alpaca_lora/src/inference.py \
    --model-dir /opt/data/private/ckpt/alpaca/lora/ \
    --model-file checkpoint3.pt \
    --llama-model-inf /opt/data/private/data/llama/7B/model.pt \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/llama/tokenizer.model \
