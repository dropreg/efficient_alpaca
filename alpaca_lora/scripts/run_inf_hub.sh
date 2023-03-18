export CUDA_VISIBLE_DEVICES=3

# --lora-model-inf /opt/data/private/ckpt/alpaca/parallel_lora/checkpoint3-model_part-0.pt \
torchrun --master_port 29004 --nproc_per_node 1 alpaca_lora/src/inference.py \
    --model-dir /opt/data/private/data/llama/7B/ \
    --model-file model.pt \
    --lora-tuning \
    --lora-model-inf /opt/data/private/ckpt/alpaca/lora/checkpoint3.pt \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/llama/tokenizer.model \
