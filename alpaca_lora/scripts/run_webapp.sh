
export CUDA_VISIBLE_DEVICES=1

torchrun --master_port 29009 --nproc_per_node 1 alpaca_lora/src/webapp.py \
    --model-dir /opt/data/private/data/llama/7B/ \
    --model-file model.pt \
    --lora-model-inf /opt/data/private/ckpt/alpaca/lora/checkpoint3.pt \
    --lora-tuning \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/llama/tokenizer.model \
