
export CUDA_VISIBLE_DEVICES=0

torchrun --master_port 29002 alpaca/src/webapp.py \
    --model-dir /opt/data/private/data/llama/7B/ \
    --model-file model_no_pad.pt \
    --lora-model-inf /opt/data/private/ckpt/alpaca/lora/checkpoint3.pt \
    --lora-tuning \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/llama/tokenizer.model \
