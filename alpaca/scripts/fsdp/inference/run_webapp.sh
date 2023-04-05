
export CUDA_VISIBLE_DEVICES=0

python alpaca/src/webapp.py \
    --model-dir /opt/data/private/ckpt/alpaca/fsdp/ \
    --model-file checkpoint1.pt \
    --bpe sentencepiece \
    --sentencepiece-model /opt/data/private/data/llama/tokenizer.model \
