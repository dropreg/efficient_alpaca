#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0

data_dir=/opt/data/private/data/llama/llama_instruction/inf/data-bin/
llama_dir=/opt/data/private/data/llama/7B/
lora_dir=/opt/data/private/ckpt/alpaca/lora/checkpoint3.pt
bpe_dir=/opt/data/private/data/llama/tokenizer.model


torchrun --master_port 29001 alpaca/src/generate.py $data_dir \
    --user-dir alpaca/src \
    --task seq2seq_lora_task \
    --arch llama_7b \
    --lora-model-inf $lora_dir \
    --lora-tuning \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $llama_dir/model_no_pad.pt \
    --required-batch-size-multiple 1 \
    --batch-size 1 \
    --beam 1 --sampling --sampling-topp 0.95 --temperature 0.8 \
