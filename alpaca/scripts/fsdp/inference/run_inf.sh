#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0

data_dir=/opt/data/private/data/llama/llama_instruction/inf/data-bin/
llama_dir=/opt/data/private/ckpt/alpaca/fsdp/
bpe_dir=/opt/data/private/data/llama/tokenizer.model


python alpaca/src/generate.py $data_dir \
    --user-dir alpaca/src \
    --task seq2seq_ft_task \
    --arch llama_7b \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $llama_dir/checkpoint1.pt \
    --required-batch-size-multiple 1 \
    --batch-size 1 \
    --beam 1 --sampling --sampling-topp 0.95 --temperature 0.8 \
