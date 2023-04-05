#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_dir=/opt/data/private/data/llama/llama_instruction/inf/data-bin/
llama_dir=/opt/data/private/ckpt/alpaca/megatron8_ft/
bpe_dir=/opt/data/private/data/llama/tokenizer.model
world_size=8


torchrun --master_port 29006 --nproc_per_node $world_size alpaca/src/generate.py $data_dir \
    --user-dir alpaca/src \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --task seq2seq_ft_task \
    --megatron-model \
    --arch llama_7b \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $llama_dir/checkpoint1.pt \
    --required-batch-size-multiple 1 \
    --batch-size 1 \
    --beam 1 --sampling --sampling-topp 0.95 --temperature 0.8 \
