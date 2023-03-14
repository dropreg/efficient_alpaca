#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=5

data_dir=/opt/data/private/data/nmt_data/llama_data/story/data-bin
save_dir=/opt/data/private/ckpt/model_parallel_xlmr/LLaMA/alpaca_e/
bpe_dir=/opt/data/private/data/llama/tokenizer.model
world_size=1


torchrun --master_port 29504 --nproc_per_node $world_size examples_nlg/llama/src/generate.py $data_dir \
    --user-dir examples_nlg/llama/src \
    --task llama_translation \
    --arch llama_7b \
    -s $src -t $tgt \
    --gen-subset train \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $save_dir/checkpoint1.pt \
    --fp16 \
    --seed 1 \
    --required-batch-size-multiple 1 \
    --batch-size 1 \
    --beam 1 \
    --sampling \
    --sampling-topp 0.95 --temperature 0.8 \


