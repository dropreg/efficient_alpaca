#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1

data_dir=/opt/data/private/data/llama/llama_instruction/inf/data-bin
llama_dir=/opt/data/private/data/llama/7B/megatron_2/
lora_dir=/opt/data/private/ckpt/alpaca/megatron_lora/checkpoint1-model_part-0.pt
bpe_dir=/opt/data/private/data/llama/tokenizer.model
world_size=2


torchrun --master_port 29006 --nproc_per_node $world_size alpaca/src/generate.py $data_dir \
    --user-dir alpaca/src \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --lora-model-inf $lora_dir \
    --task seq2seq_lora_task \
    --arch llama_7b \
    --megatron-model \
    --lora-tuning \
    -s $src -t $tgt \
    --gen-subset test \
    --bpe 'sentencepiece' --sentencepiece-model $bpe_dir \
    --path $llama_dir/model.pt \
    --required-batch-size-multiple 1 \
    --batch-size 1 \
    --beam 1 --sampling --sampling-topp 0.95 --temperature 0.8 \
