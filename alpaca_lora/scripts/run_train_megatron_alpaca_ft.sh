#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_dir=/opt/data/private/data/llama/llama_raw/data-bin
save_dir=/opt/data/private/ckpt/alpaca/parallel_lora_ft/
llama_dir=/opt/data/private/data/llama/7B/megatron_8/model.pt
max_token=1024
update_freq=2
world_size=8


torchrun --master_port 29002 --nproc_per_node $world_size alpaca_lora/src/train.py $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $llama_dir \
    --user-dir alpaca_lora/src \
    --max-target-positions 1024 \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --task llama_task \
    --arch llama_7b \
    --criterion llama_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.0 \
    --total-num-update 7000 --warmup-updates 200 \
    --no-progress-bar \
    --max-epoch 3 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
