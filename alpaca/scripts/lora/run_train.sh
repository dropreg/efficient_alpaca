#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3

data_dir=/opt/data/private/data/llama/llama_instruction/data-bin
save_dir=/opt/data/private/ckpt/alpaca/lora/
llama_dir=/opt/data/private/data/llama/7B/model_no_pad.pt
max_token=1024
update_freq=1
world_size=4


torchrun --master_port 29000 --nproc_per_node $world_size alpaca/src/train_lora.py $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $llama_dir \
    --user-dir alpaca/src \
    --max-source-positions 2048 \
    --max-target-positions 2048 \
    --memory-efficient-fp16 \
    --fp16 --fp16-init-scale 4 \
    --task seq2seq_lora_task \
    --arch llama_7b \
    --criterion lm_loss \
    --lora-tuning \
    --data-para \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 3e-4 \
    --weight-decay 0.0 \
    --total-num-update 5000 --warmup-updates 200 \
    --max-epoch 3 \
    --no-progress-bar \
    --log-interval 100 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
