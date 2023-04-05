#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

data_dir=/opt/data/private/data/llama/llama_instruction/data-bin
save_dir=/opt/data/private/ckpt/alpaca/fsdp/
llama_dir=/opt/data/private/data/llama/7B/model_no_pad.pt
max_token=2048


python alpaca/src/train_fsdp.py $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $llama_dir \
    --user-dir alpaca/src \
    --ddp-backend fully_sharded \
    --fp16 --fp16-init-scale 4 \
    --checkpoint-activations \
    --no-reshard-after-forward \
    --no-save-optimizer-state \
    --max-target-positions 2048 \
    --task seq2seq_ft_task \
    --arch llama_7b \
    --data-para \
    --criterion lm_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.0 \
    --total-num-update 2000 --warmup-updates 100 \
    --max-epoch 3 \
    --no-progress-bar \
    --log-interval 10 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
