#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_dir=/opt/data/private/data/llama/llama_instruction/data-bin
save_dir=/opt/data/private/ckpt/alpaca/megatron8_ft/
llama_dir=/opt/data/private/data/llama/7B/megatron_8/model.pt
max_token=2048
update_freq=1
world_size=8


torchrun --master_port 29000 --nproc_per_node $world_size alpaca/src/train_megatron.py $data_dir \
    --model-parallel-size $world_size \
    --distributed-world-size $world_size \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $llama_dir \
    --user-dir alpaca/src \
    --max-source-positions 2048 \
    --max-target-positions 2048 \
    --memory-efficient-fp16 \
    --fp16 --fp16-init-scale 4 \
    --checkpoint-activations \
    --task seq2seq_ft_task \
    --arch llama_7b \
    --megatron-model \
    --criterion lm_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2e-5 \
    --weight-decay 0.0 \
    --total-num-update 7000 --warmup-updates 200 \
    --max-epoch 3 \
    --no-progress-bar \
    --log-interval 100 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
