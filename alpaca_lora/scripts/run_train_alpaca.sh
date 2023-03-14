#!/bin/bash
src=src
tgt=tgt

export CUDA_VISIBLE_DEVICES=0

data_dir=/opt/data/private/data/llama/data/data-bin
save_dir=/opt/data/private/ckpt/model_parallel_xlmr/LLaMA/alpaca_new/
xlmr_path=/opt/data/private/data/llama/7B/model.pt
max_token=1024
update_freq=2
world_size=1

torchrun --master_port 29001 --nproc_per_node $world_size alpaca_lora/scripts/train.py $data_dir \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $xlmr_path \
    --user-dir alpaca_lora/src \
    --max-target-positions 1024 \
    --share-all-embeddings \
    --task llama_task \
    --arch llama_7b \
    --criterion llama_loss \
    -s $src -t $tgt \
    --max-tokens $max_token \
    --update-freq $update_freq \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 2e-4 \
    --weight-decay 0.01 \
    --total-num-update 8000 --warmup-updates 200 \
    --no-progress-bar \
    --max-epoch 3 \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 128 \
    --save-dir $save_dir | tee -a $save_dir/train.log \
