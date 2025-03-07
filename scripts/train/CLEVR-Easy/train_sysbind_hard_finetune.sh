#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
#SEED=$2

#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python sysbinder/train.py \
--seed 0 --batch_size 40 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ \
--lr_dvae 3e-4 --lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 1 --lr_half_life 2 --clip 0.05 \
--epochs 2 --num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 \
--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
--d_model 192 --dropout 0.1 --tau_start 0.1 --tau_final 0.1 --tau_steps 1 --use_dp --temp 1e-4  \
--data_path '/workspace/datasets-local/CLEVR-Easy-1/train/images/*.png' \
--checkpoint_path 'logs/clevr_easy_500_epochs/sysbind_hard_seed0/best_model.pt'

CUDA_VISIBLE_DEVICES=$DEVICE python sysbinder/train.py \
--seed 1 --batch_size 40 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ \
--lr_dvae 3e-4 --lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 1 --lr_half_life 2 --clip 0.05 \
--epochs 2 --num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 \
--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
--d_model 192 --dropout 0.1 --tau_start 0.1 --tau_final 0.1 --tau_steps 1 --use_dp --temp 1e-4  \
--data_path '/workspace/datasets-local/CLEVR-Easy-1/train/images/*.png' \
--checkpoint_path 'logs/clevr_easy_500_epochs/sysbind_hard_seed1/best_model.pt'

CUDA_VISIBLE_DEVICES=$DEVICE python sysbinder/train.py \
--seed 2 --batch_size 40 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ \
--lr_dvae 3e-4 --lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 1 --lr_half_life 2 --clip 0.05 \
--epochs 2 --num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 \
--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
--d_model 192 --dropout 0.1 --tau_start 0.1 --tau_final 0.1 --tau_steps 1 --use_dp --temp 1e-4  \
--data_path '/workspace/datasets-local/CLEVR-Easy-1/train/images/*.png' \
--checkpoint_path 'logs/clevr_easy_500_epochs/sysbind_hard_seed2/best_model.pt'
