#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
#DATA=$3

#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python sysbinder/train.py \
--seed 0 --batch_size 40 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ \
--lr_dvae 3e-4 --lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 30000 --lr_half_life 250000 --clip 0.05 \
--epochs 150 --num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 \
--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
--d_model 192 --dropout 0.1 --tau_start 1.0 --tau_final 0.1 --tau_steps 30000 --use_dp --temp 1. --temp_step \
--data_path '/workspace/datasets-local/clevr-easy/**/*.png'

#CUDA_VISIBLE_DEVICES=$DEVICE python sysbinder/train.py \
#--seed 1 --batch_size 40 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ \
#--lr_dvae 3e-4 --lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 30000 --lr_half_life 250000 --clip 0.05 \
#--epochs 500 --num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 \
#--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
#--d_model 192 --dropout 0.1 --tau_start 1.0 --tau_final 0.1 --tau_steps 30000 --use_dp --temp 1. --temp_step \
#--data_path '/workspace/datasets-local/clevr-easy/**/*.png' \
#--checkpoint_path 'logs/2024-02-27T13:27:16.310419/checkpoint.pt.tar'

#CUDA_VISIBLE_DEVICES=$DEVICE python sysbinder/train.py \
#--seed 2 --batch_size 40 --num_workers 0 --image_size 128 --image_channels 3 --log_path logs/ \
#--lr_dvae 3e-4 --lr_enc 1e-4 --lr_dec 3e-4 --lr_warmup_steps 30000 --lr_half_life 250000 --clip 0.05 \
#--epochs 500 --num_iterations 3 --num_slots 4 --num_blocks 8 --cnn_hidden_size 512 --slot_size 2048 \
#--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
#--d_model 192 --dropout 0.1 --tau_start 1.0 --tau_final 0.1 --tau_steps 30000 --use_dp --temp 1. --temp_step \
#--data_path '/workspace/datasets-local/clevr-easy/**/*.png'