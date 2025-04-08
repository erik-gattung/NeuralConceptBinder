#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
#DATA=$3

#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python -m sysbinder.train \
--seed 0 --batch_size 32 --num_workers 0 --image_height 128 --image_width 128 --image_channels 3 --log_path logs/ \
--lr_dvae 3e-5 --lr_enc 2e-5 --lr_dec 3e-5 --lr_warmup_steps 15000 --lr_half_life 125000 --clip 0.05 \
--epochs 600 --num_iterations 3 --num_slots 4 --num_blocks 16 --cnn_hidden_size 512 --slot_size 2048 \
--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
--d_model 192 --dropout 0.1 --tau_start 1.0 --tau_final 0.1 --tau_steps 15000 --use_dp --temp 1. \
--data_path '<path to video frames dir>/*.jpg' \
