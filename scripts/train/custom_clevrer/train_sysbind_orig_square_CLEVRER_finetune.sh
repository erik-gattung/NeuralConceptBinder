#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
SEED=$2
#DATA=$3

# batch 14, epoch 2, constrast True
# --contrastive_loss

#-------------------------------------------------------------------------------#
CUDA_VISIBLE_DEVICES=$DEVICE python -m sysbinder.train \
--seed 0 --batch_size 14 --num_workers 0 --image_height 128 --image_width 128 --image_channels 3 --log_path logs/ \
--lr_dvae 3e-6 --lr_enc 1e-6 --lr_dec 3e-6 --lr_warmup_steps 2 --lr_half_life 5 --clip 0.05 \
--epochs 10 --num_iterations 3 --num_slots 4 --num_blocks 16 --cnn_hidden_size 512 --slot_size 2048 \
--mlp_hidden_size 192 --num_prototypes 64 --vocab_size 4096 --num_decoder_layers 8 --num_decoder_heads 4 \
--d_model 192 --dropout 0.1 --tau_start 0.1 --tau_final 0.1 --tau_steps 1 --use_dp --temp 1. \
--contrastive_loss_temp 0.4 --contrastive_loss_weight 0.1 \
--data_path '/app/custom_clevrer/square_face_images/*.png' \
--checkpoint_path '/app/ncb/CLEVR-4/retbind_seed_1/best_model.pt'
