#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python inception_score.py \
    ./best_model.ckpt \
    --classes 66 \
    --store_every_score freesound.pkl \
    --mean_std_dir /home/lego/NAS189/home/freesound/mel_80_256 \
    --data_dir /home/lego/NAS189/home/VQVAE_transformer/generate/80_32_1_ \
    # --data_dir /home/lego/NAS189/home/VQVAE_transformer/generate \

