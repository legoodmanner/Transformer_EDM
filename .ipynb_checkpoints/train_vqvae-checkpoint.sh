# !bash

python train.py \
    --data_path "/home/lego/NAS189/homes/allenhung/Project/loop-generation/data/freesound/drum_audio/mel_80_320" \
    --iter 1000 \
    --lr 0.0001 \
    --batch 128 \
    --cuda 1 \
    --save_model_iter 50 \
    --checkpoint_dir '/home/lego/NAS189/home/VQVAE/params' \
    --sample_dir '/home/lego/NAS189/home/VQVAE/sample' \
    --wandb \
    --scheduler \
    --mode 'vqvae' \
    --embed_dim 20 \
    --latent_loss_weight 0.05 \
    --n_embed 512