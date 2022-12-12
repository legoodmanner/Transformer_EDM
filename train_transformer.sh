# !bash

DATASET=looperman

python train.py \
    --dataset $DATASET \
    --data_path "/home/lego/NAS189/home/$DATASET/all/mel_80_256" \
    --iter 1000 \
    --lr 0.0001 \
    --batch 512 \
    --cuda 1 \
    --save_model_iter 100 \
    --checkpoint_dir "/home/lego/NAS189/home/VQVAE/$DATASET/params" \
    --sample_dir "/home/lego/NAS189/home/VQVAE/$DATASET/sample" \
    --scheduler \
    --mode 'transformer' \
    --transformer_checkpoint_dir "/home/lego/NAS189/home/VQVAE_transformer/$DATASET/params" \
    --transformer_sample_dir "/home/lego/NAS189/home/VQVAE_transformer/$DATASET/sample" \
    --valid_sample_ratio 0.05 \
    --transformer_n_layer 4 \
    --wandb \
    --d_model 512 \
    --embed_dim 80 \
    --n_embed 64 \
    --latent_loss_weight 0.1 \
    --encoder_scale_factors 0.5 0.5 0.5 0.5 \
    --decoder_scale_factors 2 2 2 2 \
    --w_genres