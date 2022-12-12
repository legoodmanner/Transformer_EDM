# bin/bash
python generate.py \
    --embed_dim 80 \
    --n_embed 32 \
    --latent_loss_weight 0.1 \
    --cuda 0 \
    --d_model 512 \
    --transformer_n_layer 4 \
    --checkpoint_dir '/home/lego/NAS189/home/VQVAE/params' \
    --transformer_checkpoint_dir '/home/lego/NAS189/home/VQVAE_transformer/params' \
    --save_file_path '/home/lego/NAS189/home/VQVAE_transformer/generate' \
    --data_path "/home/lego/NAS189/home/freesound/mel_80_256" \
    --threshold_p 1 \
    --encoder_scale_factors 0.5 0.5 0.5 0.5 \
    --decoder_scale_factors 2 2 2 2 \