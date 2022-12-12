python train.py \
    --model_save_path '/home/lego/NAS189/home/VQVAE_transformer/eval/IS/params' \
    --data_path '/home/lego/NAS189/home/looperman/all/mel_80_256' \
    --cuda 0 \
    --batch_size 128 \
    --wandb \
    --lr 0.0001 \
    --eval_interval 1