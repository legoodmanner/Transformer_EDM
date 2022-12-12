from email.policy import default
import torch
import os
from model import VQVAE
from model.transformer import VQTransformer
import numpy as np
import argparse
from tqdm import tqdm
import re

def best_fname(path):
    fl = os.listdir(path)
    fl.sort(key=lambda f: int(re.sub('\D', '', f)))
    return fl[-1]

def generate(args):
    device = torch.device(f'cuda:{args.cuda}')
    n_latent = int(np.array(args.encoder_scale_factors).prod() * 256)
    runs = str(args.embed_dim)+'_'+str(args.n_embed)+'_'+ str(args.latent_loss_weight).split('.')[-1] + '_' + str(n_latent)
    os.makedirs(f'{args.save_file_path}/{runs}/', exist_ok=True)

    model = VQVAE(
        feat_dim = 80, 
        z_dim = args.embed_dim, 
        encoder_scale_factors = args.encoder_scale_factors, 
        decoder_scale_factors = args.decoder_scale_factors, 
        embed_dim = args.embed_dim, 
        n_embed = args.n_embed,
    ).to(device)

    transformer = VQTransformer(
                d_model=args.d_model, 
                n_layer=args.transformer_n_layer, 
                n_embed=args.n_embed,
                softmax_temp=None
    ).to(device)
    # load model
    # == VQVAE
    best_fn = best_fname(os.path.join(args.checkpoint_dir, runs))
    ckpt = torch.load(
            os.path.join(args.checkpoint_dir, runs, best_fn), 
            map_location= lambda storage, loc: storage
    )
    model.load_state_dict(ckpt['model'])

    # == transformer
    best_fn = best_fname(os.path.join(args.transformer_checkpoint_dir, runs))
    ckpt = torch.load(
            os.path.join(args.transformer_checkpoint_dir, runs, best_fn), 
            map_location= lambda storage, loc: storage
    )
    transformer.load_state_dict(ckpt['transformer'])

    mean_fp = os.path.join(args.data_path, f'mean.mel.npy')
    std_fp = os.path.join(args.data_path, f'std.mel.npy')
    mean = torch.from_numpy(np.load(mean_fp)).float().to(device).view(1, 80, 1)
    std = torch.from_numpy(np.load(std_fp)).float().to(device).view(1, 80, 1)

    model.eval()
    transformer.eval()
    sample_size = args.n_file//100
    with torch.no_grad():
        for i in tqdm(range(100)):
            gen, pl = transformer.evaluate(sample_size, 16, model.quantize, device, threshold_p=args.threshold_p) #[n_sample, L, embed_dim]
            gen = model.decoder(gen.permute(0,2,1))
            norm = gen.data.clone()
            gen = gen * std + mean 
            # torch.Size([sample_size, 80, 256])
            gen = gen.detach().cpu().numpy() 
            for idx, g in enumerate(gen):
                np.save(f'{args.save_file_path}/{runs}/{idx+i*sample_size}.npy', g)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuda', type=int,
    )
    parser.add_argument(
        '--embed_dim', type=int,
    )
    parser.add_argument(
        '--n_embed', type=int,
    )
    parser.add_argument(
        '--d_model', type=int,
    )
    parser.add_argument(
        '--transformer_n_layer', type=int,
    )
    parser.add_argument(
        '--checkpoint_dir', type=str
    )
    parser.add_argument(
        '--transformer_checkpoint_dir', type=str
    )
    parser.add_argument(
        '--save_file_path', type=str
    )
    parser.add_argument(
        '--threshold_p', type=float, default=0.7, help='interface of probablity sampling'
    )
    parser.add_argument(
        '--n_file', type=int, default=2000, help='Number of mel samples to be generated'
    )
    parser.add_argument(
        '--data_path', type=str, 
    )
    parser.add_argument(
        '--latent_loss_weight', type=float,
    )
    parser.add_argument(
        '--bypass', action='store_true', help='bypass the qunatization until utilization reaches the configuration'
    )
    parser.add_argument(
        '--encoder_scale_factors', type=float, nargs="*", default=[0.5, 0.5, 0.5, 0.5], help='encoder downsample factors'
    )
    parser.add_argument(
        '--decoder_scale_factors', type=float, nargs="*", default=[2, 2, 2, 2], help='decoder upsample factors'
    )
    args = parser.parse_args()
    generate(args)