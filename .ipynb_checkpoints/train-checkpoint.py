from genericpath import exists
import os
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import argparse
from tqdm import tqdm
import torchvision.utils
from fast_transformers.masking import TriangularCausalMask
from einops import rearrange

import data
from model import VQVAE
from model.transformer import VQTransformer
from utils.scheduler import CycleScheduler
from utils.tester import get_utilization
try:
    import wandb
except ImportError:
    wandb = None

def train_vqvae(args, tr_dataloader, va_dataloader, model, optim, scheduler, device):
    os.makedirs(
        os.path.join(
            args.checkpoint_dir, str(args.embed_dim)+'_'+str(args.n_embed)
            ),
        exist_ok=True
        )
    os.makedirs(
        os.path.join(
            args.sample_dir, str(args.embed_dim)+'_'+str(args.n_embed)
            ),
        exist_ok=True
        )
    pbar = tqdm(range(args.iter), initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    criterion = nn.MSELoss()
    latent_loss_weight = args.latent_loss_weight

    if args.distributed:
        model = model.module
    
    mean_fp = os.path.join(args.data_path, f'mean.mel.npy')
    std_fp = os.path.join(args.data_path, f'std.mel.npy')
    mean = torch.from_numpy(np.load(mean_fp)).float().to(device).view(1, 80, 1)
    std = torch.from_numpy(np.load(std_fp)).float().to(device).view(1, 80, 1)

    for _epoch in pbar:
        epoch = _epoch + args.start_iter
        if epoch > args.iter:
            print('Done')
            break


        #tr_dataloader = tqdm(tr_dataloader, dynamic_ncols=True, smoothing=0.01)
        #va_dataloader = tqdm(va_dataloader, dynamic_ncols=True, smoothing=0.01)

        # train
        tr_loss = 0
        tl_loss = 0
        model.train()
        id_list = []
        for idx, x in enumerate(tr_dataloader):
            model.zero_grad()

            x = x.to(device)
            x = (x - mean) / std

            recnst, latent_loss, _id = model(x, bypass=_epoch < 0)
            reconstruct_loss = criterion(recnst, x)
            latent_loss = latent_loss.mean()
            loss = reconstruct_loss + latent_loss * latent_loss_weight
            loss.backward()

            if scheduler is not None:
                scheduler.step()
            optim.step()

            tl_loss += latent_loss.item()
            tr_loss += reconstruct_loss.item()
            id_list += _id.detach().flatten().tolist()
        tl_loss /= (idx + 1)
        tr_loss /= (idx + 1)
        tr_util_rate = get_utilization(id_list, args.n_embed)
        # Valid
        vr_loss = 0
        vl_loss = 0
        model.eval()
        id_list = []
        for idx, x in enumerate(va_dataloader):
            with torch.no_grad():
                model.zero_grad()

                x = x.to(device)
                x = (x - mean) / std
                
                recnst, latent_loss, _id = model(x)
                reconstruct_loss = criterion(recnst, x)
                latent_loss = latent_loss.mean()
                vl_loss += latent_loss.item()
                vr_loss += reconstruct_loss.item()
                id_list += _id.detach().flatten().tolist()
        vl_loss /= (idx + 1)
        vr_loss /= (idx + 1)
        va_util_rate = get_utilization(id_list, args.n_embed)

        if epoch % 10 == 0:
            sample_size = 4
            x = x[:sample_size].unsqueeze(1)
            recnst = recnst[:sample_size].unsqueeze(1)

            torchvision.utils.save_image(
                torch.cat([x, recnst], 0),
                os.path.join(args.sample_dir, str(args.embed_dim)+'_'+str(args.n_embed), str(epoch).zfill(5)+".png"),
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
  
        lr = optim.param_groups[0]['lr']
        pbar.set_description(
            (
                f"train | latent loss: {tl_loss:.4f}; reconstruct loss: {tr_loss:.4f} "
                f"valid | latent loss: {vl_loss:.4f}; reconstruct loss: {vr_loss:.4f} "
                f"lr : {lr}"
            )
        )
        if wandb and args.wandb:
                wandb.log(
                    {
                        "LatentLoss": {'Train': tl_loss, 'Valid': vl_loss},
                        "ReconstructLoss": {'Train': tr_loss, 'Valid': vr_loss},
                        "Embed Utilization": {'Train': tr_util_rate, 'Valid': va_util_rate},
                        "LearningRate": lr
                    }
                )

        # save model
        if epoch % args.save_model_iter == 0:
            torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "args": args,
                    },
                    f"{args.checkpoint_dir}/{args.embed_dim}_{args.n_embed}/{str(epoch).zfill(6)}.pt",
            )

def train_transformer(args, tr_dataloader, va_dataloader, model, transformer, optim, scheduler, device):
    os.makedirs(args.transformer_checkpoint_dir, exist_ok=True)
    os.makedirs(args.transformer_sample_dir, exist_ok=True)
    pbar = tqdm(range(args.iter), initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    criterion = nn.CrossEntropyLoss()
    
    if args.distributed:
        model = model.module
        transformer = transformer.module
    
    ckpt_path = os.path.join(args.checkpoint_dir, str(args.embed_dim)+'_'+str(args.n_embed), args.ckpt_id.zfill(6)+'.pt')
    print(f'loadind vqvae model from {ckpt_path}')
    ckpt = torch.load(
        ckpt_path, 
        map_location= lambda storage, loc: storage
    )
    model.load_state_dict(ckpt['model'])

    mean_fp = os.path.join(args.data_path, f'mean.mel.npy')
    std_fp = os.path.join(args.data_path, f'std.mel.npy')
    mean = torch.from_numpy(np.load(mean_fp)).float().to(device).view(1, 80, 1)
    std = torch.from_numpy(np.load(std_fp)).float().to(device).view(1, 80, 1)

    model.eval()
    transformer.train()

    for _epoch in pbar:
        epoch = _epoch + args.start_iter
        if epoch > args.iter:
            print('Done')
            break

        # train
        t_loss = 0
        for idx, x in enumerate(tr_dataloader):
            model.zero_grad()
            transformer.zero_grad()

            x = x.to(device)
            x = (x - mean) / std

            with torch.no_grad():
                x = model.encoder(x)
                quant, _, _id = model.quantizer(x) 
                # quant:[bs, embed_dim, T'] |  _id:[bs, T']
            p = transformer(_id) # [bs, T', n_embed]
            p = rearrange(p, 'b t c -> (b t) c')
            _id = rearrange(_id, 'b t -> (b t)')
            loss = criterion(p, _id)
            loss.backward()

            if scheduler is not None:
                scheduler.step()
            optim.step()
            t_loss += loss.item()
        t_loss /= (idx+1)

        lr = optim.param_groups[0]['lr']
        pbar.set_description(
            (
                f"train | CrossEntropy loss: {t_loss:.4f} |"
                f"lr : {lr} |"
            )
        )
        if wandb and args.wandb:
            wandb.log(
                {
                    "CrossEntropyLoss": t_loss,
                    "LearningRate": lr
                }
            )
        
        if epoch % 10 == 0:
            model.eval()
            transformer.eval()
            sample_size = 8
            with torch.no_grad():
                gen, _ = transformer.evaluate(sample_size, 20, model.quantize, device) #[n_sample, L, embed_dim]
                gen = model.decoder(gen.permute(0,2,1))
            torchvision.utils.save_image(
                gen.unsqueeze(1),
                os.path.join(args.transformer_sample_dir, str(epoch).zfill(5)+".png"),
                nrow=sample_size//2,
                normalize=True,
                range=(-1, 1),
            )
            p = rearrange(p, '(b t) c -> b t c',b=x.size(0), t=x.size(-1))
            _id = rearrange(_id, '(b t) -> b t',b=x.size(0), t=x.size(-1))
    
        if epoch % args.save_model_iter == 0:
            torch.save(
                    {
                        "transformer": transformer.state_dict(),
                        "optim": optim.state_dict(),
                        "args": args,
                    },
                    f"{args.transformer_checkpoint_dir}/{str(epoch).zfill(6)}.pt",
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQVAE-UNAencoder trainer')

    parser.add_argument( 
        '--data_path', type=str, help='the path to the root of dataset',
    )
    parser.add_argument(
        '--iter', type=int, default=10000, help='the total iters to complete the training',
    )
    parser.add_argument(
        '--start_iter', type=int, default=0, help='the number of the iter to start training'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate'
    )
    parser.add_argument(
        '--batch', type=int, help='batch size'
    )
    parser.add_argument(
        '--cuda', type=int, help='cuda #' 
    )
    parser.add_argument(
        '--distributed', action='store_true', help='parellel maybe?'
    )
    parser.add_argument(
        "--save_model_iter", type=int, default=1000, help="saving model parameters for what number of iteration"
    )
    parser.add_argument(
        '--checkpoint_dir', type=str, help='path to save the model pt'
    )
    parser.add_argument(
        '--sample_dir', type=str, help='the path to store the samples of mel images'
    )
    parser.add_argument(
        '--wandb', action='store_true', help='using wandb or not'
    )
    parser.add_argument(
        '--scheduler', action='store_true', help='using lr scheduler or not'
    )
    parser.add_argument(
        '--mode', type=str, help='determined which stage and which trainer sholud be used'
    )
    parser.add_argument(
        '--transformer_checkpoint_dir', type=str, help='path to save the transformer pt'
    )
    parser.add_argument(
        '--transformer_sample_dir', type=str, help='the path to store the samples of mel images'
    )
    parser.add_argument(
        '--valid_sample_ratio', type=float, default=0.2, help='determine the ratio between training/valid of data'
    )

    # model config
    parser.add_argument(
        '--embed_dim', type=int, default=20, help='the dimension of single latent vector'
    )
    parser.add_argument(
        '--n_embed', type=int, default=512, help='the number of the vectors stored in the codebook in VQ stage'
    )
    parser.add_argument(
        '--transformer_n_layer', type=int, default=8, help='number of layers in transformer'
    )
    parser.add_argument(
        '--d_model', type=int, default=256, help='transformers dmodel'
    )
    parser.add_argument(
        '--ckpt_id', type=str, default=None, help='determined what VQVAE model id ckpt to load'
    )
    parser.add_argument(
        '--latent_loss_weight', type=float, default=0.25, help='latent loss weight'
    )

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')
    model = VQVAE(
        feat_dim = 80, 
        z_dim = args.embed_dim, 
        encoder_scale_factors = [0.5, 0.5, 0.5, 0.5], 
        decoder_scale_factors = [2, 2, 2, 2], 
        embed_dim = args.embed_dim, 
        n_embed = args.n_embed,
    ).to(device)

    if args.mode == 'vqvae':
        optim = opt.Adam(
            model.parameters(),
            lr= args.lr,
        )
    elif args.mode == 'transformer':
        transformer = VQTransformer(
            d_model=args.d_model, 
            n_layer=args.transformer_n_layer, 
            n_embed=args.n_embed
            ).to(device)
        print(f'transformer parameters: {sum(p.numel() for p in transformer.parameters())}')
        optim = opt.Adam(
            transformer.parameters(),
            lr= args.lr,
        )
    else:
        print('invalid mode')

    tr_dataloader, num_tr, va_dataloader, num_va = data.get_loop_datasets(args.data_path, "", args.batch, args.valid_sample_ratio)
    
    if args.scheduler:
        scheduler = CycleScheduler(
            optim,
            args.lr,
            n_iter=len(tr_dataloader) * args.iter,
            momentum=None,
            warmup_proportion=0.05,
        )
    else:
        scheduler = None

    print('preparing dataset...')
    if wandb is not None and args.wandb:
        wandb.init(project=f"LoopGeneration-{args.mode}", entity="yklego", config=args)
        print('using wandb')
    
    print('start training')
    
    if args.mode == 'vqvae':
        train_vqvae(args, tr_dataloader, va_dataloader, model, optim, scheduler, device)
    elif args.mode == 'transformer':
        train_transformer(args, tr_dataloader, va_dataloader, model, transformer, optim, scheduler, device)
    else:
        print('no trainer detected')