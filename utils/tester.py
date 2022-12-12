import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from mel2wav.modules import Generator

class VQVAE_Tester():
    def __init__(self, checkpoint_dir: str, ckpt_id: str, model: nn.Module, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor, device):
        self.model = model
        ckpt = torch.load(
            os.path.join(checkpoint_dir, ckpt_id.zfill(6)+'.pt'), 
            map_location= lambda storage, loc: storage
        )
        model.load_state_dict(ckpt['model'])
        self.loader = loader
        self.device = device
        self.mean = mean
        self.std = std

    def __call__(self):
        data = iter(self.loader).next().to(self.device)
        norm = (data - self.mean) / self.std
        rcnst_mel, _, _id = self.model(norm)
        rcnst_mel = rcnst_mel * self.std + self.mean
        return rcnst_mel, data, _id

class Transformer_Tester():
    def __init__(self, 
        checkpoint_dir: str, 
        ckpt_id: str, 
        model: nn.Module, 
        loader: DataLoader, 
        mean: torch.Tensor, 
        std: torch.Tensor, 
        transformer: nn.Module,
        transformer_checkpoint_dir: str, 
        transformer_ckpt_id: str,
        device
    ):
        self.model = model
        ckpt = torch.load(
            os.path.join(checkpoint_dir, ckpt_id.zfill(6)+'.pt'), 
            map_location= lambda storage, loc: storage
        )
        model.load_state_dict(ckpt['model'])

        self.transformer = transformer
        ckpt = torch.load(
            os.path.join(transformer_checkpoint_dir, transformer_ckpt_id.zfill(6)+'.pt'), 
            map_location= lambda storage, loc: storage
        )
        transformer.load_state_dict(ckpt['transformer'])

        self.loader = loader
        self.device = device
        self.mean = mean
        self.std = std
    
    def __call__(self):
        data = iter(self.loader).next().to(self.device)
        norm = (data - self.mean) / self.std
        norm = self.model.encoder(norm)
        quant, _, _id = self.model.quantizer(norm) 
        self.transformer.eval()
        p = self.transformer(_id)
        return torch.softmax(p,dim=-1), _id

class Mel2Wav():
    def __init__(self, device):
        self.generator = Generator(80, 32, 3).to(device)
        self.generator.load_state_dict(torch.load('/home/lego/NAS189/home/looperman/all/melgan_80_256/best_netG.pt'))
        self.generator.eval()
    def __call__(self, mel):
        with torch.no_grad():
            pred_audio = self.generator(mel)
            pred_audio = pred_audio.detach().cpu().numpy()
        return pred_audio

def get_utilization(id_list, n_embed):
    d = [0] * n_embed
    for i in id_list:
        d[i] += 1

    c = c_ = 0
    for idx in d:
        if idx == 0:
            c_ += 1
        else:
            c += 1
    return c/n_embed