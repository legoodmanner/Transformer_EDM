import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import sys
from data_loader.one_bar_loader import get_audio_loader
from model import ShortChunkCNN_one_bar
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser(description="short chunck CNN one bar")
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--iter", type=int, default=16, help="iteration"
    )
    args = parser.parse_args()
    

    
    # load checkpoint discriminator
    print("load model:", args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    try:
        ckpt_name = os.path.basename(args.ckpt)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

    except ValueError:
        pass
    model = ShortChunkCNN_one_bar(n_class=66).cuda()
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_loader = get_audio_loader(None, 16, 'TEST')
    pbar = range(args.iter)
    
    real_label_numpy = np.zeros(args.batch * args.iter)
    pred_label_numpy = np.zeros(args.batch * args.iter)
    for idx in pbar:
        try:
            real_img, real_label = next(test_iter)
        except:
            test_iter = iter(test_loader)
            real_img, real_label = next(test_iter)
        real_img, real_label = real_img.cuda(), real_label
        
        real_class = model(real_img)
        
        predicted_label = real_class.max(dim=1).indices.cpu()
         
        real_label_numpy[idx * args.batch: (idx + 1) * args.batch] = real_label.numpy()
        pred_label_numpy[idx * args.batch: (idx + 1) * args.batch] = predicted_label.numpy()
    
    
    #compute confusion table
    
    #plt.xticks(range(32), range(32), rotation=90)
    cm = confusion_matrix(real_label_numpy, pred_label_numpy, labels=range(66))
    label_list = np.array(['Dubstep', 'Crunk', 'Techno', 'Industrial', 'Ambient', 'Trap', 'Funk', 'Lo-Fi', 'Breakbeat', 'Electronic', 'Dub', 'Ethnic', 'Jazz', 'Hip Hop', 'Chill Out', 'Rock', 'Rap', 'House', 'Acoustic', 'Fusion', 'Reggaeton', 'Drum And Bass', 'Dance', 'Garage', 'Deep House', 'Glitch', 'Hardstyle', 'Weird', 'Heavy Metal', 'Boom Bap', 'Electro', '8Bit Chiptune', 'Cinematic', 'RnB', 'Pop', 'UK Drill', 'Hardcore', 'Trance', 'Big Room', 'Punk', 'Soul', 'EDM', 'Disco', 'Reggae', 'Samba', 'Dancehall', 'Trip Hop', 'Country', 'Orchestral', 'Jungle', 'Classical', 'Rave', 'Indie', 'Psychedelic', 'Moombahton', 'Religious', 'Dirty', 'Afrobeat', 'Acid', 'Blues', 'Grime', 'Ska', 'Folk', 'Spoken Word', 'Comedy', 'Grunge'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=label_list)
    
    fig, ax = plt.subplots(figsize=(32, 32))
    
    matplotlib.rc('xtick', labelsize=32) 
    matplotlib.rc('ytick', labelsize=32)
    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 10}

    matplotlib.rc('font', **font)
    #disp.xticks(range(32), range(32), rotation=90)
    disp.plot(ax = ax, xticks_rotation=90)
    plt.savefig('./confusion.png')
    #compute accuracy
    
    
    accuracy = sum(1 if real == pred else 0 for real, pred in zip(real_label_numpy, pred_label_numpy))
    accuracy = accuracy / (args.iter* args.batch) 
    print('accuracy: {:.4f}'.format(accuracy))