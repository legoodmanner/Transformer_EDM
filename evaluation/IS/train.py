import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import argparse
import wandb
from model import ShortChunkCNN_one_bar
from sklearn import metrics

class GenreDataset(Dataset):
    def __init__(self, dataset, path):
        self.metadata = dataset
        self.path = path

    def __getitem__(self, index):
        id = self.metadata[index][0]
        genre = self.metadata[index][1]
        voc_fp = os.path.join(self.path, id )
        voc = np.load(voc_fp)
        return voc, genre

    def __len__(self):
        return len(self.metadata)

def get_genre_data(path, feat_type, batch_size, va_samples):
    dataset_fp = os.path.join(path, f'train_dataset.pkl')
    in_dir = os.path.join(path, feat_type)
    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    random.seed(1234)
    random.shuffle(dataset)

    # va_samples = round(va_sample_ratio * len(dataset))
    va_ids = dataset[-va_samples:]
    tr_ids = dataset[:-va_samples]

    tr_dataset = GenreDataset(tr_ids, in_dir)
    va_dataset = GenreDataset(va_ids, in_dir)
    num_tr = len(tr_dataset)
    num_va = len(va_dataset)

    iterator_tr = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True)
    
    iterator_va = DataLoader(
        va_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=True,
        pin_memory=True)
    return iterator_tr, num_tr, iterator_va, num_va


def solver(args, model, opt, tr_dataloader, va_dataloader, device):
    os.makedirs(args.model_save_path, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    best_loss = 1000
    best_epoch = 0
    for epoch in range(args.n_epochs+1):
        trloss = 0
        model.train()
        for idx, (x, label) in enumerate(tqdm(tr_dataloader)):
            pred = model(x.to(device))
            loss = criterion(pred, label.long().to(device))
            loss.backward()
            opt.step()
            trloss += loss.item()
        trloss /= (idx+1)
        
        if epoch % args.eval_interval == 0:
            valoss = 0
            model.eval()
            val_true = []
            val_pred = []
            for idx, (x, label) in enumerate(tqdm(va_dataloader)):
                with torch.no_grad():
                    pred = model(x.to(device))
                loss = criterion(pred, label.long().to(device))
                valoss += loss.item()
                val_pred.extend(list(pred.argmax(dim=1).detach().cpu().numpy()))
                val_true.extend(list(label.detach().cpu().numpy()))
            valoss /= (idx+1)
                
              
            p, r, f1, _ = metrics.precision_recall_fscore_support(val_true, val_pred, average='weighted')
            if wandb and args.wandb:
                wandb.log(
                    {
                        "CrossEntropyLoss": {'Train': trloss, 'Valid': valoss},
                        "Precision": p,
                        "Recall": r,
                        "f1": f1,
                    }
                )
            print(
                f'train loss: {trloss:.4f} | valid loss: {valoss:.4f}'
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": opt.state_dict(),
                    "args": args,
                },
                f"{args.model_save_path}/{str(epoch).zfill(6)}.pt",
            )
            if valoss < best_loss:
                best_loss = valoss
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "args": args,
                    },
                    f"{args.model_save_path}/best_model.pt",
                ) 
        else:
            print(
                f'train loss: {trloss:.4f}'
            )
    print(f'best epoch : {best_epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--model_save_path', type=str, default='.')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--valid_sample', type=int, default=10000)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}')
    tr_dataloader, num_tr, va_dataloader, num_va = get_genre_data(args.data_path, "", args.batch_size, args.valid_sample)
    model = ShortChunkCNN_one_bar(n_class=66).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.wandb:
        wandb.init(
            project=f"LoopGeneration-IS", 
            entity="yklego", 
            config=args, 
            name= 'Looperman 66 classes classifier inception score'
        )
        print('using wandb')
    
    solver(args, model, opt, tr_dataloader, va_dataloader, device)