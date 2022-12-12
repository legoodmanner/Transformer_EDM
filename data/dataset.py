import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LoopDataset(Dataset):
    def __init__(self, ids, path):
        self.metadata = ids
        self.path = path

    def __getitem__(self, index):
        id = self.metadata[index]
        voc_fp = os.path.join(self.path, id + '.npy')

        voc = np.load(voc_fp)

        return voc

    def __len__(self):
        return len(self.metadata)

def get_loop_datasets(path, feat_type, batch_size, va_sample_ratio):

    dataset_fp = os.path.join(path, f'dataset.pkl')
    in_dir = os.path.join(path, feat_type)
    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    va_samples = round(va_sample_ratio * len(dataset_ids))
    va_ids = dataset_ids[-va_samples:]
    tr_ids = dataset_ids[:-va_samples]

    tr_dataset = LoopDataset(tr_ids, in_dir)
    va_dataset = LoopDataset(va_ids, in_dir)
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


def get_genre_datasets(path, feat_type, batch_size, va_sample_ratio):
    dataset_fp = os.path.join(path, f'genre_dataset.pkl')
    in_dir = os.path.join(path, feat_type)
    with open(dataset_fp, 'rb') as f:
        dataset = pickle.load(f)

    random.seed(1234)
    random.shuffle(dataset)

    va_samples = round(va_sample_ratio * len(dataset))
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