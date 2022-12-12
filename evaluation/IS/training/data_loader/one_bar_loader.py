# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

META_PATH = '/home/lego/NAS189/home/looperman/all/'
# 'Trap': 35288, 'Hip Hop': 15844, 'Industrial': 9246, 'Dubstep': 8500, 'Drum And Bass': 6061, 'Rap': 5156, 'Electronic': 5059, 'Rock': 4105, 'House': 2463, 'Funk': 2290, 'Chill Out': 2230, 'Dance': 2097, 'RnB': 1986, 'Electro': 1781, 'Techno': 1695, 'Pop': 1676, 'Glitch': 1634, 'Heavy Metal': 1607, 'Breakbeat': 1413, 'Lo-Fi': 1291, 'Weird': 1145, 'Deep House': 1062, 'Ambient': 974, 'Cinematic': 931, 'Boom Bap': 905, 'Reggaeton': 778, 'Dancehall': 723, 'Trance': 721, 'Ethnic': 678, 'Jazz': 661, 'Hardcore': 623, 'Soul': 608, 'Fusion': 589, 'Crunk': 466, 'Reggae': 453, 'Dub': 400, 'Disco': 399, 'EDM': 350, 'Trip Hop': 350, 'Hardstyle': 333, 'Jungle': 322, 'Dirty': 322, 'Garage': 304, 'UK Drill': 278, 'Acoustic': 229, 'Samba': 214, 'Psychedelic': 213, 'Grime': 207, 'Orchestral': 191, 'Blues': 184, 'Moombahton': 163, '8Bit Chiptune': 156, 'Country': 137, 'Afrobeat': 134, 'Punk': 120, 'Rave': 88, 'Indie': 69, 'Acid': 61, 'Classical': 49, 'Grunge': 25, 'Folk': 20, 'Big Room': 17, 'Spoken Word': 14, 'Ska': 10, 'Religious': 8, 'Comedy': 5
TAGS = ['Dubstep', 'Crunk', 'Techno', 'Industrial', 'Ambient', 'Trap', 'Funk', 'Lo-Fi', 'Breakbeat', 'Electronic', 'Dub', 'Ethnic', 'Jazz', 'Hip Hop', 'Chill Out', 'Rock', 'Rap', 'House', 'Acoustic', 'Fusion', 'Reggaeton', 'Drum And Bass', 'Dance', 'Garage', 'Deep House', 'Glitch', 'Hardstyle', 'Weird', 'Heavy Metal', 'Boom Bap', 'Electro', '8Bit Chiptune', 'Cinematic', 'RnB', 'Pop', 'UK Drill', 'Hardcore', 'Trance', 'Big Room', 'Punk', 'Soul', 'EDM', 'Disco', 'Reggae', 'Samba', 'Dancehall', 'Trip Hop', 'Country', 'Orchestral', 'Jungle', 'Classical', 'Rave', 'Indie', 'Psychedelic', 'Moombahton', 'Religious', 'Dirty', 'Afrobeat', 'Acid', 'Blues', 'Grime', 'Ska', 'Folk', 'Spoken Word', 'Comedy', 'Grunge']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy'),
                'tags': row[5:],
            }
    return tracks


class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()
        self.get_tag_label()

    def __getitem__(self, index):
        npy, label = self.get_npy(index)
        return npy.astype('float32'), label

    def get_songlist(self):
        #self.mlb = LabelBinarizer().fit(TAGS)
        if self.split == 'TRAIN':
            train_file = os.path.join(META_PATH, 'dict_one_bar.pickle')
            with open(train_file, 'rb') as f:
                self.file_dict = pickle.load(f)
            self.fl = list(self.file_dict.keys())[:100000]
        elif self.split == 'VALID':
            train_file = os.path.join(META_PATH, 'dict_one_bar.pickle')
            with open(train_file, 'rb') as f:
                self.file_dict = pickle.load(f)
            self.fl = list(self.file_dict.keys())[100000:110000]
        elif self.split == 'TEST':
            train_file = os.path.join(META_PATH, 'dict_one_bar.pickle')
            with open(train_file, 'rb') as f:
                self.file_dict = pickle.load(f)
            self.fl = list(self.file_dict.keys())[110000:]
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')
    def get_tag_label(self):
        self.tag_to_label = {}
        for i in range(len(TAGS)):
            self.tag_to_label[TAGS[i]] = i

    def get_npy(self, index):
        filename = self.fl[index]
        tag = self.file_dict[filename]
        label = self.tag_to_label[tag]
        npy_path = os.path.join(META_PATH, f'mel_80_256/{filename}')
        npy = np.load(npy_path)
        #random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        #npy = np.array(npy[random_idx:random_idx+self.input_length])
        #tag_binary = self.mlb.transform([tag]).flatten()
        return npy, label

    def __len__(self):
        return len(self.fl)

def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None):
    if split in ['TRAIN', 'VALID']:
        data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      drop_last=False,
                                      num_workers=num_workers)
        
    return data_loader

if __name__ == '__main__':
    loader = get_audio_loader(None, 16)
    for data in loader:
        import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
