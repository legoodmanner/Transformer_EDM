import torch
import numpy as np
import os
import json
from tqdm import tqdm
import argparse
import utils
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
from madmom.features.key import CNNKeyRecognitionProcessor

import warnings
warnings.filterwarnings("ignore")

def builder(model_type):
    # get model information from csv
    input_csv_name = 'spl_models_hmmparams.csv'
    input_csv_path = os.path.join('drum_aware4beat', input_csv_name)

    # creating inputinfo_list for evaluation
    rnn, modelinfo = utils.string2model(model_type=model_type, input_csv_path=input_csv_path)

    model_fn = 'RNNBeatProc.pth'
    model_path = os.path.join('drum_aware4beat', modelinfo['model_dir'] , model_fn)
    state = torch.load(model_path, map_location = device)
    rnn.load_state_dict(state)

    
    
    return rnn, modelinfo

def beatdownbeat_estimator(audio_file_path, rnn, hmm_proc, device):
    feat = utils.get_feature(audio_file_path)
    try:
        activation = utils.get_dlm_activation(rnn, device, feat)
        beat_fuser_est = hmm_proc(activation)
        # shape: [number of downbeats]
        downbeat = beat_fuser_est[np.where(beat_fuser_est[:,1]==1), 0].squeeze().tolist()
        # shape: [number of beats]
        beat = beat_fuser_est[:, 0].tolist()
    except:
        beat = []
        downbeat = []
    return beat, downbeat

def freesound_parser(wav_root, meta_path, device):
    # wav files root .../.../audio/
    # meta path .../.../metadata.json
    key_proc = CNNKeyRecognitionProcessor()
    beats_model, modelinfo = builder(model_type='DA1')

    print('processing freesound...')

    new_meta = {}
    with open(meta_path) as f:
        meta = json.load(f)
        for value in tqdm(meta.values()): 
            data = {}
            # ex: "http://freesound.org/data/previews/426/426027_2155630-hq.mp3" -> 426027_2155630
            data['file_id'] = value['preview_url'].split('/')[-1].split('-')[0]
            if os.path.isfile(os.path.join(wav_root, data['file_id']+'.wav.wav')):
                data['filename'] = data['file_id'] + '.wav.wav'
            elif os.path.isfile(os.path.join(wav_root, data['file_id']+'.aiff.wav')):
                data['filename'] = data['file_id'] + '.aiff.wav'
            else:
                continue
            if os.path.getsize(os.path.join(wav_root, data['filename'])) < 0:
                continue
            data['genre'] = None
            data['category'] = None
            data['key'] = None
            data['bpm'] = None
            data['calculate_key'] = utils.estimate_key(os.path.join(wav_root, data['filename']), key_proc)
            
            try:
                
                hmm_proc = DownBproc(beats_per_bar = beats_per_bar, min_bpm = min_bpm, 
                                    max_bpm = max_bpm, num_tempi = modelinfo['n_tempi'], 
                                    transition_lambda = modelinfo['transition_lambda'], 
                                    observation_lambda = modelinfo['observation_lambda'], 
                                    threshold = modelinfo['threshold'], fps = 100)

            
                beats, downbeats = beatdownbeat_estimator(
                    os.path.join(wav_root, data['filename']), 
                    beats_model, 
                    hmm_proc, 
                    device
                    )
            except:
                print(f"error")
                continue

            estimate_bpm = utils.estimate_bpm(beats)

            data['beats'] = beats
            data['downbeats'] = downbeats
            data['calculate_bpm'] = estimate_bpm

            new_meta[data['file_id']] = data

    return new_meta

def bandlab_parser(wav_root, meta_path, device):
    # wav files root .../.../BANDLAB_INSTRUMENT/
    # meta path .../.../bandlab.json.json
    key_proc = CNNKeyRecognitionProcessor()
    beats_model, modelinfo = builder(model_type='DA1')
    beats_per_bar = [3, 4]
    print('processing bandlab...')

    new_meta = {}
    with open(meta_path,'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = {}
            dic = json.loads(line)
            if not os.path.isfile(os.path.join(wav_root, dic['instrument'], dic['audio_file'])):
                continue
            if os.path.getsize(os.path.join(wav_root, dic['instrument'], dic['audio_file'])) == 0:
                continue
            data['filename'] = dic['audio_file']
            data['file_id'] = dic['audio_file'].split('.')[0]
            data['genre'] = dic['genres']
            data['category'] = dic['instrument']
            data['key'] = dic['key']
            data['bpm'] = dic['bpm']
            data['calculate_key'] = utils.estimate_key(os.path.join(wav_root, data['category'], data['filename']), key_proc)
            try:
                hmm_proc = DownBproc(beats_per_bar = beats_per_bar, 
                                    min_bpm = int(data['bpm'])-10 if not data['bpm'] == '' else 60, 
                                    max_bpm = int(data['bpm'])+10 if not data['bpm'] == '' else 200,
                                    num_tempi = modelinfo['n_tempi'], 
                                    transition_lambda = modelinfo['transition_lambda'], 
                                    observation_lambda = modelinfo['observation_lambda'], 
                                    threshold = modelinfo['threshold'], fps = 100)

                beats, downbeats = beatdownbeat_estimator(
                    os.path.join(wav_root, data['category'], data['filename']), 
                    beats_model, 
                    hmm_proc, 
                    device
                    )
            except:
                print('error')
                continue

            estimate_bpm = utils.estimate_bpm(beats)

            data['beats'] = beats
            data['downbeats'] = downbeats
            data['calculate_bpm'] = estimate_bpm
            
            new_meta[data['file_id']] = data
    return new_meta

parser = argparse.ArgumentParser(description="v1 parser")
parser.add_argument(
    '--cuda_num', type=int, default=0,  
)
parser.add_argument(
    '--dataset', type=str, help='determined which dataset to parse'
)
args = parser.parse_args()


# CONFIG
f_measure_threshold=0.07 # 70ms tolerance as set in paper
cuda_str = 'cuda:'+str(args.cuda_num)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# Process
if args.dataset == 'freesound':
    new_meta = freesound_parser(
                    wav_root='../NAS189/homes/allenhung/Project/loop-generation/data/freesound/audio/wav/', 
                    meta_path='../NAS189/homes/allenhung/Project/loop-generation/data/freesound/metadata.json',
                    device=device,
                    )
elif args.dataset == 'bandlab':
    new_meta = bandlab_parser(
                    wav_root='../NAS189/homes/allenhung/Project/loop-generation/data/bandlab/BANDLAB_INSTRUMENT/', 
                    meta_path='../NAS189/homes/allenhung/Project/loop-generation/data/bandlab/BANDLAB/bandlab.json',
                    device=device,
                    )

# IO
with open(f'{args.dataset}_meta.json','w') as f:
    f.write(json.dumps(new_meta, indent=4))


