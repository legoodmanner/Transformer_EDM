import torch
import numpy as np
import os
import json
from tqdm import tqdm
import librosa
import warnings
import torchaudio

def detect_instrument(audio, threshold=0.05):
    rms = librosa.feature.rms(y=audio)
    mean_energy = np.mean(rms)
    return int(mean_energy >= threshold)

def load_audio(audio_path, device, model_rate=44100):
    wav, sr = librosa.load(audio_path)
    wav = torch.tensor(wav).to(device)
    if len(wav.shape) == 2:
        wav = wav.permute(1,0) if wav.shape[-1] == 2 else wav
    else:
        wav = wav.unsqueeze(0)
    wav = wav.unsqueeze(0)

    if wav.shape[1] == 1:
        # if we have mono, we duplicate it to get stereo
        wav = torch.repeat_interleave(wav, 2, dim=1)

    if sr != model_rate:
        warnings.warn("resample to model sample rate")
        # we have to resample to model samplerate if needed
        # this makes sure we resample input only once
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=model_rate, resampling_method="sinc_interpolation"
        ).to(wav.device)
        wav = resampler(wav)
    return wav

def segment(wav, downbeats):
    downbeats = librosa.time_to_samples(downbeats, 44100)
    if not isinstance(downbeats, np.ndarray):
        downbeats = np.array([downbeats])
    seg_wavs = []
    for i in range(len(downbeats)-1):
        seg_wavs.append(wav[...,downbeats[i]:downbeats[i+1]])
    return seg_wavs

def freesound_parser(device, loop_detect):
    meta_path = 'bandlab_meta.json'
    audio_root = '../NAS189/homes/allenhung/Project/loop-generation/data/bandlab/BANDLAB_INSTRUMENT/'
    output_root = '../NAS189/home/bandlab_v2/'
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxl', device=device)
    separator.sample_rate = torch.tensor([44100])
    type_tag = ['drum', 'bass', 'other', 'vocals']
    new_meta = {}
    with open(meta_path) as f:
        meta = json.load(f)
        for value in tqdm(meta.values()): 
            
            downbeats = value['downbeats']
            try:
                wavs = load_audio(os.path.join(audio_root, value['category'], value['filename']), device)
                seg_wavs = segment(wavs, downbeats)
            except:
                print('error')
                continue
            for i, wav in enumerate(seg_wavs):
                loop_type = []

                if loop_detect:
                    with torch.no_grad():
                        results = separator(wav).squeeze().detach().cpu().numpy()
                    for instrument_track, t in zip(results, type_tag):
                        if detect_instrument(instrument_track):
                            loop_type.append(t)
                
                data = {}
                data['file_id'] = value['file_id'] + f'_{downbeats[i]}_{downbeats[i+1]}'
                data['filename'] = value['filename'].split('.',1)[0] + f'_{downbeats[i]}_{downbeats[i+1]}' + '.' + value['filename'].split('.',1)[1]
                data['genre'] = value['genre']
                data['category'] = value['category']
                data['key'] = value['key']
                data['bpm'] = value['bpm']
                data['calculate_key'] = value['calculate_key']
                data['beats']  = value['beats'] 
                data['downbeats'] = value['downbeats']
                data['calculate_bpm'] = value['calculate_bpm']
                data['loop_type'] = loop_type
                new_meta[data['file_id']] = data
                if not os.path.isfile(os.path.join(output_root, data['filename'])):
                    librosa.output.write_wav(os.path.join(output_root, data['filename']), np.asfortranarray(wav.squeeze().detach().cpu().numpy()), 44100, norm=False)
        return new_meta
            
            
            

device = torch.device('cuda:1')
new_meta = freesound_parser(device, loop_detect=False)
with open('bandlab_meta_v2.json','w') as f:
    f.write(json.dumps(new_meta, indent=4))
