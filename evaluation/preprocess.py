import os
import numpy as np
from multiprocessing import Pool
import pickle


def process_audios(feat_fn):
    feat_fp = os.path.join(feat_dir, f'{feat_fn}.npy')

    if os.path.exists(feat_fp):
        data = np.load(feat_fp)
        if not (np.isnan(data).any() or np.isinf(data).any()):
            return feat_fn, data.shape[-1]
        else:
            return feat_fn, 0
    else:
        return feat_fn, 


if __name__ == "__main__":
    feat_type = ''
    exp_dir = '/home/lego/NAS189/home/looperman/mel_80_256'  # base_out_dir from step2

    out_fp = os.path.join(exp_dir, 'dataset.pkl')

    # ### Process ###
    feat_dir = os.path.join(exp_dir, feat_type)
    feat_fns = [fn.replace('.npy', '') for fn in os.listdir(feat_dir)]
    feat_fns = [f for f in feat_fns if not('std' in f or 'mean' in f)]

    pool = Pool(processes=20)
    dataset = []

    for i, (feat_fn, length) in enumerate(pool.imap_unordered(process_audios, feat_fns), 1):
        print(feat_fn)
        if length == 0:
            continue
        dataset += [(feat_fn, length, )]

    with open(out_fp, 'wb') as f:
        pickle.dump(dataset, f)

    print(len(dataset))
