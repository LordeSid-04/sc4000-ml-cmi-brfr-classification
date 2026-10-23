"""Pre-compute augmented copies of every training recording (run once, CPU, parallel).
Uses the repo's physics-aware augment() (crop, stretch, wrist tilt, yaw, scale, jitter, sensor dropout) and re-featurises,
so training itself stays a pure-GPU loop.   python -m models.train.augment [--data data/processed] [--copies 6]"""
import argparse
from multiprocessing import Pool

import numpy as np

from models.common import load
from preprocessing.augmentation import augment
from preprocessing.dataset import CanonicalStore
from preprocessing.pipeline import featurize
from preprocessing.step5_shaping import fixed_length


def init(folder, cols, L):
    global STORE, COLS, LEN
    STORE, COLS, LEN = CanonicalStore(folder), cols, L


def make(job):
    i, copies = job
    x, n = np.zeros((copies, LEN, len(COLS)), np.float16), np.zeros(copies, np.int64)
    for j in range(copies):
        feats, _ = featurize(augment(STORE.get(i), np.random.default_rng([42, i, j])))
        a, m = fixed_length(feats[COLS].to_numpy(np.float32), LEN)
        x[j], n[j] = np.clip(a, -6e4, 6e4), int(m.sum())          # NaN (sensor off) is kept
    return x, n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--copies', type=int, default=6)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    d, meta, seqs, _, _ = load(a.data)
    with Pool(a.workers, init, (d / 'canonical', meta['feature_columns'], meta['fixed_length'])) as pool:
        res = pool.map(make, [(i, a.copies) for i in range(len(seqs))], chunksize=32)
    N = len(seqs)
    np.save(d / 'aug_x.npy', np.concatenate([np.stack([r[0][j] for r in res]) for j in range(a.copies)]))
    np.save(d / 'aug_len.npy', np.concatenate([[r[1][j] for r in res] for j in range(a.copies)]))
    np.save(d / 'aug_src.npy', np.tile(np.arange(N), a.copies))
    print(f'{N * a.copies} augmented sequences saved to {d}')
