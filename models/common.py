"""Shared helpers: data shaping, GPU scaling, the competition metric, OOF files, subject bootstrap."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from preprocessing.constants import GESTURES
from preprocessing.step4_labels import SCORE_INDEX, score_class

SID = np.array([SCORE_INDEX[score_class(g)] for g in GESTURES])   # gesture id -> score class id (0-7 BFRB, 8 other)


# ---- data -------------------------------------------------------------------------------------------
def load(data='data/processed'):
    d = Path(data)
    meta = json.loads((d / 'meta.json').read_text())
    z = np.load(d / 'features.npz')
    return d, meta, pd.read_csv(d / 'sequences.csv'), z['features'], z['phase']


def shape(feats, phase, seqs, L):
    """Every recording -> right-aligned (N, L, F) raw features (NaN = sensor missing, 0 = padding),
    lengths (N,) and per-row phases (N, L) with -100 on padding."""
    X = np.zeros((len(seqs), L, feats.shape[1]), np.float32)
    P = np.full((len(seqs), L), -100, np.int64)
    for i, (a, r) in enumerate(zip(seqs['start_row'], seqs['rows'])):
        k = min(r, L)
        X[i, L - k:] = feats[a + r - k:a + r]
        P[i, L - k:] = phase[a + r - k:a + r]
    return X, np.minimum(seqs['rows'].to_numpy(), L), P


def groups(cols):
    """Column indices of the motion group, the thermopile/ToF group, and the two availability flags."""
    prox = [i for i, c in enumerate(cols) if c.startswith(('thm_', 'tof_'))]
    flags = [i for i, c in enumerate(cols) if c in ('thm_missing', 'tof_missing')]
    return [i for i in range(len(cols)) if i not in prox], prox, flags


def scaler(d, k, dev, cols):
    s = json.loads((Path(d) / 'scalers' / f'fold{k}.json').read_text())
    assert s['columns'] == cols
    return (torch.tensor(s['mean'], dtype=torch.float32, device=dev),
            torch.tensor(s['std'], dtype=torch.float32, device=dev))


def fit_scaler(feats, seqs, train_mask, cols):
    """Scaler fitted on the full-length rows of the training recordings only (same rule as Scaler.fit).
    Used for inner tuning splits, which have no stored scaler."""
    assert seqs['rows'].sum() == len(feats) and (seqs['start_row'].diff().fillna(0) >= 0).all()
    v = feats[np.repeat(np.asarray(train_mask), seqs['rows'].to_numpy())]     # rows are stored in sequence order
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean, std = np.nan_to_num(np.nanmean(v, 0)), np.nan_to_num(np.nanstd(v, 0))
    std = np.where(std > 1e-8, std, 1.0)
    for i, c in enumerate(cols):
        if c in ('rot_missing', 'thm_missing', 'tof_missing'):
            mean[i], std[i] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def prep(x, ln, mean, std):
    """Raw (B, L, F) -> scaled network input and mask. Identical to Scaler.transform + fixed_length."""
    L = x.shape[1]
    m = (torch.arange(L, device=x.device)[None] >= (L - ln)[:, None]).float()
    z = torch.nan_to_num((x.float() - mean) / std, nan=0., posinf=0., neginf=0.)
    return z * m[..., None], m


def blank(z, m, prox, flags):
    """Make an input look like a motion-only recording (what a missing sensor produces)."""
    z = z.clone()
    z[..., prox] = 0
    z[..., flags] = m[..., None]
    return z


# ---- metric -----------------------------------------------------------------------------------------
def score_ids(yt, yp):
    """0.5 * binary F1 + 0.5 * macro F1 over the 9 score classes (inputs are score-class ids)."""
    return 0.5 * f1_score(yt < 8, yp < 8) + 0.5 * f1_score(yt, yp, average='macro', zero_division=0)


def score(y, p):
    return score_ids(SID[y], SID[p.argmax(1)])


def combined(y, pf, pi):
    """Competition score over the full-sensor and motion-only versions of the same recordings, pooled."""
    return score_ids(SID[np.r_[y, y]], SID[np.r_[pf.argmax(1), pi.argmax(1)]])


# ---- out-of-fold files: runs/<name>/<kind>_s<seed>_f<fold>.npy (probabilities of that fold's rows) ----
def oof_path(name, kind, seed, k):
    return Path('runs') / name / f'{kind}_s{seed}_f{k}.npy'


def save_oof(name, kind, seed, k, probs):
    p = oof_path(name, kind, seed, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, probs)


def complete(name, kinds=('full', 'imu')):
    """True when every fold has predictions of every kind (i.e. the run finished)."""
    return all(list((Path('runs') / name).glob(f'{kd}_s*_f{k}.npy')) for kd in kinds for k in range(5))


def load_oof(name, kind, fold):
    """(N, 18) out-of-fold probabilities, averaged over the seeds that were run."""
    out = np.zeros((len(fold), 18))
    for k in range(5):
        files = sorted((Path('runs') / name).glob(f'{kind}_s*_f{k}.npy'))
        assert files, f'missing runs/{name}/{kind}_s*_f{k}.npy'
        out[fold == k] = np.mean([np.load(f) for f in files], axis=0)
    return out


def get(name, fold):
    """(full-sensor probs, motion-only probs). 'name:blank' uses the full model with its sensors blanked."""
    n, _, alt = name.partition(':')
    return load_oof(n, 'full', fold), load_oof(n, alt or 'imu', fold)


# ---- uncertainty and multiple comparisons -----------------------------------------------------------
def pvalue(d):
    """Two-sided bootstrap p-value that a paired difference is zero."""
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(p):
    """Holm-Bonferroni adjusted p-values (controls the family-wise error over several comparisons)."""
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj = np.maximum.accumulate([min(1.0, (m - i) * p[j]) for i, j in enumerate(order)])
    out = np.empty(m)
    out[order] = adj
    return out


def bootstrap(y, subject, preds, n=300, seed=0):
    """Subject-level bootstrap (people are the independent units). preds = {name: (pf, pi)}.
    All models are scored on the same resamples, so differences are paired."""
    idx = pd.Series(np.arange(len(y))).groupby(np.asarray(subject)).apply(np.array).tolist()
    rng = np.random.default_rng(seed)
    out = {k: [] for k in preds}
    for _ in range(n):
        rows = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        for k, (pf, pi) in preds.items():
            out[k].append(combined(y[rows], pf[rows], pi[rows]))
    return {k: np.array(v) for k, v in out.items()}
