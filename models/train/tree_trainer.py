"""Training loop shared by the summary-feature models (lgbm, xgb, rf, mlp). Each model's file just calls cli(model).
Every recording is used twice (all sensors + motion-only copy), so one model serves both halves of the test set.
Finished (seed, fold) runs are skipped.
  python -m models.train.lgbm [--seeds 0 1 2] [--params results/02_hyperparameter_tuning/params/lgbm.json --name lgbm_tuned] [--save]"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.common import oof_path, save_oof, score
from models.registry import TREES


def parse(model, argv=None):
    ap = argparse.ArgumentParser(description=f'train {model} on the summary features')
    ap.add_argument('--name', default=model, help='run name (runs/<name>/)')
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--params', help='JSON written by models.train.tune')
    ap.add_argument('--shuffle-labels', action='store_true', help='sanity check: train on permuted labels')
    ap.add_argument('--save', action='store_true', help='keep the fitted models (needed only for the deployed model)')
    a = ap.parse_args(argv)
    a.kw = json.load(open(a.params))['params'] if a.params else {}
    return a


def tables(data):
    d = Path(data)
    seqs = pd.read_csv(d / 'sequences.csv')
    Xf = pd.read_parquet(d / 'tree_full.parquet').drop(columns='sequence_id')
    Xi = pd.read_parquet(d / 'tree_imu.parquet').drop(columns='sequence_id')
    return seqs, Xf, Xi


def fit(model, seed, trees, kw, Xf, Xi, y, tr, va, shuffle=False):
    """Fit on the training rows (all-sensor + motion-only copies), return (model, full probs, motion-only probs)."""
    yt = np.random.default_rng(seed).permutation(y[tr]) if shuffle else y[tr]
    m = TREES[model](seed, trees, **kw).fit(pd.concat([Xf[tr], Xi[tr]]), np.r_[yt, yt])
    return m, m.predict_proba(Xf[va]), m.predict_proba(Xi[va])


def train(model, a):
    todo = [(s, k) for s in a.seeds for k in a.folds
            if not all(oof_path(a.name, kd, s, k).exists() for kd in ('full', 'imu'))]
    if not todo:
        print(f'{a.name}: all requested runs already done', flush=True)
        return
    seqs, Xf, Xi = tables(a.data)
    y, fold = seqs['gesture_id'].to_numpy(), seqs['fold'].to_numpy()
    print(f'{a.name}: {len(todo)} runs, params {a.kw or "defaults"}', flush=True)
    for seed, k in todo:
        m, pf, pi = fit(model, seed, a.trees, a.kw, Xf, Xi, y, fold != k, fold == k, a.shuffle_labels)
        save_oof(a.name, 'full', seed, k, pf)
        save_oof(a.name, 'imu', seed, k, pi)
        if a.save:
            joblib.dump(m, Path('runs') / a.name / f'model_s{seed}_f{k}.joblib')
        print(f'{a.name} seed {seed} fold {k}: full {score(y[fold == k], pf):.4f}  imu {score(y[fold == k], pi):.4f}',
              flush=True)


def cli(model):
    train(model, parse(model))
