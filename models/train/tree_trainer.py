"""Training loop shared by the summary-feature models (lgbm, xgb, rf, mlp). Each model's file just calls cli(model).
Every recording is used twice (all sensors + motion-only copy), so one model serves both halves of the test set.
Finished (seed, fold) runs are skipped.   python -m models.train.lgbm [--seeds 0 1 2] [--trees 500] [--save]"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.common import oof_path, save_oof, score
from models.registry import TREES


def parse(model, argv=None):
    ap = argparse.ArgumentParser(description=f'train {model} on the summary features')
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--save', action='store_true', help='keep the fitted models (needed only for the deployed model)')
    return ap.parse_args(argv)


def train(model, a):
    todo = [(s, k) for s in a.seeds for k in a.folds
            if not all(oof_path(model, kd, s, k).exists() for kd in ('full', 'imu'))]
    if not todo:
        print(f'{model}: all requested runs already done', flush=True)
        return
    d = Path(a.data)
    seqs = pd.read_csv(d / 'sequences.csv')
    Xf = pd.read_parquet(d / 'tree_full.parquet').drop(columns='sequence_id')
    Xi = pd.read_parquet(d / 'tree_imu.parquet').drop(columns='sequence_id')
    y, fold = seqs['gesture_id'].to_numpy(), seqs['fold'].to_numpy()
    for seed, k in todo:
        tr, va = fold != k, fold == k
        m = TREES[model](seed, a.trees).fit(pd.concat([Xf[tr], Xi[tr]]), np.r_[y[tr], y[tr]])
        pf, pi = m.predict_proba(Xf[va]), m.predict_proba(Xi[va])
        save_oof(model, 'full', seed, k, pf)
        save_oof(model, 'imu', seed, k, pi)
        if a.save:
            joblib.dump(m, Path('runs') / model / f'model_s{seed}_f{k}.joblib')
        print(f'{model} seed {seed} fold {k}: full {score(y[va], pf):.4f}  imu {score(y[va], pi):.4f}', flush=True)


def cli(model):
    train(model, parse(model))
