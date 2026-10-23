"""Train a summary-feature model (5 folds x seeds). Each recording is used twice (all sensors + motion-only copy),
so one model serves both halves of the test set.   python train_trees.py --model lgbm|xgb|rf|mlp [--seeds 0 1 2]"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.common import save_oof, score

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['lgbm', 'xgb', 'rf', 'mlp'])
ap.add_argument('--data', default='data/processed')
ap.add_argument('--seeds', type=int, nargs='+', default=[0])
ap.add_argument('--folds', type=int, nargs='+', default=range(5))
ap.add_argument('--trees', type=int, default=500)
a = ap.parse_args()


def make(seed):
    if a.model == 'lgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(learning_rate=0.05, n_estimators=a.trees, num_leaves=31, min_child_samples=20,
                                  colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                  random_state=seed, verbose=-1)
    if a.model == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=a.trees, learning_rate=0.05, max_depth=6, colsample_bytree=0.3,
                                 subsample=0.8, min_child_weight=2, tree_method='hist', random_state=seed,
                                 device='cuda' if torch.cuda.is_available() else 'cpu')
    if a.model == 'rf':
        return make_pipeline(SimpleImputer(strategy='constant', fill_value=-999),
                             RandomForestClassifier(a.trees, max_features='sqrt', min_samples_leaf=2, n_jobs=-1,
                                                    random_state=seed))
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         MLPClassifier((512, 256), alpha=1e-3, early_stopping=True, max_iter=200, random_state=seed))


d = Path(a.data)
seqs = pd.read_csv(d / 'sequences.csv')
Xf = pd.read_parquet(d / 'tree_full.parquet').drop(columns='sequence_id')
Xi = pd.read_parquet(d / 'tree_imu.parquet').drop(columns='sequence_id')
y, fold = seqs['gesture_id'].to_numpy(), seqs['fold'].to_numpy()
Path(f'runs/{a.model}').mkdir(parents=True, exist_ok=True)
for seed in a.seeds:
    for k in a.folds:
        tr, va = fold != k, fold == k
        m = make(seed).fit(pd.concat([Xf[tr], Xi[tr]]), np.r_[y[tr], y[tr]])
        pf, pi = m.predict_proba(Xf[va]), m.predict_proba(Xi[va])
        save_oof(a.model, 'full', seed, k, pf)
        save_oof(a.model, 'imu', seed, k, pi)
        joblib.dump(m, f'runs/{a.model}/model_s{seed}_f{k}.joblib')
        print(f'seed {seed} fold {k}: full {score(y[va], pf):.4f}  imu {score(y[va], pi):.4f}', flush=True)
