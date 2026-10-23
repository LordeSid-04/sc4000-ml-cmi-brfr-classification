"""Nested hyper-parameter tuning with Optuna. Reported scores stay honest:

  * tuning only sees the TRAINING people of outer fold 0 (outer folds 1-4), split into --inner subject-grouped folds;
  * outer fold 0's people are never seen while tuning, so fold 0 is an untouched check of the tuned settings;
  * inner splits get their own scaler fitted on inner-training recordings only;
  * the objective is the combined score (all-sensor + motion-only pooled); networks are tuned on the all-sensor
    model (its motion-only score comes from switching its sensors off) and the settings are reused for both variants;
  * Optuna's median pruner stops clearly bad trials after their first inner fold.

  python -m models.train.tune --model cnn_gru --trials 12      # GPU
  python -m models.train.tune --model lgbm --trials 20         # CPU
Writes results/params/<model>.json (best settings, read by the trainers via --params) and <model>_trials.csv."""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold

from models.common import combined, fit_scaler
from models.registry import NETS, TREES

SPACE = {   # trial -> settings
    'net': lambda t: dict(lr=t.suggest_float('lr', 5e-4, 4e-3, log=True), epochs=t.suggest_categorical('epochs', [30, 40, 60]),
                          phase=t.suggest_categorical('phase', [0.0, 0.1, 0.3, 0.5]), drop=t.suggest_float('drop', 0.1, 0.4),
                          d=t.suggest_categorical('d', [128, 192, 256])),
    'lgbm': lambda t: dict(num_leaves=t.suggest_int('num_leaves', 15, 127, log=True),
                           learning_rate=t.suggest_float('learning_rate', 0.02, 0.1, log=True),
                           n_estimators=t.suggest_int('n_estimators', 200, 1000, step=100),
                           colsample_bytree=t.suggest_float('colsample_bytree', 0.1, 0.6),
                           min_child_samples=t.suggest_int('min_child_samples', 10, 100, log=True),
                           reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True)),
    'xgb': lambda t: dict(max_depth=t.suggest_int('max_depth', 3, 10),
                          learning_rate=t.suggest_float('learning_rate', 0.02, 0.1, log=True),
                          n_estimators=t.suggest_int('n_estimators', 200, 1000, step=100),
                          colsample_bytree=t.suggest_float('colsample_bytree', 0.1, 0.6),
                          min_child_weight=t.suggest_int('min_child_weight', 1, 20, log=True),
                          reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True)),
}


def inner_splits(seqs, n):
    """Subject-grouped, gesture-stratified folds over the training people of outer fold 0 (original indices)."""
    pool = np.where(seqs['fold'].to_numpy() != 0)[0]
    cv = StratifiedGroupKFold(n, shuffle=True, random_state=0)
    return [(pool[a], pool[b]) for a, b in cv.split(pool, seqs['gesture'].iloc[pool], seqs['subject'].iloc[pool])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=list(NETS) + ['lgbm', 'xgb'])
    ap.add_argument('--trials', type=int, default=None, help='default: 12 for networks, 20 for trees')
    ap.add_argument('--inner', type=int, default=3)
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--epochs', type=int, help='fix the epoch count instead of tuning it (quick tests only)')
    a = ap.parse_args()
    is_net = a.model in NETS
    trials = a.trials or (12 if is_net else 20)

    if is_net:
        import torch
        from models.train.nn_trainer import HP, Data
        D = Data(a.data)
        seqs, y = D.seqs, D.seqs['gesture_id'].to_numpy()
    else:
        from models.train.tree_trainer import fit, tables
        seqs, Xf, Xi = tables(a.data)
        y = seqs['gesture_id'].to_numpy()
    splits = inner_splits(seqs, a.inner)

    def objective(trial):
        scores = []
        for step, (tr, va) in enumerate(splits):
            if is_net:
                hp = {**HP, **SPACE['net'](trial), **({'epochs': a.epochs} if a.epochs else {})}
                mask = torch.zeros(len(seqs), dtype=torch.bool, device=D.dev)
                mask[torch.as_tensor(tr, device=D.dev)] = True
                mean, std = fit_scaler(D.feats, seqs, mask.cpu().numpy(), D.cols)
                _, res = D.fit(a.model, mask, torch.as_tensor(va, device=D.dev), mean, std, 0, hp)
                pf, pi = res['full'], res['blank']
            else:
                trm, vam = np.isin(np.arange(len(seqs)), tr), np.isin(np.arange(len(seqs)), va)
                _, pf, pi = fit(a.model, 0, 500, SPACE[a.model](trial), Xf, Xi, y, trm, vam)
            scores.append(combined(y[va], pf, pi))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(objective, n_trials=trials)
    out = Path('results/params')
    out.mkdir(parents=True, exist_ok=True)
    best = {**HP, **study.best_params} if is_net else study.best_params
    json.dump({'model': a.model, 'params': best, 'inner_combined_score': study.best_value, 'trials': trials,
               'protocol': f'Optuna TPE + median pruning, {a.inner} subject-grouped inner folds over outer folds 1-4; '
                           'outer fold 0 untouched'}, open(out / f'{a.model}.json', 'w'), indent=2)
    study.trials_dataframe().to_csv(out / f'{a.model}_trials.csv', index=False)
    print(f'{a.model}: best inner combined score {study.best_value:.4f} with {best}')


if __name__ == '__main__':
    main()
