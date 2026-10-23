"""Tables, confidence intervals, hybrids.
  python evaluate.py table rf lgbm xgb mlp cnn gru transformer cnn_gru 'cnn_gru:blank'
  python evaluate.py diff cnn_gru lgbm            # paired subject-bootstrap difference + folds won
  python evaluate.py hybrid cnn_gru lgbm          # H1 blend, H2 stacker, H3 trees on probs+features (all nested over folds)
'name:blank' = the full-sensor model evaluated on motion-only inputs (routing vs one model).  Subject bootstrap, 95% CI."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.common import bootstrap, combined, get, save_oof, score

seqs = pd.read_csv('data/processed/sequences.csv')
y, fold, subj = seqs['gesture_id'].to_numpy(), seqs['fold'].to_numpy(), seqs['subject']
Path('results').mkdir(exist_ok=True)


def table(names):
    preds = {n: get(n, fold) for n in names}
    bs = bootstrap(y, subj, preds)
    rows = [dict(model=n, full=score(y, pf), imu=score(y, pi), combined=combined(y, pf, pi),
                 lo=np.percentile(bs[n], 2.5), hi=np.percentile(bs[n], 97.5)) for n, (pf, pi) in preds.items()]
    df = pd.DataFrame(rows).round(4)
    df['combined [95% CI]'] = df.apply(lambda r: f"{r.combined:.4f} [{r.lo:.4f}, {r.hi:.4f}]", axis=1)
    print(df[['model', 'full', 'imu', 'combined [95% CI]']].to_string(index=False))
    df.to_csv('results/table_' + '_'.join(n.replace(':', '-') for n in names)[:80] + '.csv', index=False)


def diff(a, b):
    pa, pb = get(a, fold), get(b, fold)
    bs = bootstrap(y, subj, {a: pa, b: pb})
    dlt = bs[a] - bs[b]
    wins = sum(combined(y[fold == k], pa[0][fold == k], pa[1][fold == k]) > combined(y[fold == k], pb[0][fold == k], pb[1][fold == k]) for k in range(5))
    print(f'{a} - {b}: {combined(y, *pa) - combined(y, *pb):+.4f}  95% CI [{np.percentile(dlt, 2.5):+.4f}, {np.percentile(dlt, 97.5):+.4f}]  '
          f'{a} wins {wins}/5 folds  -> {"significant" if np.percentile(dlt, 2.5) > 0 or np.percentile(dlt, 97.5) < 0 else "not significant"}')


def hybrid(a, b):
    import lightgbm as lgb
    A, B = get(a, fold), get(b, fold)
    X = [pd.read_parquet(f'data/processed/tree_{t}.parquet').drop(columns='sequence_id') for t in ('full', 'imu')]
    grid = np.linspace(0, 1, 21)
    out = {h: [np.zeros_like(A[0]), np.zeros_like(A[0])] for h in ('h1', 'h2', 'h3')}
    weights = []
    for s in (0, 1):                                       # 0 = all sensors, 1 = motion-only
        pa, pb = A[s], B[s]
        for k in range(5):
            tr, va = fold != k, fold == k
            # choosing weights / fitting stackers only on the other folds' out-of-fold predictions
            w = grid[np.argmax([score(y[tr], w * pa[tr] + (1 - w) * pb[tr]) for w in grid])]
            out['h1'][s][va] = w * pa[va] + (1 - w) * pb[va]
            feats = np.log(np.c_[pa, pb] + 1e-6)
            out['h2'][s][va] = LogisticRegression(C=1.0, max_iter=500).fit(feats[tr], y[tr]).predict_proba(feats[va])
            F = pd.concat([X[s].reset_index(drop=True), pd.DataFrame(pa, columns=[f'nn{i}' for i in range(18)])], axis=1)
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, colsample_bytree=0.3,
                                   subsample=0.8, subsample_freq=1, verbose=-1).fit(F[tr], y[tr])
            out['h3'][s][va] = m.predict_proba(F[va])
        weights.append(float(grid[np.argmax([score(y, w * pa + (1 - w) * pb) for w in grid])]))
    json.dump({'full': weights[0], 'imu': weights[1]}, open('results/blend_weights.json', 'w'))   # deployed H1 weights
    print(f'deployed blend weights on {a} (1 - w on {b}): full {weights[0]:.2f}, imu {weights[1]:.2f}')
    for h, (pf, pi) in out.items():
        for k in range(5):
            save_oof(f'{h}_{a}_{b}'.replace(':', '-'), 'full', 0, k, pf[fold == k])
            save_oof(f'{h}_{a}_{b}'.replace(':', '-'), 'imu', 0, k, pi[fold == k])
    table([a, b] + [f'{h}_{a}_{b}'.replace(':', '-') for h in out])
    ea, eb = A[0].argmax(1) == y, B[0].argmax(1) == y
    print(f'error complementarity (all sensors): exactly one of {a}/{b} right on {np.mean(ea != eb):.1%} of recordings; both wrong {np.mean(~ea & ~eb):.1%}')


cmd, args = sys.argv[1], sys.argv[2:]
{'table': table, 'diff': diff, 'hybrid': hybrid}[cmd](*([args] if cmd == 'table' else args))
