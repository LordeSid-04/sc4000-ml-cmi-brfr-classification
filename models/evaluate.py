"""Tables, confidence intervals and hybrids from the saved out-of-fold predictions (runs/<name>/).
  python -m models.evaluate table rf lgbm xgb mlp cnn gru transformer cnn_gru 'cnn_gru:blank'
  python -m models.evaluate diff cnn_gru lgbm        # paired subject-bootstrap difference + folds won
  python -m models.evaluate hybrid cnn_gru lgbm      # h1/h2/h3 + writes runs/blend_weights.json for the submission
'name:blank' = the all-sensor model evaluated with its sensors switched off (routing vs one model).
Scores: full / imu / combined (competition score over both versions pooled); CI = 95% subject bootstrap."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models.common import bootstrap, combined, get, save_oof, score
from models.hybrids import best_weight, fit_predict

D = Path('data/processed')


def setup():
    seqs = pd.read_csv(D / 'sequences.csv')
    return seqs['gesture_id'].to_numpy(), seqs['fold'].to_numpy(), seqs['subject']


def table(names, y, fold, subj, save=True):
    preds = {n: get(n, fold) for n in names}
    bs = bootstrap(y, subj, preds)
    rows = [dict(model=n, full=score(y, pf), imu=score(y, pi), combined=combined(y, pf, pi),
                 lo=np.percentile(bs[n], 2.5), hi=np.percentile(bs[n], 97.5)) for n, (pf, pi) in preds.items()]
    df = pd.DataFrame(rows)
    df['combined [95% CI]'] = df.apply(lambda r: f"{r.combined:.4f} [{r.lo:.4f}, {r.hi:.4f}]", axis=1)
    df = df.round(4)
    print(df[['model', 'full', 'imu', 'combined [95% CI]']].to_string(index=False), flush=True)
    if save:
        Path('results').mkdir(exist_ok=True)
        df.to_csv('results/table_' + '_'.join(n.replace(':', '-') for n in names)[:80] + '.csv', index=False)


def diff(a, b, y, fold, subj):
    pa, pb = get(a, fold), get(b, fold)
    bs = bootstrap(y, subj, {a: pa, b: pb})
    d = bs[a] - bs[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    wins = sum(combined(y[fold == k], pa[0][fold == k], pa[1][fold == k]) >
               combined(y[fold == k], pb[0][fold == k], pb[1][fold == k]) for k in range(5))
    print(f'{a} - {b}: {combined(y, *pa) - combined(y, *pb):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  '
          f'{a} wins {wins}/5 folds -> {"significant" if lo > 0 or hi < 0 else "not significant"}')


def hybrid(a, b, y, fold, subj):
    A, B = get(a, fold), get(b, fold)
    X = [pd.read_parquet(D / f'tree_{t}.parquet').drop(columns='sequence_id') for t in ('full', 'imu')]
    outs = [fit_predict(y, fold, A[s], B[s], X[s]) for s in (0, 1)]
    w = {'full': best_weight(y, A[0], B[0]), 'imu': best_weight(y, A[1], B[1])}
    for p in (Path('runs'), Path('results')):                      # runs/ is what gets uploaded with the models
        p.mkdir(exist_ok=True)
        json.dump(w, open(p / 'blend_weights.json', 'w'))
    print(f'deployed blend: weight on {a} = {w["full"]:.2f} (all sensors), {w["imu"]:.2f} (motion-only)')
    names = []
    for h in outs[0]:
        n = f'{h}_{a}_{b}'.replace(':', '-')
        names.append(n)
        for k in range(5):
            for kind, s in (('full', 0), ('imu', 1)):
                save_oof(n, kind, 0, k, outs[s][h][fold == k])
    table([a, b] + names, y, fold, subj)
    ea, eb = A[0].argmax(1) == y, B[0].argmax(1) == y
    print(f'complementarity (all sensors): exactly one of {a}/{b} is right on {np.mean(ea != eb):.1%} of recordings, '
          f'both wrong on {np.mean(~ea & ~eb):.1%}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    ctx = setup()
    if cmd == 'table':
        table(args, *ctx)
    else:
        {'diff': diff, 'hybrid': hybrid}[cmd](*args, *ctx)
