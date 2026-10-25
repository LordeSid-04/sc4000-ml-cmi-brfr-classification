"""Tables, statistics, hybrids and error analysis from the saved out-of-fold predictions (runs/<name>/).

  python -m models.evaluate table rf lgbm xgb mlp cnn gru transformer cnn_gru 'cnn_gru:blank' [--tag main]
  python -m models.evaluate compare cnn_gru lgbm xgb rf     # first model vs each other: paired bootstrap, Holm-corrected
  python -m models.evaluate hybrid cnn_gru lgbm             # h1/h2/h3 + runs/blend_weights.json for the submission
  python -m models.evaluate analysis cnn_gru                # per-class F1, confusion matrix, per-person, per-posture
  python -m models.evaluate seeds cnn_gru lgbm ...          # each seed alone vs the seed average
  python -m models.evaluate breakdown cnn_gru lgbm ...      # binary F1 (BFRB or not) vs macro F1 (which BFRB)
  python -m models.evaluate ensemble NAME a b c             # equal-weight average of runs (no fitting) -> runs/NAME

'name:blank' = the all-sensor model evaluated with its sensors switched off (routing vs one model).
full / imu / combined = competition score on all-sensor inputs, motion-only inputs, and both pooled (ranks models).
CI = 95% subject-level bootstrap (people are the independent units). 'fold0' = outer fold 0 only, which the
tuning step never sees, so it is the clean check for *_tuned models."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from models.common import SID, bootstrap, combined, get, holm, load_oof, pvalue, save_oof, score
from models.hybrids import best_weight, fit_predict
from models.paths import COMPARISON, ERRORS, HYBRIDS, out_dir
from preprocessing.constants import SCORE_CLASSES

D = Path('data/processed')


def setup():
    seqs = pd.read_csv(D / 'sequences.csv')
    return seqs, seqs['gesture_id'].to_numpy(), seqs['fold'].to_numpy(), seqs['subject']


def table(names, seqs, y, fold, subj, tag=None):
    preds = {n: get(n, fold) for n in names}
    bs = bootstrap(y, subj, preds, n=1000)
    f0 = fold == 0
    rows = [dict(model=n, full=score(y, pf), imu=score(y, pi), combined=combined(y, pf, pi),
                 lo=np.percentile(bs[n], 2.5), hi=np.percentile(bs[n], 97.5), fold0=combined(y[f0], pf[f0], pi[f0]))
            for n, (pf, pi) in preds.items()]
    df = pd.DataFrame(rows)
    df['combined [95% CI]'] = df.apply(lambda r: f'{r.combined:.4f} [{r.lo:.4f}, {r.hi:.4f}]', axis=1)
    print(df[['model', 'full', 'imu', 'combined [95% CI]', 'fold0']].round(4).to_string(index=False), flush=True)
    df.round(5).to_csv(out_dir(tag) / f'table_{tag or "_".join(n.replace(":", "-") for n in names)[:80]}.csv', index=False)
    return df


def compare(ref, *others, seqs=None, y=None, fold=None, subj=None, tag=None):
    """Is `ref` better than each other model? Paired subject bootstrap; Holm correction over the comparisons."""
    preds = {n: get(n, fold) for n in (ref, *others)}
    bs = bootstrap(y, subj, preds, n=2000)
    rows = []
    for o in others:
        d = bs[ref] - bs[o]
        wins = sum(combined(y[fold == k], *[p[fold == k] for p in preds[ref]]) >
                   combined(y[fold == k], *[p[fold == k] for p in preds[o]]) for k in range(5))
        rows.append(dict(comparison=f'{ref} - {o}', delta=combined(y, *preds[ref]) - combined(y, *preds[o]),
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=pvalue(d), folds_won=f'{wins}/5'))
    df = pd.DataFrame(rows)
    df['p_holm'] = holm(df['p'])
    df['significant'] = df['p_holm'] < 0.05
    print(df.round(4).to_string(index=False), flush=True)
    df.round(5).to_csv(out_dir(tag) / f'compare_{tag or ref.replace(":", "-")}.csv', index=False)


def hybrid(a, b, seqs, y, fold, subj):
    A, B = get(a, fold), get(b, fold)
    X = [pd.read_parquet(D / f'tree_{t}.parquet').drop(columns='sequence_id') for t in ('full', 'imu')]
    outs = [fit_predict(y, fold, A[s], B[s], X[s]) for s in (0, 1)]
    w = {'full': best_weight(y, A[0], B[0]), 'imu': best_weight(y, A[1], B[1])}
    for p in (Path('runs'), HYBRIDS):                           # runs/ is uploaded together with the models
        p.mkdir(parents=True, exist_ok=True)
        json.dump(w, open(p / 'blend_weights.json', 'w'))
        json.dump({'nn': a, 'gb': b}, open(p / 'deploy.json', 'w'))
    print(f'deployed blend: weight on {a} = {w["full"]:.2f} (all sensors), {w["imu"]:.2f} (motion-only)')
    names = []
    for h in outs[0]:
        n = f'{h}_{a}_{b}'
        names.append(n)
        for k in range(5):
            for kind, s in (('full', 0), ('imu', 1)):
                save_oof(n, kind, 0, k, outs[s][h][fold == k])
    table([a, b] + names, seqs, y, fold, subj, tag='hybrids')
    ea, eb = A[0].argmax(1) == y, B[0].argmax(1) == y
    print(f'complementarity (all sensors): exactly one of {a}/{b} is right on {np.mean(ea != eb):.1%} of recordings, '
          f'both wrong on {np.mean(~ea & ~eb):.1%}')


def analysis(name, seqs, y, fold, subj):
    """Error analysis for the report: where does the model fail, and for whom?"""
    pf, pi = get(name, fold)
    out = ERRORS / name.replace(':', '-')
    out.mkdir(parents=True, exist_ok=True)
    yt = SID[y]
    per_class = pd.DataFrame({s: f1_score(yt, SID[p.argmax(1)], average=None, labels=range(9), zero_division=0)
                              for s, p in (('full', pf), ('imu', pi))}, index=SCORE_CLASSES)
    per_class['imu_drop'] = per_class['full'] - per_class['imu']
    per_class.round(4).to_csv(out / 'per_class_f1.csv')
    for s, p in (('full', pf), ('imu', pi)):
        pd.DataFrame(confusion_matrix(yt, SID[p.argmax(1)], labels=range(9)), SCORE_CLASSES, SCORE_CLASSES) \
            .to_csv(out / f'confusion_{s}.csv')
    per_subject = pd.DataFrame([dict(subject=sj, n=int(m.sum()), combined=combined(y[m], pf[m], pi[m]),
                                     handedness=seqs.loc[m, 'handedness'].iloc[0])
                                for sj in subj.unique() for m in [(subj == sj).to_numpy()]]).sort_values('combined')
    per_subject.round(4).to_csv(out / 'per_subject.csv', index=False)
    if 'orientation' in seqs:
        pd.DataFrame([dict(orientation=o, n=int(m.sum()), full=score(y[m], pf[m]), imu=score(y[m], pi[m]))
                      for o in seqs['orientation'].unique() for m in [(seqs['orientation'] == o).to_numpy()]]) \
            .round(4).to_csv(out / 'by_orientation.csv', index=False)
    print(per_class.round(3).to_string())
    print(f'worst 5 people: {per_subject.head(5)[["subject", "combined"]].round(3).values.tolist()}')
    print(f'saved to {out}/')


def seeds(names, seqs, y, fold, subj):
    """How much does averaging seeds help, and how much do single seeds vary?"""
    rows = []
    for n in names:
        per = []
        for s in sorted({int(p.name.split('_s')[1].split('_')[0]) for p in (Path('runs') / n).glob('full_s*_f0.npy')}):
            pf, pi = np.zeros((len(y), 18)), np.zeros((len(y), 18))
            for k in range(5):
                pf[fold == k] = np.load(Path('runs') / n / f'full_s{s}_f{k}.npy')
                pi[fold == k] = np.load(Path('runs') / n / f'imu_s{s}_f{k}.npy')
            per.append(combined(y, pf, pi))
        ens = combined(y, *get(n, fold))
        rows.append(dict(model=n, seeds=len(per), single_mean=np.mean(per), single_min=np.min(per),
                         single_max=np.max(per), seed_sd=np.std(per, ddof=1) if len(per) > 1 else np.nan,
                         averaged=ens, gain=ens - np.mean(per)))
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    COMPARISON.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(COMPARISON / 'seed_averaging.csv', index=False)


def breakdown(names, seqs, y, fold, subj):
    """The metric's two halves: binary F1 (is it a BFRB?) and macro F1 (which BFRB, 9 classes)."""
    rows = []
    for n in names:
        pf, pi = get(n, fold)
        r = {'model': n}
        for s, p in (('full', pf), ('imu', pi)):
            yt, yp = SID[y], SID[p.argmax(1)]
            r[f'{s}_binary_f1'] = f1_score(yt < 8, yp < 8)
            r[f'{s}_macro_f1'] = f1_score(yt, yp, average='macro', zero_division=0)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    COMPARISON.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(COMPARISON / 'metric_breakdown.csv', index=False)


def ensemble(name, *members, seqs=None, y=None, fold=None, subj=None):
    """Equal-weight average of several runs' seed-averaged predictions. Nothing is fitted, so nothing can leak."""
    for kind in ('full', 'imu'):
        p = np.mean([load_oof(m, kind, fold) for m in members], axis=0)
        for k in range(5):
            save_oof(name, kind, 0, k, p[fold == k])
    print(f'{name} = mean of {", ".join(members)}: combined {combined(y, *get(name, fold)):.4f}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    tag = None
    if '--tag' in args:                                         # --tag NAME -> results/table_NAME.csv etc.
        i = args.index('--tag')
        tag, args = args[i + 1], args[:i] + args[i + 2:]
    seqs, y, fold, subj = setup()
    if cmd == 'table':
        table(args, seqs, y, fold, subj, tag=tag)
    elif cmd == 'compare':
        compare(*args, seqs=seqs, y=y, fold=fold, subj=subj, tag=tag)
    elif cmd in ('seeds', 'breakdown'):
        {'seeds': seeds, 'breakdown': breakdown}[cmd](args, seqs, y, fold, subj)
    elif cmd == 'ensemble':
        ensemble(*args, seqs=seqs, y=y, fold=fold, subj=subj)
    else:
        {'hybrid': hybrid, 'analysis': analysis}[cmd](*args, seqs, y, fold, subj)
