"""Evaluate the conservative pipeline with the fixed subject-grouped folds.

This is an out-of-fold reference result, not an untouched final test score.
The feature switches below must stay fixed for this run. Repeated tuning on
these folds still requires a separate final evaluation.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.constants import GESTURES
from preprocessing.step4_labels import competition_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', default='data/processed')
    args = parser.parse_args()
    import lightgbm as lgb
    root = Path(args.processed)
    meta = json.loads((root / 'meta.json').read_text())
    config = meta['pipeline_config']
    if config['mirror_left'] or config['fix_upside_down']:
        raise ValueError('Rebuild with label-tuned orientation disabled before validation')
    seqs = pd.read_csv(root / 'sequences.csv')
    if not (seqs.groupby('subject').fold.nunique() == 1).all():
        raise ValueError('Subject leakage in folds')
    tables = {}
    for name in ('full', 'imu'):
        table = pd.read_parquet(root / f'tree_{name}.parquet')
        if table.sequence_id.tolist() != seqs.sequence_id.tolist():
            raise ValueError('Feature and label ordering mismatch')
        tables[name] = table.drop(columns='sequence_id').to_numpy()
    params = dict(learning_rate=0.05, n_estimators=500, num_leaves=31,
                  min_child_samples=20, colsample_bytree=0.3, subsample=0.8,
                  subsample_freq=1, reg_lambda=1., random_state=42,
                  verbose=-1, n_jobs=6)
    y = seqs.gesture_id.to_numpy()
    folds = seqs.fold.to_numpy()
    scores = []
    oof = {name: np.empty(len(y), dtype=int) for name in tables}
    for k in sorted(np.unique(folds)):
        tr, va = folds != k, folds == k
        model = lgb.LGBMClassifier(**params)
        model.fit(np.vstack([tables['full'][tr], tables['imu'][tr]]), np.r_[y[tr], y[tr]])
        row = {'fold': int(k)}
        for name, x in tables.items():
            pred = model.predict(x[va]).astype(int)
            oof[name][va] = pred
            score, binary, macro = competition_score(seqs.loc[va, 'gesture'], [GESTURES[i] for i in pred])
            row[name] = {'score': score, 'binary_f1': binary, 'macro_f1': macro}
        scores.append(row)
        print(json.dumps(row), flush=True)
    result = {'pipeline_config': config, 'model_params': params,
              'lightgbm_version': lgb.__version__, 'folds': scores,
              'orientation': 'global label-tuned corrections disabled',
              'caveat': 'Fixed grouped-CV reference; not a fresh test set or nested tuning estimate.',
              'summary': {}}
    for name in tables:
        values = [s[name]['score'] for s in scores]
        pooled, _, _ = competition_score(seqs.gesture, [GESTURES[i] for i in oof[name]])
        result['summary'][name] = {'mean_fold_score': float(np.mean(values)),
                                   'fold_std': float(np.std(values)), 'pooled_oof_score': pooled}
    (root / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
