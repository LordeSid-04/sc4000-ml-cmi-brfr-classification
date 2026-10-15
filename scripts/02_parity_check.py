"""Parity test (guide step 13): the Kaggle predict() path must produce exactly the
features the models were trained on.

For a sample of training sequences, this rebuilds features the way Kaggle
would call us (one polars sequence without train-only columns + a one-row
polars demographics frame) and compares them with data/processed/features.npz.
It also runs the real test.csv examples end to end.

    python scripts/02_parity_check.py [--data PATH] [--n 300]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.constants import TRAIN_ONLY  # noqa: E402
from preprocessing.dataset import sequence_slices  # noqa: E402
from preprocessing.step3_features import FeatureConfig  # noqa: E402
from preprocessing.data_loading import load_test, load_train  # noqa: E402
from preprocessing.pipeline import PipelineConfig, preprocess  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--processed', default='data/processed')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    import polars as pl

    root = Path(args.processed)
    meta = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
    cfg = meta['pipeline_config']
    config = PipelineConfig(continuity=cfg['continuity'], tof_no_object=cfg['tof_no_object'],
                            fix_upside_down=cfg['fix_upside_down'], mirror_left=cfg['mirror_left'],
                            features=FeatureConfig(**cfg['features']))
    seqs = pd.read_csv(root / 'sequences.csv')
    stored = np.load(root / 'features.npz')['features']
    slices = sequence_slices(seqs)

    train, demo = load_train(args.data)
    rng = np.random.default_rng(args.seed)
    # Always include the awkward cases: missing rotation, IMU-only, upside-down and left-handed.
    special = pd.concat([seqs[seqs.rot_missing == 1].head(20), seqs[seqs.tof_missing == 1].head(20),
                         seqs[seqs.subject.isin(meta['upside_down_subjects'])].head(20),
                         seqs[seqs.handedness == 0].head(20)])
    pick = sorted(set(special.index) | set(rng.choice(len(seqs), min(args.n, len(seqs)), replace=False)))
    groups = train.groupby('sequence_id')
    worst = 0.0
    mismatched = []
    for i in pick:
        sid = seqs.loc[i, 'sequence_id']
        rows = groups.get_group(sid).drop(columns=[c for c in TRAIN_ONLY if c in train]).sample(frac=1, random_state=0)
        subject = rows['subject'].iloc[0]
        out = preprocess(pl.from_pandas(rows), pl.from_pandas(demo[demo.subject == subject]), config)
        a = out.features[meta['feature_columns']].to_numpy(np.float32)
        b = stored[slices[i]]
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0, np.abs(a - b))
        if a.shape != b.shape or np.isnan(diff).any() or diff.max() > 1e-5:
            mismatched.append(sid)
        else:
            worst = max(worst, float(diff.max()))
    test, test_demo = load_test(args.data)
    test_ok = 0
    for sid, rows in test.groupby('sequence_id'):
        subject = rows['subject'].iloc[0]
        f = preprocess(pl.from_pandas(rows), pl.from_pandas(test_demo[test_demo.subject == subject]), config).features
        test_ok += int(list(f.columns) == meta['feature_columns'] and len(f) == len(rows))
    result = {'checked': len(pick), 'mismatched': mismatched, 'max_abs_difference': worst,
              'test_sequences_ok': f'{test_ok}/{test.sequence_id.nunique()}'}
    print(json.dumps(result, indent=2))
    Path('outputs/verification').mkdir(parents=True, exist_ok=True)
    Path('outputs/verification/parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    sys.exit(1 if mismatched or test_ok != test.sequence_id.nunique() else 0)


if __name__ == '__main__':
    main()
