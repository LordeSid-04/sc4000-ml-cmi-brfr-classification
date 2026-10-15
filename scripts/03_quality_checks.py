"""Quality checks on the finished data/processed/ folder (what the modellers receive).

    python scripts/03_quality_checks.py [--processed data/processed]

Every check prints PASS/FAIL; results go to outputs/verification/quality.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.constants import GESTURES  # noqa: E402
from preprocessing.dataset import CanonicalStore, sequence_slices  # noqa: E402
from preprocessing.step5_shaping import Scaler  # noqa: E402

ROT_COLUMNS = ['rot_w', 'angular_vel_x', 'angular_distance', 'rot6d_1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', default='data/processed')
    root = Path(parser.parse_args().processed)
    meta = json.loads((root / 'meta.json').read_text(encoding='utf-8'))
    seqs = pd.read_csv(root / 'sequences.csv')
    z = np.load(root / 'features.npz')
    feats, phase = z['features'], z['phase']
    cols = meta['feature_columns']
    frame = pd.DataFrame(feats, columns=cols)
    slices = sequence_slices(seqs)
    row_seq = np.repeat(np.arange(len(seqs)), seqs['rows'].to_numpy())
    checks = {}

    def check(name, ok, detail=''):
        checks[name] = {'pass': bool(ok), 'detail': detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    check('row count matches source', len(feats) == 574945 and seqs['rows'].sum() == len(feats), f'{len(feats)} rows')
    check('sequence count', len(seqs) == 8151, f'{len(seqs)} sequences')
    check('no infinite values', not np.isinf(feats).any())
    acc_cols = [c for c in cols if c.startswith(('acc_', 'linear_acc', 'gravity_dir', 'jerk', 'acc_cos'))] + ['time_frac']
    check('IMU features always present', not frame[acc_cols].isna().any().any(), f'{len(acc_cols)} columns')

    rot_missing = seqs['rot_missing'].to_numpy()[row_seq] == 1
    nan_rot = frame[ROT_COLUMNS].isna().all(axis=1).to_numpy()
    check('rotation features NaN exactly when rotation is missing', np.array_equal(nan_rot, rot_missing),
          f'{int(rot_missing.sum())} rows without rotation')
    # Short gaps are interpolated, so a sensor column may only be NaN for a WHOLE
    # recording (that sensor delivered nothing), never for part of one.
    sensor_cols = [f'thm_{k}' for k in range(1, 6)] + [f'tof_{k}_mean' for k in range(1, 6)]
    partial, dead = 0, {}
    for c in sensor_cols:
        nan = frame[c].isna().to_numpy()
        per_seq = np.bincount(row_seq, weights=nan, minlength=len(seqs))
        partial += int(((per_seq > 0) & (per_seq < seqs['rows'].to_numpy())).sum())
        dead[c] = int((per_seq == seqs['rows'].to_numpy()).sum())
    check('sensor columns NaN only for whole-recording dropouts (gaps filled)', partial == 0,
          'recordings with a dead sensor: ' + ', '.join(f'{k} {v}' for k, v in dead.items() if v))
    check('ToF proximity in [0, 1]', np.nanmin(frame.filter(like='tof_').filter(regex='_mean$|_max$|_r\\d\\d$').to_numpy()) >= 0
          and np.nanmax(frame.filter(like='tof_').filter(regex='_mean$|_max$|_r\\d\\d$').to_numpy()) <= 1)
    gmag = np.linalg.norm(frame[['gravity_dir_x', 'gravity_dir_y', 'gravity_dir_z']].to_numpy(), axis=1)
    check('gravity direction is a unit vector', np.allclose(gmag, 1, atol=1e-3), f'max deviation {np.abs(gmag - 1).max():.2e}')
    r6 = frame[[f'rot6d_{i}' for i in range(1, 7)]].dropna().to_numpy()
    check('6D rotation columns are orthonormal', np.allclose(np.linalg.norm(r6[:, :3], axis=1), 1, atol=1e-3)
          and np.allclose(np.sum(r6[:, :3] * r6[:, 3:], axis=1), 0, atol=1e-3))

    folds = seqs.groupby('subject')['fold'].nunique()
    check('no subject appears in two folds', (folds == 1).all(), f'{seqs.subject.nunique()} subjects')
    per_fold = seqs.groupby('fold')['gesture'].nunique()
    check('all 18 gestures in every fold', (per_fold == len(GESTURES)).all(), per_fold.to_dict().__repr__())
    share = seqs.groupby('fold')['is_target'].mean()
    check('BFRB share similar across folds', share.max() - share.min() < 0.05, f'{share.min():.3f}-{share.max():.3f}')
    check('one label per recording', seqs['gesture'].notna().all())
    check('phase labels present', set(np.unique(phase)) == {0, 1, 2}, f'counts {np.bincount(phase).tolist()}')
    check('every recording except one has gesture rows', (seqs['gesture_rows'] == 0).sum() <= 1,
          f"{int((seqs['gesture_rows'] == 0).sum())} without")
    check('no train-only column leaks into features', not any(c in cols for c in ['gesture', 'orientation', 'behavior', 'phase', 'sequence_type', 'subject']))

    # Scaler fitted on folds != k: training rows should be ~N(0, 1), flags untouched.
    k = 0
    scaler = Scaler.load(root / 'scalers' / f'fold{k}.json')
    train_rows = np.flatnonzero(seqs['fold'].to_numpy()[row_seq] != k)
    sample = np.random.default_rng(0).choice(train_rows, 50000, replace=False)
    x = scaler.transform(frame.iloc[sample])
    flags = [cols.index(c) for c in ('rot_missing', 'thm_missing', 'tof_missing')]
    other = [i for i in range(len(cols)) if i not in flags]
    check('fold scaler gives mean ~0 on training rows', np.abs(x[:, other].mean(axis=0)).max() < 0.1)
    check('scaled values finite (NaN -> 0)', np.isfinite(x).all())

    store = CanonicalStore(root / 'canonical')
    check('canonical store aligned with sequences.csv', store.sequence_ids == seqs['sequence_id'].tolist())
    s = store.get(0)
    check('canonical quaternions are unit length', np.allclose(np.linalg.norm(s.rot, axis=1), 1, atol=1e-3))

    lengths_ok = all(sl.stop - sl.start == n for sl, n in zip(slices, seqs['rows']))
    check('row slices match recording lengths', lengths_ok)
    check('pipeline config: smoothing off', meta['pipeline_config']['features']['smooth_sigma'] == 0.0)
    check('label-tuned orientation disabled for validated dataset',
          not meta['pipeline_config']['fix_upside_down'] and not meta['pipeline_config']['mirror_left'])

    summary = {'passed': sum(c['pass'] for c in checks.values()), 'total': len(checks), 'checks': checks}
    Path('outputs/verification').mkdir(parents=True, exist_ok=True)
    Path('outputs/verification/quality.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\n{summary['passed']}/{summary['total']} checks passed")
    sys.exit(0 if summary['passed'] == summary['total'] else 1)


if __name__ == '__main__':
    main()
