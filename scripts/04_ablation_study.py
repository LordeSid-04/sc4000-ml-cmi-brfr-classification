"""Ablation study with a LightGBM baseline (guide section 9).

    python scripts/04_ablation_study.py [--data PATH] [--rows A B C ...] [--workers 5]

Same model, same 5 subject folds and same seed for every row. Each row is
scored on all-sensor validation and on IMU-only validation (thermopile + ToF
blanked), since about half of the hidden test set is IMU-only.
Results: outputs/ablation/trees.csv and trees.json; features are cached in
outputs/ablation/features/.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.ablation import ALL, cross_validate_lgbm, tree_rows_for_sequence  # noqa: E402
from preprocessing.data_loading import load_train  # noqa: E402
from preprocessing.labels import GESTURE_INDEX  # noqa: E402
from preprocessing.orientation import UPSIDE_DOWN_SUBJECTS  # noqa: E402

OUT = Path('outputs/ablation')


def _work(args):
    """Featurise a chunk of recordings; return compact float32 matrices per experiment."""
    import warnings
    warnings.simplefilter('ignore')
    chunk, hand, names = args
    ids, results = [], []
    for sid, rows in chunk.groupby('sequence_id', sort=True):
        ids.append(sid)
        results.append(tree_rows_for_sequence(rows, hand.get(rows['subject'].iloc[0]), names))
    out = {}
    for n in names:
        cols = [c for c in results[0][n][0] if c != 'sequence_id']
        full = np.array([[r[n][0][c] for c in cols] for r in results], dtype=np.float32)
        imu = np.array([[r[n][1][c] for c in cols] for r in results], dtype=np.float32)
        out[n] = (cols, full, imu)
    return ids, out


def build_features(train, demo, names, workers, chunk_size=60):
    hand = demo.set_index('subject')['handedness'].to_dict()
    ids = train['sequence_id'].unique()
    chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    starts = train.groupby('sequence_id', sort=False).indices
    jobs = (train.iloc[np.concatenate([starts[s] for s in c])] for c in chunks)
    all_ids, parts = [], {n: ([], []) for n in names}
    columns = {}
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (chunk_ids, out) in enumerate(pool.imap(_work, ((j, hand, names) for j in jobs))):
            all_ids.extend(chunk_ids)
            for n in names:
                columns[n] = out[n][0]
                parts[n][0].append(out[n][1])
                parts[n][1].append(out[n][2])
            if (i + 1) % 20 == 0:
                print(f'  features: {(i + 1) * chunk_size} sequences ({time.time() - t0:.0f}s)', flush=True)
    for n in names:
        for k, kind in enumerate(['full', 'imu']):
            frame = pd.DataFrame(np.vstack(parts[n][k]), columns=columns[n])
            frame.insert(0, 'sequence_id', all_ids)
            frame.to_parquet(OUT / 'features' / f'{n}_{kind}.parquet', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--rows', nargs='*', default=list(ALL))
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--folds', default='processed/folds.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    (OUT / 'features').mkdir(parents=True, exist_ok=True)

    missing = [n for n in args.rows if not (OUT / 'features' / f'{n}_imu.parquet').exists()]
    train = demo = None
    if missing:
        train, demo = load_train(args.data)
        train = train.sort_values(['sequence_id', 'sequence_counter'], kind='stable').reset_index(drop=True)
        print(f'building features for {missing}', flush=True)
        build_features(train, demo, missing, args.workers)

    if Path(args.folds).exists():
        folds = pd.read_csv(args.folds)
        labels = pd.read_csv(Path(args.folds).parent / 'sequences.csv')[['sequence_id', 'gesture', 'subject']]
    else:
        from preprocessing.shaping import make_folds
        if train is None:
            train, demo = load_train(args.data)
        labels = train.groupby('sequence_id', sort=True).agg(gesture=('gesture', 'first'), subject=('subject', 'first')).reset_index()
        folds = make_folds(labels, seed=args.seed)
    table = labels.merge(folds[['sequence_id', 'fold']], on='sequence_id').sort_values('sequence_id').reset_index(drop=True)
    y = table['gesture'].map(GESTURE_INDEX).to_numpy()
    fold = table['fold'].to_numpy()

    results_path = OUT / 'trees.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in args.rows:
        full = pd.read_parquet(OUT / 'features' / f'{name}_full.parquet').set_index('sequence_id').loc[table['sequence_id']]
        imu = pd.read_parquet(OUT / 'features' / f'{name}_imu.parquet').set_index('sequence_id').loc[table['sequence_id']]
        print(f'{name}: {ALL[name][0]}  ({full.shape[1]} features)', flush=True)
        t0 = time.time()
        scores, oof_full, oof_imu = cross_validate_lgbm(full.to_numpy(np.float32), imu.to_numpy(np.float32), y, fold, args.seed)
        s = pd.DataFrame(scores)
        results[name] = {
            'description': ALL[name][0], 'n_features': int(full.shape[1]),
            'full_mean': float(s.full.mean()), 'full_std': float(s.full.std()),
            'imu_mean': float(s.imu.mean()), 'imu_std': float(s.imu.std()),
            'full_binary': float(s.full_binary.mean()), 'full_macro': float(s.full_macro.mean()),
            'imu_binary': float(s.imu_binary.mean()), 'imu_macro': float(s.imu_macro.mean()),
            'folds': scores, 'seconds': round(time.time() - t0),
        }
        np.savez_compressed(OUT / f'oof_{name}.npz', full=oof_full, imu=oof_imu, y=y,
                            sequence_id=table['sequence_id'].to_numpy())
        print(f'  -> full {s.full.mean():.4f} ± {s.full.std():.4f} | imu {s.imu.mean():.4f} ± {s.imu.std():.4f}', flush=True)
        results_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    # Extra comparison: drop the upside-down subjects from training instead of fixing them.
    if 'D' in args.rows or 'G' in args.rows:
        name = 'G'
        full = pd.read_parquet(OUT / 'features' / 'G-no_upside_down_full.parquet').set_index('sequence_id').loc[table['sequence_id']] \
            if (OUT / 'features' / 'G-no_upside_down_full.parquet').exists() else None
        if full is not None:
            imu = pd.read_parquet(OUT / 'features' / 'G-no_upside_down_imu.parquet').set_index('sequence_id').loc[table['sequence_id']]
            exclude = table['subject'].isin(UPSIDE_DOWN_SUBJECTS).to_numpy()
            print('G-drop_upside_down: upside-down subjects removed from training', flush=True)
            scores, _, _ = cross_validate_lgbm(full.to_numpy(np.float32), imu.to_numpy(np.float32), y, fold, args.seed, exclude)
            s = pd.DataFrame(scores)
            results['G-drop_upside_down'] = {
                'description': 'full pipeline, upside-down subjects removed from training (not fixed)',
                'n_features': int(full.shape[1]),
                'full_mean': float(s.full.mean()), 'full_std': float(s.full.std()),
                'imu_mean': float(s.imu.mean()), 'imu_std': float(s.imu.std()),
                'full_binary': float(s.full_binary.mean()), 'full_macro': float(s.full_macro.mean()),
                'imu_binary': float(s.imu_binary.mean()), 'imu_macro': float(s.imu_macro.mean()),
                'folds': scores}
            results_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    rows = [{'row': k, **{c: v[c] for c in ['description', 'n_features', 'full_mean', 'full_std', 'imu_mean', 'imu_std',
                                             'full_binary', 'full_macro', 'imu_binary', 'imu_macro']}} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(OUT / 'trees.csv', index=False)
    print(pd.DataFrame(rows)[['row', 'n_features', 'full_mean', 'full_std', 'imu_mean', 'imu_std']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
