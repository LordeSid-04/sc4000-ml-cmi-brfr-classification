"""Build the processed training set that every model on the team uses.

    python scripts/02_build_dataset.py [--data PATH] [--out processed]

Outputs (in --out):
  sequences.csv        one row per recording: ids, labels, fold, flags, row slice
  features.npz         per-row features (float32) + per-row phase label
  tree_full.parquet    one summary row per recording (all sensors)
  tree_imu.parquet     same, with thermopile + ToF blanked (the IMU-only test half)
  canonical/           cleaned + oriented raw signals (.npy), for on-the-fly augmentation
  folds.csv            the shared 5-fold subject split
  scalers/             per-fold and all-data standardisation statistics
  meta.json            feature names, fixed length L, label maps, pipeline config
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.constants import GESTURES, SCORE_CLASSES  # noqa: E402
from preprocessing.dataset import process_training_set, save_canonical  # noqa: E402
from preprocessing.data_loading import load_train  # noqa: E402
from preprocessing.labels import PHASES  # noqa: E402
from preprocessing.orientation import MIRROR, UPSIDE_DOWN, UPSIDE_DOWN_SUBJECTS  # noqa: E402
from preprocessing.pipeline import PipelineConfig  # noqa: E402
from preprocessing.shaping import Scaler, make_folds  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--out', default='processed')
    parser.add_argument('--length', type=int, default=0, help='fixed length L (0 = 95th percentile, rounded up to 8)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scalers').mkdir(exist_ok=True)

    t0 = time.time()
    train, demo = load_train(args.data)
    train = train.sort_values(['sequence_id', 'sequence_counter'], kind='stable').reset_index(drop=True)
    print(f'loaded {train.shape} in {time.time() - t0:.0f}s', flush=True)

    config = PipelineConfig()
    result = process_training_set(train, demo, config, keep_canonical=True)
    del train
    seqs = result['sequences']
    folds = make_folds(seqs, n_splits=5, seed=args.seed)
    seqs = seqs.merge(folds[['sequence_id', 'fold']], on='sequence_id')
    seqs['start_row'] = np.r_[0, np.cumsum(seqs['rows'].to_numpy())[:-1]]
    seqs.to_csv(out / 'sequences.csv', index=False)
    folds.to_csv(out / 'folds.csv', index=False)

    columns = list(result['features'][0].columns)
    features = np.concatenate([f.to_numpy(dtype=np.float32) for f in result['features']])
    phase = np.concatenate(result['phase']).astype(np.int8)
    np.savez_compressed(out / 'features.npz', features=features, phase=phase)
    pd.DataFrame(result['tree_full']).to_parquet(out / 'tree_full.parquet', index=False)
    pd.DataFrame(result['tree_imu']).to_parquet(out / 'tree_imu.parquet', index=False)
    save_canonical(out / 'canonical', result['canonical'])

    # Scalers: one per fold (fitted on the other four folds only) + one on everything.
    frame = pd.DataFrame(features, columns=columns)
    row_fold = np.repeat(seqs['fold'].to_numpy(), seqs['rows'].to_numpy())
    for k in range(5):
        Scaler().fit(frame[row_fold != k]).save(out / 'scalers' / f'fold{k}.json')
    Scaler().fit(frame).save(out / 'scalers' / 'all.json')

    lengths = seqs['rows']
    length = args.length or int(np.ceil(lengths.quantile(0.95) / 8) * 8)
    meta = {
        'feature_columns': columns,
        'fixed_length': length,
        'length_kept_whole': float((lengths <= length).mean()),
        'gestures': GESTURES, 'score_classes': SCORE_CLASSES, 'phases': PHASES,
        'pipeline_config': dataclasses.asdict(config),
        'mirror': dataclasses.asdict(MIRROR), 'upside_down': dataclasses.asdict(UPSIDE_DOWN),
        'upside_down_subjects': list(UPSIDE_DOWN_SUBJECTS),
        'sequences': int(len(seqs)), 'rows': int(len(features)),
        'sequences_with_issues': int((seqs['issues'].fillna('') != '').sum()),
        'seed': args.seed,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps({k: meta[k] for k in ['fixed_length', 'length_kept_whole', 'sequences', 'rows', 'sequences_with_issues']}, indent=2))
    print(f'{len(columns)} per-row features; {pd.read_parquet(out / "tree_full.parquet").shape[1] - 1} tree features')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
