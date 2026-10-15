"""Batch processing of the training set and compact storage of canonical recordings.

The canonical store keeps the cleaned, oriented raw signals (float32, ToF in
float16) so neural-network training can augment and recompute features on the
fly without re-reading the 1 GB CSV.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .augmentation import drop_sensors
from .step1_cleaning import CleanSequence, DataError
from .step4_labels import row_phase, sequence_labels
from .pipeline import PipelineConfig, canonical, featurize
from .step5_shaping import tree_summary


def process_training_set(train, demographics, config=PipelineConfig(), keep_canonical=False,
                         imu_only=True, summaries=True, log_every=1000):
    """Run the full pipeline on every training sequence.

    Returns a dict with per-sequence table, per-row features (list of frames),
    per-row phases, tree summaries (full and IMU-only) and optionally the
    canonical recordings.
    """
    hand = demographics.set_index('subject')['handedness']
    out = {'sequences': [], 'features': [], 'phase': [], 'tree_full': [], 'tree_imu': [],
           'canonical': []}
    start = time.time()
    for i, (sid, rows) in enumerate(train.groupby('sequence_id', sort=True)):
        try:
            seq = canonical(rows, demographics, config, strict=True)
            issue = ''
        except DataError as exc:  # audit it, then repair like inference would
            seq = canonical(rows, demographics, config, strict=False)
            issue = f'{exc}; ' + '; '.join(seq.issues)
        features, _ = featurize(seq, config)
        subject = rows['subject'].iloc[0]
        record = {'sequence_id': sid, 'subject': subject,
                  'handedness': int(hand.get(subject, -1)),
                  'rows': seq.n, 'rot_missing': int(seq.rot_missing),
                  'thm_missing': int(seq.thm_missing), 'tof_missing': int(seq.tof_missing),
                  'issues': issue}
        record.update(sequence_labels(seq.meta))
        phase = row_phase(seq.meta)
        record['gesture_rows'] = int((phase == 2).sum()) if phase is not None else -1
        out['sequences'].append(record)
        out['features'].append(features)
        out['phase'].append(phase if phase is not None else np.zeros(seq.n, np.int8))
        if summaries:
            out['tree_full'].append({'sequence_id': sid, **tree_summary(features)})
            if imu_only:
                blank = drop_sensors(seq, True, False)
                out['tree_imu'].append({'sequence_id': sid, **tree_summary(featurize(blank, config)[0])})
        if keep_canonical:
            compact = seq.copy()
            for name, dtype in _ARRAYS.items():
                setattr(compact, name, getattr(seq, name).astype(dtype))
            compact.meta = compact.meta.iloc[:0]
            out['canonical'].append(compact)
        if log_every and (i + 1) % log_every == 0:
            print(f'  {i + 1} sequences ({time.time() - start:.0f}s)', flush=True)
    out['sequences'] = pd.DataFrame(out['sequences'])
    return out


# ---- compact canonical store -------------------------------------------------

_ARRAYS = {'acc': np.float32, 'rot': np.float32, 'thm': np.float32, 'tof': np.float16,
           'acc_observed': bool, 'rot_observed': bool, 'thm_observed': bool,
           'tof_observed': bool, 'tof_valid': bool}


def save_canonical(folder, seqs):
    """One .npy per array so readers can memory-map them (shared across workers)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, dtype in _ARRAYS.items():
        np.save(folder / f'{name}.npy', np.concatenate([getattr(s, name).astype(dtype) for s in seqs]))
    np.save(folder / 'offsets.npy', np.cumsum([0] + [s.n for s in seqs]))
    np.save(folder / 'sequence_id.npy', np.array([s.sequence_id for s in seqs]))


class CanonicalStore:
    """Memory-mapped canonical recordings. Pickles as just the folder path, so
    DataLoader workers re-open the maps instead of copying the data."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.offsets = np.load(self.folder / 'offsets.npy')
        self.sequence_ids = np.load(self.folder / 'sequence_id.npy').tolist()
        self.index = {sid: i for i, sid in enumerate(self.sequence_ids)}
        self._arrays = None

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = {name: np.load(self.folder / f'{name}.npy', mmap_mode='r') for name in _ARRAYS}
        return self._arrays

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_arrays'] = None
        return state

    def __len__(self):
        return len(self.sequence_ids)

    def get(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        parts = {name: np.array(arr[a:b], dtype=np.float64 if arr.dtype != bool else bool)
                 for name, arr in self.arrays.items()}
        meta = pd.DataFrame({'sequence_counter': np.arange(b - a)})
        return CleanSequence(self.sequence_ids[i], meta, parts['acc'], parts['rot'], parts['thm'],
                             parts['tof'], parts['acc_observed'], parts['rot_observed'],
                             parts['thm_observed'], parts['tof_observed'], parts['tof_valid'])


# ---- processed-folder helpers ---------------------------------------------------

def load_processed(folder):
    folder = Path(folder)
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    sequences = pd.read_csv(folder / 'sequences.csv')
    z = np.load(folder / 'features.npz', allow_pickle=False)
    return meta, sequences, z['features'], z['phase']


def sequence_slices(sequences):
    return [slice(int(a), int(a) + int(n)) for a, n in zip(sequences['start_row'], sequences['rows'])]
