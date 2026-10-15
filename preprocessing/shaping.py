"""Steps 10-12 of the guide: fixed length + mask, tree summaries, scaling, folds."""
import json

import numpy as np
import pandas as pd

FLAG_COLUMNS = ('rot_missing', 'thm_missing', 'tof_missing')


def fixed_length(x, length):
    """Keep the END of the recording (the gesture happens last) and pad at the
    start. Returns (length, ...) array and a (length,) mask, 1 = real row."""
    n = len(x)
    out = np.zeros((length,) + x.shape[1:], dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    keep = min(n, length)
    out[length - keep:] = x[n - keep:]
    mask[length - keep:] = 1.0
    return out, mask


def tree_summary(features, tail_fraction=0.25, use_tail=True):
    """One row per recording for tree models: statistics over the whole
    recording and (use_tail=True) over its last part, where the gesture happens."""
    frame = features.drop(columns=[c for c in FLAG_COLUMNS + ('time_frac',) if c in features])
    n = len(frame)
    tail = frame.iloc[max(0, n - max(1, int(round(n * tail_fraction)))):]
    values = frame.to_numpy(dtype=np.float64)
    tail_values = tail.to_numpy(dtype=np.float64)
    names = frame.columns
    with np.errstate(all='ignore'), _quiet():
        parts = {
            'mean': np.nanmean(values, axis=0),
            'std': np.nanstd(values, axis=0),
            'min': np.nanmin(values, axis=0),
            'max': np.nanmax(values, axis=0),
            'tail_mean': np.nanmean(tail_values, axis=0),
            'tail_std': np.nanstd(tail_values, axis=0),
            'last': values[-1],
        }
    if not use_tail:
        parts = {k: parts[k] for k in ('mean', 'std', 'min', 'max')}
    row = {f'{c}__{s}': v[i] for s, v in parts.items() for i, c in enumerate(names)}
    for c in FLAG_COLUMNS:
        if c in features:
            row[c] = float(features[c].iloc[0])
    row['length'] = float(n)
    return row


class _quiet:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


class Scaler:
    """Per-feature standardisation fitted on training rows only.

    NaN (feature unavailable) is ignored while fitting and becomes 0 (the
    training mean) after scaling; the *_missing flags carry the information.
    Flags and the mask are left unscaled.
    """

    def __init__(self, columns=None, mean=None, std=None):
        self.columns = list(columns) if columns is not None else None
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, frames):
        stacked = pd.concat(frames, ignore_index=True) if isinstance(frames, list) else frames
        self.columns = list(stacked.columns)
        values = stacked.to_numpy(dtype=np.float64)
        with _quiet():
            self.mean = np.nan_to_num(np.nanmean(values, axis=0))
            std = np.nan_to_num(np.nanstd(values, axis=0))
        self.std = np.where(std > 1e-8, std, 1.0)
        for i, c in enumerate(self.columns):
            if c in FLAG_COLUMNS:
                self.mean[i], self.std[i] = 0.0, 1.0
        return self

    def transform(self, frame):
        values = frame[self.columns].to_numpy(dtype=np.float64)
        scaled = (values - self.mean) / self.std
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'columns': self.columns, 'mean': self.mean.tolist(), 'std': self.std.tolist()}, fh)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        return cls(d['columns'], d['mean'], d['std'])


def make_folds(sequences, n_splits=5, seed=42):
    """Subject-grouped, gesture-stratified folds; one row per sequence.

    `sequences` needs sequence_id, subject and gesture columns.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(sequences), -1, dtype=int)
    for k, (_, valid) in enumerate(splitter.split(sequences, sequences['gesture'], sequences['subject'])):
        fold[valid] = k
    return pd.DataFrame({'sequence_id': sequences['sequence_id'].to_numpy(),
                         'subject': sequences['subject'].to_numpy(), 'fold': fold})
