"""Step 1 + 3 of the guide: load one recording, order it, and repair the raw sensors.

Everything here looks at a single sequence only (no labels, other subjects or
fitted statistics), so the exact same code runs on training data and inside
the Kaggle predict() call.

Design choices (each one is ablated or verified in the report):
- Rows are ordered by sequence_counter.
- Acceleration / rotation / temperature gaps inside a recording are filled by
  linear interpolation over the counter, nearest value at the edges. Rotation
  is interpolated on sign-continuous quaternions and re-normalised (nlerp).
- Quaternions are made sign-continuous: q and -q are the same rotation and a
  sign switch would otherwise look like a sudden full twist.
- ToF is converted to *proximity* in [0, 1]: 1 = touching, -> 0 = far away.
  A reading of -1 ("nothing in range") becomes 0, i.e. as far as possible.
  This keeps the physical ordering intact, unlike mapping -1 to distance 0
  (which would mean "touching").
- A sensor that delivered nothing for the whole recording stays NaN and is
  reported in the flags; the model-specific steps decide how to fill it.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import quaternion as quat
from .constants import ACC, ROT, THM, TOF, TOF_MAX


@dataclass
class CleanSequence:
    sequence_id: str
    meta: pd.DataFrame          # non-sensor columns (row order matches arrays)
    acc: np.ndarray             # (n, 3) m/s^2, always finite
    rot: np.ndarray             # (n, 4) unit quaternion (w, x, y, z), continuous
    thm: np.ndarray             # (n, 5) degC, NaN if that sensor is absent
    tof: np.ndarray             # (n, 5, 64) proximity in [0, 1], NaN if sensor absent
    acc_observed: np.ndarray    # (n,) original row had a finite acceleration
    rot_observed: np.ndarray    # (n,) original row had a usable quaternion
    thm_observed: np.ndarray    # (n, 5)
    tof_observed: np.ndarray    # (n, 5) sensor delivered a frame on this row
    tof_valid: np.ndarray       # (n, 5, 64) pixel saw an object (not -1, not missing)
    issues: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.acc)

    @property
    def rot_missing(self):
        return not self.rot_observed.any()

    @property
    def thm_missing(self):
        return not self.thm_observed.any()

    @property
    def tof_missing(self):
        return not self.tof_observed.any()

    def copy(self):
        return CleanSequence(
            self.sequence_id, self.meta.copy(), self.acc.copy(), self.rot.copy(),
            self.thm.copy(), self.tof.copy(), self.acc_observed.copy(),
            self.rot_observed.copy(), self.thm_observed.copy(),
            self.tof_observed.copy(), self.tof_valid.copy(), list(self.issues))


class DataError(ValueError):
    """Raised in strict mode when a recording violates an assumption."""


def fill_gaps(x, t):
    """Linear interpolation over t for NaNs in each column of x; nearest value at
    the edges. Columns without any finite value stay NaN."""
    x = np.array(x, dtype=np.float64, copy=True)
    finite = np.isfinite(x)
    for j in np.flatnonzero(~finite.all(axis=0) & finite.any(axis=0)):
        ok = finite[:, j]
        x[~ok, j] = np.interp(t[~ok], t[ok], x[ok, j])
    x[~np.isfinite(x)] = np.nan
    return x


def _order(sequence, strict, issues):
    if 'sequence_counter' not in sequence:
        if strict:
            raise DataError('Missing sequence_counter')
        issues.append('no sequence_counter; using row order')
        data = sequence.reset_index(drop=True).copy()
        data['sequence_counter'] = np.arange(len(data))
        return data
    counter = pd.to_numeric(sequence['sequence_counter'], errors='coerce')
    bad = ~np.isfinite(counter.to_numpy(dtype=float))
    if bad.any():
        if strict:
            raise DataError('Non-numeric sequence_counter')
        issues.append(f'{int(bad.sum())} invalid counters; using row order for them')
        counter = counter.where(~bad, np.nan).interpolate(limit_direction='both')
        if counter.isna().all():
            counter = pd.Series(np.arange(len(sequence)), index=sequence.index)
    data = sequence.copy()
    data['sequence_counter'] = counter.to_numpy()
    data = data.sort_values('sequence_counter', kind='stable').reset_index(drop=True)
    duplicated = data['sequence_counter'].duplicated()
    if duplicated.any():
        if strict:
            raise DataError('Duplicate sequence counters')
        issues.append(f'{int(duplicated.sum())} duplicate counters dropped')
        data = data.loc[~duplicated].reset_index(drop=True)
    return data


def clean_sequence(sequence, strict=False, continuity=True, tof_no_object='far'):
    """Clean one recording given as a pandas DataFrame with the raw columns.

    strict=True (training) raises DataError on anything unexpected so the data
    can be audited. strict=False (inference) never raises for repairable
    problems: Kaggle gives us one sequence at a time and a crash would fail the
    whole submission.

    continuity and tof_no_object are switches for experiments; keep the defaults.
    tof_no_object='near' reproduces the earlier "-1 -> distance 0" choice.
    """
    issues = []
    if len(sequence) == 0:
        raise DataError('Empty sequence')
    if sequence.columns.duplicated().any():
        if strict:
            raise DataError('Duplicate column names')
        sequence = sequence.loc[:, ~sequence.columns.duplicated()]
    if 'sequence_id' in sequence:
        ids = sequence['sequence_id'].dropna().unique()
        if strict and (len(ids) != 1 or sequence['sequence_id'].isna().any()):
            raise DataError('Pass exactly one sequence with a non-missing id')
        sequence_id = str(ids[0]) if len(ids) else 'unknown'
    else:
        if strict:
            raise DataError('Missing sequence_id')
        sequence_id = 'unknown'

    absent = [c for c in ACC + ROT + THM + TOF if c not in sequence]
    if absent:
        if strict and any(c in ACC for c in absent):
            raise DataError(f'Missing acceleration columns: {absent}')
        issues.append(f'{len(absent)} sensor columns absent; treated as missing')
        sequence = sequence.assign(**{c: np.nan for c in absent})

    data = _order(sequence, strict, issues)
    t = data['sequence_counter'].to_numpy(dtype=np.float64)
    n = len(data)

    def block(columns):
        frame = data[columns]
        if not all(pd.api.types.is_numeric_dtype(d) for d in frame.dtypes):
            frame = frame.apply(pd.to_numeric, errors='coerce')
        return np.array(frame.to_numpy(dtype=np.float64), copy=True)

    # Acceleration: always needed; interpolate rare gaps.
    acc = block(ACC)
    acc_observed = np.isfinite(acc).all(axis=1)
    if not acc_observed.all():
        if strict:
            raise DataError('Missing acceleration values')
        issues.append(f'{int((~acc_observed).sum())} rows with missing acceleration')
        acc[~acc_observed] = np.nan
        acc = fill_gaps(acc, t)
        acc[~np.isfinite(acc)] = 0.0

    # Rotation: usable = finite with a positive norm.
    rot = block(ROT)
    with np.errstate(over='ignore', invalid='ignore'):
        norm = np.linalg.norm(rot, axis=1)
    rot_observed = np.isfinite(rot).all(axis=1) & np.isfinite(norm) & (norm > 1e-8)
    rot[~rot_observed] = np.nan
    rot[rot_observed] /= norm[rot_observed, None]
    if rot_observed.any():
        if continuity:
            rot = quat.make_continuous(rot)
        rot = quat.normalize(fill_gaps(rot, t))
    else:
        # No orientation at all. Identity is a numeric placeholder only; every
        # rotation-based feature checks rot_missing and uses a fallback instead.
        rot = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))

    # Thermopiles.
    thm = block(THM)
    thm_observed = np.isfinite(thm)
    thm = fill_gaps(thm, t)

    # Time-of-flight: distance -> proximity; -1 -> 0 (nothing in range).
    raw = block(TOF).reshape(n, 5, 64)
    finite = np.isfinite(raw)
    tof_valid = finite & (raw >= 0)
    tof_observed = finite.any(axis=2)
    no_object = 0.0 if tof_no_object == 'far' else 1.0
    proximity = np.where(tof_valid, 1.0 - np.clip(raw, 0, TOF_MAX) / (TOF_MAX + 1.0), no_object)
    proximity[~finite] = np.nan
    tof = fill_gaps(proximity.reshape(n, 320), t).reshape(n, 5, 64)

    meta = data.drop(columns=ACC + ROT + THM + TOF)
    return CleanSequence(sequence_id, meta, acc, rot, thm, tof, acc_observed,
                         rot_observed, thm_observed, tof_observed, tof_valid, issues)
