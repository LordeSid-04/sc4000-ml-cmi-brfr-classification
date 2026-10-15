"""Section 8b: training-only augmentations on canonical recordings.

Augmentations act on the cleaned raw signals (acceleration, quaternion,
thermopile, ToF) and features are recomputed afterwards, so physics features
such as gravity removal stay consistent with the augmented motion.
"""
from dataclasses import dataclass

import numpy as np

from . import quaternion as quat
from .step3_features import gravity_in_device_frame


@dataclass(frozen=True)
class AugmentConfig:
    p_crop: float = 0.5
    crop_min: int = 70
    p_stretch: float = 0.5
    stretch: float = 0.15          # +-15 % speed
    p_scale: float = 0.5
    scale: float = 0.10            # +-10 % motion magnitude, smooth over time
    p_jitter: float = 0.5
    jitter: float = 0.02           # noise std relative to motion std
    p_wrist: float = 0.5
    wrist_deg: float = 7.0         # small watch tilt
    p_yaw: float = 0.5
    yaw_deg: float = 60.0          # person facing another direction
    p_drop_extra: float = 0.5      # blank thermopile + ToF (IMU-only test half)
    p_drop_rot: float = 0.05       # blank rotation (test sequences without it)


def take(seq, idx):
    out = seq.copy()
    for name in ['acc', 'rot', 'thm', 'tof', 'acc_observed', 'rot_observed',
                 'thm_observed', 'tof_observed', 'tof_valid']:
        setattr(out, name, getattr(seq, name)[idx])
    out.meta = seq.meta.iloc[idx].reset_index(drop=True)
    return out


def random_crop(seq, rng, min_len):
    """Window that ends at (or within a few rows of) the end of the recording."""
    n = seq.n
    if n <= min_len:
        return seq
    end = n - int(rng.integers(0, min(4, n - min_len) + 1))
    length = int(rng.integers(min_len, end + 1))
    return take(seq, np.arange(end - length, end))


def time_stretch(seq, rng, amount):
    """Replay the recording slightly faster or slower (linear interpolation)."""
    factor = 1.0 + rng.uniform(-amount, amount)
    n = seq.n
    m = max(8, int(round(n * factor)))
    src = np.linspace(0, n - 1, m)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None]
    out = take(seq, np.rint(src).astype(int))       # masks / meta: nearest row
    out.acc = seq.acc[lo] * (1 - w) + seq.acc[hi] * w
    out.rot = quat.normalize(seq.rot[lo] * (1 - w) + seq.rot[hi] * w)
    out.thm = seq.thm[lo] * (1 - w) + seq.thm[hi] * w
    w3 = w[:, :, None]
    out.tof = seq.tof[lo] * (1 - w3) + seq.tof[hi] * w3
    return out


def _smooth_curve(rng, n, amount, knots=4):
    x = np.linspace(0, n - 1, knots)
    return np.interp(np.arange(n), x, 1.0 + rng.uniform(-amount, amount, knots))


def magnitude_scale(seq, rng, amount):
    """Scale the motion (not gravity) by a smoothly varying factor."""
    out = seq.copy()
    g, _ = gravity_in_device_frame(seq)
    out.acc = g + (seq.acc - g) * _smooth_curve(rng, seq.n, amount)[:, None]
    return out


def jitter(seq, rng, amount):
    out = seq.copy()
    g, _ = gravity_in_device_frame(seq)
    spread = np.std(seq.acc - g, axis=0) + 1e-3
    out.acc = seq.acc + rng.normal(0, 1, seq.acc.shape) * spread * amount
    return out


def wrist_rotation(seq, rng, degrees):
    """Tilt the watch on the wrist: same motion seen from a slightly rotated device."""
    axis = rng.normal(size=3)
    r = quat.from_axis_angle(axis, np.deg2rad(rng.uniform(-degrees, degrees)))
    out = seq.copy()
    out.acc = quat.rotate(np.broadcast_to(r, (seq.n, 4)), seq.acc)
    if not seq.rot_missing:
        out.rot = quat.multiply(seq.rot, np.broadcast_to(quat.conjugate(r[None])[0], seq.rot.shape))
    return out


def yaw_rotation(seq, rng, degrees):
    """Turn the whole scene about the vertical: the person faces another way."""
    if seq.rot_missing:
        return seq
    r = quat.from_axis_angle([0, 0, 1], np.deg2rad(rng.uniform(-degrees, degrees)))
    out = seq.copy()
    out.rot = quat.multiply(np.broadcast_to(r, seq.rot.shape), seq.rot)
    return out


def drop_sensors(seq, drop_extra, drop_rot):
    out = seq.copy()
    if drop_extra:
        out.thm = np.full_like(seq.thm, np.nan)
        out.thm_observed = np.zeros_like(seq.thm_observed)
        out.tof = np.full_like(seq.tof, np.nan)
        out.tof_observed = np.zeros_like(seq.tof_observed)
        out.tof_valid = np.zeros_like(seq.tof_valid)
    if drop_rot:
        out.rot = np.tile([1.0, 0.0, 0.0, 0.0], (seq.n, 1))
        out.rot_observed = np.zeros_like(seq.rot_observed)
    return out


def augment(seq, rng, cfg=AugmentConfig()):
    """Randomly apply each augmentation with its own probability."""
    if rng.random() < cfg.p_crop:
        seq = random_crop(seq, rng, cfg.crop_min)
    if rng.random() < cfg.p_stretch:
        seq = time_stretch(seq, rng, cfg.stretch)
    if rng.random() < cfg.p_wrist:
        seq = wrist_rotation(seq, rng, cfg.wrist_deg)
    if rng.random() < cfg.p_yaw:
        seq = yaw_rotation(seq, rng, cfg.yaw_deg)
    if rng.random() < cfg.p_scale:
        seq = magnitude_scale(seq, rng, cfg.scale)
    if rng.random() < cfg.p_jitter:
        seq = jitter(seq, rng, cfg.jitter)
    seq = drop_sensors(seq, rng.random() < cfg.p_drop_extra, rng.random() < cfg.p_drop_rot)
    return seq


def mixup(x_a, x_b, y_a, y_b, rng, alpha=0.4):
    """Blend two (already featurised) examples; use with same-class pairs or soft labels."""
    lam = rng.beta(alpha, alpha)
    return lam * x_a + (1 - lam) * x_b, lam * y_a + (1 - lam) * y_b
