"""Steps 5-8 of the guide: physics features, smoothing and ToF / thermopile summaries.

Input is a canonical CleanSequence (cleaned, upside-down fixed, mirrored).
Output is one row of features per time step. Values that do not exist for
this recording (e.g. thermopile features when the sensor is off) are NaN:
tree models handle NaN natively, and the neural-network path replaces it with
0 after scaling, next to an explicit *_missing flag.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import quaternion as quat
from .constants import GRAVITY, SAMPLE_RATE_HZ

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class FeatureConfig:
    gravity: bool = True          # step 5: remove gravity (quaternion or low-pass fallback)
    engineered: bool = True       # step 7: jerk, angular velocity, 6D rotation, ...
    # step 6: Gaussian sigma (rows) before derivatives. Off: the ablation showed smoothing
    # (sigma=1) lowers the score in 4/5 folds (all sensors) and 5/5 folds (IMU-only).
    smooth_sigma: float = 0.0
    gravity_sigma: float = 10.0   # low-pass sigma (rows, ~1 s) for the no-rotation fallback
    tof_regions: bool = True      # step 8: 3x3 regional averages of each ToF image
    time: bool = True             # time_frac


def _smooth(x, sigma):
    if sigma <= 0 or len(x) < 2:
        return x
    return gaussian_filter1d(x, sigma, axis=0, mode='nearest')


def _diff(x):
    d = np.zeros_like(x)
    if len(x) > 1:
        d[1:] = np.diff(x, axis=0)
    return d * SAMPLE_RATE_HZ


def gravity_in_device_frame(seq, cfg=FeatureConfig()):
    """Gravity vector (n, 3) seen from the watch, and whether it came from the
    quaternion (True) or the low-pass fallback (False)."""
    if not seq.rot_missing:
        return quat.rotate(quat.conjugate(seq.rot), np.broadcast_to(UP * GRAVITY, seq.acc.shape)), True
    # Fallback: gravity is the slowly varying part of the accelerometer signal.
    g = _smooth(seq.acc, cfg.gravity_sigma)
    norm = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.where(norm > 1e-6, norm, 1.0) * GRAVITY, False


# Regions of an 8x8 image: rows/cols split 3-2-3 so the centre has its own cell.
_CUTS = [(0, 3), (3, 5), (5, 8)]


def tof_features(seq, regions=True):
    tof = seq.tof.reshape(seq.n, 5, 8, 8)
    out = {}
    for k in range(5):
        img = tof[:, k]
        flat = img.reshape(seq.n, 64)
        absent = ~np.isfinite(flat).any(axis=1)
        filled = np.where(np.isfinite(flat), flat, 0.0)
        name = f'tof_{k + 1}'
        mean = filled.mean(axis=1)
        out[f'{name}_mean'] = mean
        out[f'{name}_max'] = filled.max(axis=1)
        out[f'{name}_std'] = filled.std(axis=1)
        out[f'{name}_valid_frac'] = seq.tof_valid[:, k].mean(axis=1)
        out[f'{name}_mean_diff'] = _diff(mean[:, None])[:, 0]
        if regions:
            fimg = filled.reshape(seq.n, 8, 8)
            for a, (r0, r1) in enumerate(_CUTS):
                for b, (c0, c1) in enumerate(_CUTS):
                    out[f'{name}_r{a}{b}'] = fimg[:, r0:r1, c0:c1].mean(axis=(1, 2))
        for key in [c for c in out if c.startswith(name + '_')]:
            out[key] = np.where(absent, np.nan, out[key])
    return out


def sequence_features(seq, cfg=FeatureConfig()):
    """Per-row feature table (n rows) for one canonical recording."""
    n = seq.n
    f = {}
    acc = seq.acc
    f['acc_x'], f['acc_y'], f['acc_z'] = acc.T
    f['acc_mag'] = np.linalg.norm(acc, axis=1)
    rot_ok = not seq.rot_missing
    nan = np.full(n, np.nan)

    if cfg.gravity:
        g, from_quat = gravity_in_device_frame(seq, cfg)
        linear = acc - g
        gdir = g / GRAVITY
        f['linear_acc_x'], f['linear_acc_y'], f['linear_acc_z'] = linear.T
        f['linear_acc_mag'] = np.linalg.norm(linear, axis=1)
        f['gravity_dir_x'], f['gravity_dir_y'], f['gravity_dir_z'] = gdir.T
        if rot_ok:
            world = quat.rotate(seq.rot, acc) - UP * GRAVITY
            f['linear_acc_vertical'] = world[:, 2]
            f['linear_acc_horizontal'] = np.linalg.norm(world[:, :2], axis=1)
        else:
            vertical = np.sum(linear * gdir, axis=1)
            f['linear_acc_vertical'] = vertical
            f['linear_acc_horizontal'] = np.sqrt(np.maximum(f['linear_acc_mag'] ** 2 - vertical ** 2, 0))
    else:
        linear = acc

    if rot_ok:
        f['rot_w'], f['rot_x'], f['rot_y'], f['rot_z'] = seq.rot.T
    else:
        f['rot_w'] = f['rot_x'] = f['rot_y'] = f['rot_z'] = nan

    if cfg.engineered:
        smooth_lin = _smooth(linear, cfg.smooth_sigma)
        jerk = _diff(smooth_lin)
        f['jerk_x'], f['jerk_y'], f['jerk_z'] = jerk.T
        f['jerk_mag'] = np.linalg.norm(jerk, axis=1)
        prev = np.vstack([smooth_lin[:1], smooth_lin[:-1]])
        denom = np.linalg.norm(smooth_lin, axis=1) * np.linalg.norm(prev, axis=1)
        f['acc_cos_sim'] = np.where(denom > 1e-9, np.sum(smooth_lin * prev, axis=1) / np.maximum(denom, 1e-9), 1.0)
        if rot_ok:
            q = quat.normalize(_smooth(seq.rot, cfg.smooth_sigma))
            omega = quat.rotation_vector(quat.relative_rotation(q)) * SAMPLE_RATE_HZ
            omega = np.vstack([omega[-1:] * 0, omega[:-1]])  # align: row t holds t-1 -> t
            f['angular_vel_x'], f['angular_vel_y'], f['angular_vel_z'] = omega.T
            f['angular_distance'] = np.linalg.norm(omega, axis=1) / SAMPLE_RATE_HZ
            r6 = quat.to_6d(seq.rot)
            for i in range(6):
                f[f'rot6d_{i + 1}'] = r6[:, i]
        else:
            for name in ['angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'] + [f'rot6d_{i}' for i in range(1, 7)]:
                f[name] = nan

    for k in range(5):
        f[f'thm_{k + 1}'] = seq.thm[:, k]
    if cfg.engineered:
        with np.errstate(invalid='ignore'):
            present = np.isfinite(seq.thm)
            count = present.sum(axis=1)
            total = np.where(present, seq.thm, 0).sum(axis=1)
            f['thm_mean'] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        for k in range(5):
            f[f'thm_{k + 1}_diff'] = _diff(seq.thm[:, k:k + 1])[:, 0]

    f.update(tof_features(seq, cfg.tof_regions))

    f['rot_missing'] = np.full(n, float(seq.rot_missing))
    f['thm_missing'] = np.full(n, float(seq.thm_missing))
    f['tof_missing'] = np.full(n, float(seq.tof_missing))
    if cfg.time:
        f['time_frac'] = np.arange(n) / max(n - 1, 1)
    return pd.DataFrame(f).astype(np.float32)


def tof_images(seq):
    """(n, 5, 8, 8) proximity images with absent sensors set to 0."""
    img = seq.tof.reshape(seq.n, 5, 8, 8)
    return np.where(np.isfinite(img), img, 0.0).astype(np.float32)
