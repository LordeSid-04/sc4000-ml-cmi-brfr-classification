"""Vectorised quaternion maths. Quaternions are (N, 4) arrays in (w, x, y, z) order.

Convention (checked on the real data by scripts/02_verify_physics.py): the sensor
quaternion rotates device-frame vectors into the world frame,
    v_world = q * v_device * conj(q),
and in the world frame gravity lies on +z.
"""
import numpy as np


def normalize(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(norm > 0, norm, 1.0)


def conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def multiply(a, b):
    """Hamilton product a * b (broadcasts over leading axes)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def rotate(q, v):
    """Rotate vectors v (N, 3) by unit quaternions q (N, 4): q v q*."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def make_continuous(q):
    """Flip signs so consecutive quaternions have a non-negative dot product.

    q and -q describe the same rotation; the sensor may switch between them,
    which looks like a sudden full turn in the raw numbers. Rows that are NaN
    are skipped and do not break the chain.
    """
    q = q.copy()
    previous = None
    for i in range(len(q)):
        if not np.isfinite(q[i]).all():
            continue
        if previous is not None and np.dot(q[i], previous) < 0:
            q[i] = -q[i]
        previous = q[i]
    return q


def to_matrix(q):
    """Rotation matrices (N, 3, 3) for unit quaternions."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def to_6d(q):
    """Continuous 6D rotation representation (Zhou et al., 2019): the first two
    columns of the rotation matrix. Unlike quaternions it has no sign ambiguity."""
    m = to_matrix(q)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def relative_rotation(q):
    """Rotation from row t to row t+1 in the device frame, as (N, 4); last row = identity."""
    rel = np.tile([1.0, 0.0, 0.0, 0.0], (len(q), 1))
    if len(q) > 1:
        rel[:-1] = multiply(conjugate(q[:-1]), q[1:])
    return rel


def rotation_vector(q):
    """Axis * angle (N, 3) for unit quaternions, using the shorter of q / -q."""
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, np.clip(q[..., :1], -1.0, 1.0))
    scale = np.where(s > 1e-12, angle / np.where(s > 1e-12, s, 1.0), 2.0)
    return v * scale


def from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = np.asarray(angle, dtype=float)[..., None] / 2.0
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
