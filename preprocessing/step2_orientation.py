"""Steps 2 and 4 of the guide: put every recording into one canonical device frame.

Two physical situations change how the same gesture looks to the watch:
- Upside-down wearers: the device sits rotated 180 degrees about its z axis.
- Left-handed wearers: the watch is on the other wrist, a mirror image.

Both are a fixed change of device frame v' = M v. The accelerometer becomes
M @ acc. The quaternion (device -> world) becomes R' = W R M^T where W = M for a
mirror (the whole scene is mirrored, keeping gravity on world z) and W = I for
a rotation. ToF / thermopile sensors that trade places are permuted and ToF
images are flipped to match.

The exact axis, sensor permutation and image flip were chosen on the real
training data by matching left-handed (and upside-down) subjects against
right-handed ones (see the research-and-report branch).
"""
from dataclasses import dataclass

import numpy as np

from . import quaternion as quat

UPSIDE_DOWN_SUBJECTS = ('SUBJ_019262', 'SUBJ_045235')


@dataclass(frozen=True)
class FrameChange:
    kind: str                     # 'rotation' or 'mirror'
    axis: int                     # device axis: 0=x, 1=y, 2=z
    sensor_order: tuple           # new sensor k takes old sensor sensor_order[k]
    image_op: object = 'none'     # one of IMAGE_OPS, or a tuple with one per sensor

    def matrix(self):
        if self.kind == 'mirror':
            m = np.eye(3)
            m[self.axis, self.axis] = -1.0
        else:  # 180 degree rotation about `axis`
            m = -np.eye(3)
            m[self.axis, self.axis] = 1.0
        return m


# Exploratory settings selected with gesture labels from the WHOLE training set.
# These are disabled in PipelineConfig by default. Do not use them to claim
# held-out validation gains unless independently established or fit per fold.
# Selected on the training data:
# - left hand: mirror x (agreement with right-handers 0.38 -> 0.64, equal to their
#   own 0.64), swap sensors 3<->5, images flipped left-right on sensors 1/2/4 and
#   up-down on the sideways-mounted sensors 3/5.
# - upside down: 180 deg about z (agreement -0.20 -> 0.62), swap 2<->4 and 3<->5,
#   every image rotated 180 deg.
MIRROR = FrameChange('mirror', 0, (0, 1, 4, 3, 2), ('fliplr', 'fliplr', 'flipud', 'fliplr', 'flipud'))
UPSIDE_DOWN = FrameChange('rotation', 2, (0, 3, 4, 1, 2), 'rot180')


IMAGE_OPS = ('none', 'fliplr', 'flipud', 'rot180', 'transpose', 'antitranspose')


def image_op(img, op):
    """Apply one symmetry of the square to (..., 8, 8) images."""
    if op == 'fliplr':
        return img[..., :, ::-1]
    if op == 'flipud':
        return img[..., ::-1, :]
    if op == 'rot180':
        return img[..., ::-1, ::-1]
    if op == 'transpose':
        return np.swapaxes(img, -1, -2)
    if op == 'antitranspose':
        return np.swapaxes(img, -1, -2)[..., ::-1, ::-1]
    return img


def _image(tof, op):
    """tof: (n, 5, 64). op: one name for all sensors or a tuple of 5 names."""
    ops = (op,) * 5 if isinstance(op, str) else tuple(op)
    img = tof.reshape(tof.shape[0], 5, 8, 8)
    out = np.stack([image_op(img[:, k], ops[k]) for k in range(5)], axis=1)
    return np.ascontiguousarray(out).reshape(tof.shape)


def transform_quaternion(q, change):
    if change.kind == 'mirror':
        # R' = S R S for a reflection S: the rotation axis is an axial vector,
        # so the mirrored axis component is kept and the other two flip.
        sign = -np.ones(4)
        sign[0] = 1.0
        sign[1 + change.axis] = 1.0
        return q * sign
    # R' = R M^T: right-multiply by the inverse 180-degree rotation.
    axis = np.zeros(3)
    axis[change.axis] = 1.0
    inverse = quat.from_axis_angle(axis, -np.pi)
    return quat.multiply(q, np.broadcast_to(inverse, q.shape))


def apply_frame_change(seq, change):
    """Return a new CleanSequence expressed in the changed device frame."""
    out = seq.copy()
    m = change.matrix()
    out.acc = seq.acc @ m.T
    if not seq.rot_missing:
        out.rot = quat.make_continuous(transform_quaternion(seq.rot, change))
    order = list(change.sensor_order)
    out.thm = seq.thm[:, order]
    out.thm_observed = seq.thm_observed[:, order]
    out.tof = _image(seq.tof[:, order], change.image_op)
    out.tof_valid = _image(seq.tof_valid[:, order], change.image_op)
    out.tof_observed = seq.tof_observed[:, order]
    return out


def canonicalize(seq, subject=None, handedness=None, upside_down=None,
                 mirror=None, flip=None):
    """Apply the upside-down fix and left-hand mirroring for one recording.

    handedness follows the competition's demographics file: 1 = right, 0 = left.
    upside_down defaults to membership of the known subject list.
    """
    mirror = mirror or MIRROR
    flip = flip or UPSIDE_DOWN
    if upside_down is None:
        upside_down = subject in UPSIDE_DOWN_SUBJECTS
    if upside_down:
        seq = apply_frame_change(seq, flip)
    if handedness is not None and int(handedness) == 0:
        seq = apply_frame_change(seq, mirror)
    return seq
