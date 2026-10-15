"""Generate a small fake competition ZIP with KNOWN physics, for smoke tests.

Left-handed subjects are created by applying a known mirror (x axis, sensors
3<->5, images flipped left-right) and two subjects are rotated 180 degrees about
z, so scripts/01_verify_physics.py can be checked to recover exactly that.

    python scripts/tools/make_synthetic_data.py OUT.zip [--subjects 24]
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing import quaternion as quat  # noqa: E402
from preprocessing.cleaning import CleanSequence  # noqa: E402
from preprocessing.constants import ACC, GESTURES, GRAVITY, ROT, THM, TOF, TOF_MAX  # noqa: E402
from preprocessing.orientation import UPSIDE_DOWN_SUBJECTS, FrameChange, apply_frame_change  # noqa: E402

TRUE_MIRROR = FrameChange('mirror', 0, (0, 1, 4, 3, 2), 'fliplr')
TRUE_FLIP = FrameChange('rotation', 2, (0, 1, 2, 3, 4), 'none')
ORIENTATIONS = ['Seated Straight', 'Seated Lean Non Dom - FACE DOWN', 'Lie on Back', 'Lie on Side - Non Dominant']


def gesture_pattern(g, rng_master):
    rng = np.random.default_rng(1000 + g)
    tilt_axis = rng.normal(size=3)
    tilt = quat.from_axis_angle(tilt_axis, rng.uniform(0.3, 2.5))
    freq = rng.uniform(0.5, 3.0)
    amp = rng.uniform(0.5, 3.0, 3)
    thm = 27 + rng.uniform(0, 6, 5)
    img = rng.uniform(0, 1, (5, 8, 8))
    img[:, :, :4] *= rng.uniform(0.2, 1.0)       # left-right asymmetric images
    return tilt, freq, amp, thm, img


def make_sequence(g, n, rng, pattern):
    tilt, freq, amp, thm, img = pattern
    t = np.arange(n) / 10.0
    wobble = quat.from_axis_angle(np.tile([0, 0, 1.0], (n, 1)), 0.2 * np.sin(t * freq))
    q = quat.multiply(np.broadcast_to(tilt, (n, 4)), wobble)
    q = quat.multiply(np.broadcast_to(quat.from_axis_angle([0, 0, 1], rng.uniform(0, 6.28)), (n, 4)), q)  # random heading
    motion = amp * np.sin(t[:, None] * freq * 2 * np.pi + np.arange(3))
    acc = quat.rotate(quat.conjugate(q), np.broadcast_to([0, 0, GRAVITY], (n, 3))) + motion + rng.normal(0, 0.1, (n, 3))
    thm_v = thm + rng.normal(0, 0.3, (n, 5))
    prox = np.clip(img.reshape(5, 64)[None] + rng.normal(0, 0.05, (n, 5, 64)), 0, 1)
    valid = prox > 0.1
    seq = CleanSequence('x', pd.DataFrame({'sequence_counter': np.arange(n)}), acc, q, thm_v, np.where(valid, prox, 0.0),
                        np.ones(n, bool), np.ones(n, bool), np.ones((n, 5), bool), np.ones((n, 5), bool), valid)
    return seq


def to_raw(seq):
    dist = np.where(seq.tof_valid, np.rint((1 - seq.tof) * (TOF_MAX + 1)), -1).clip(-1, TOF_MAX)
    values = np.hstack([seq.acc, seq.rot, seq.thm, dist.reshape(seq.n, 320)])
    return pd.DataFrame(values, columns=ACC + ROT + THM + TOF)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--subjects', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    patterns = [gesture_pattern(g, rng) for g in range(len(GESTURES))]
    subjects = [f'SUBJ_{i:06d}' for i in range(args.subjects - 2)] + list(UPSIDE_DOWN_SUBJECTS)
    demo = pd.DataFrame({'subject': subjects, 'adult_child': 1, 'age': 20, 'sex': 1,
                         'handedness': [0 if i % 4 == 0 else 1 for i in range(len(subjects))],
                         'height_cm': 170, 'shoulder_to_wrist_cm': 55, 'elbow_to_wrist_cm': 30})
    frames = []
    counter = 0
    for si, s in enumerate(subjects):
        hand = demo.loc[si, 'handedness']
        for g, gesture in enumerate(GESTURES):
            for o in range(2):
                n = int(rng.integers(35, 90))
                seq = make_sequence(g, n, rng, patterns[g])
                if hand == 0:
                    seq = apply_frame_change(seq, TRUE_MIRROR)
                if s in UPSIDE_DOWN_SUBJECTS:
                    seq = apply_frame_change(seq, TRUE_FLIP)
                raw = to_raw(seq)
                if rng.random() < 0.3:              # IMU-only recording
                    raw[THM + TOF] = np.nan
                if rng.random() < 0.05:             # rotation missing
                    raw[ROT] = np.nan
                if rng.random() < 0.2:              # short dropout
                    raw.loc[5:7, THM] = np.nan
                    raw.loc[10:11, ROT] = np.nan
                if rng.random() < 0.3:              # quaternion sign flip
                    raw.loc[n // 2:, ROT] *= -1
                sid = f'SEQ_{counter:06d}'
                counter += 1
                behavior = np.where(np.arange(n) < n // 3, 'Moves hand to target location',
                                    np.where(np.arange(n) < n // 2, 'Hand at target location', 'Performs gesture'))
                if o == 0:
                    behavior[0] = 'Relaxes and moves hand to target location'
                meta = pd.DataFrame({
                    'row_id': [f'{sid}_{i}' for i in range(n)], 'sequence_type': 'Target' if g < 8 else 'Non-Target',
                    'sequence_id': sid, 'sequence_counter': np.arange(n), 'subject': s,
                    'orientation': ORIENTATIONS[(g + o) % 4], 'behavior': behavior,
                    'phase': np.where(behavior == 'Performs gesture', 'Gesture', 'Transition'), 'gesture': gesture})
                frames.append(pd.concat([meta, raw], axis=1))
    train = pd.concat(frames, ignore_index=True)
    test = train[train.sequence_id.isin(train.sequence_id.unique()[:2])].drop(
        columns=['sequence_type', 'orientation', 'behavior', 'phase', 'gesture'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, df in [('train.csv', train), ('train_demographics.csv', demo), ('test.csv', test),
                         ('test_demographics.csv', demo[demo.subject.isin(test.subject)])]:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            z.writestr(name, buf.getvalue())
    print(f'wrote {args.out}: {len(train)} rows, {train.sequence_id.nunique()} sequences')


if __name__ == '__main__':
    main()
