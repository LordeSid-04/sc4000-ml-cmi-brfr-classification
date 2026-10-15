"""Tests for the preprocessing package. Synthetic recordings are built from known physics
so the expected answers are exact, not just "runs without error"."""
import unittest

import numpy as np
import pandas as pd

from preprocessing import quaternion as quat
from preprocessing.augmentation import AugmentConfig, augment, time_stretch, wrist_rotation, yaw_rotation
from preprocessing.step1_cleaning import DataError, clean_sequence
from preprocessing.constants import ACC, GRAVITY, ROT, THM, TOF
from preprocessing.step3_features import FeatureConfig, gravity_in_device_frame, sequence_features
from preprocessing.step4_labels import competition_score, initial_state, phase_of, sequence_labels
from preprocessing.step2_orientation import MIRROR, UPSIDE_DOWN, apply_frame_change, canonicalize
from preprocessing.pipeline import preprocess
from preprocessing.step5_shaping import Scaler, fixed_length, make_folds, tree_summary

UP = np.array([0.0, 0.0, GRAVITY])


def random_quaternions(n, seed=0):
    rng = np.random.default_rng(seed)
    q0 = quat.normalize(rng.normal(size=4))
    steps = quat.from_axis_angle(rng.normal(size=(n, 3)), rng.uniform(0, 0.1, n))
    q = [q0]
    for s in steps[1:]:
        q.append(quat.multiply(q[-1], s))
    return np.array(q)


def recording(n=40, seed=0, motion=None):
    """Physically consistent recording: acc = gravity seen by the device + motion."""
    rng = np.random.default_rng(seed)
    q = random_quaternions(n, seed)
    gravity_device = quat.rotate(quat.conjugate(q), np.broadcast_to(UP, (n, 3)))
    motion = np.zeros((n, 3)) if motion is None else motion
    return pd.concat([
        pd.DataFrame(gravity_device + motion, columns=ACC),
        pd.DataFrame(q, columns=ROT),
        pd.DataFrame(30 + rng.normal(0, 0.5, (n, 5)), columns=THM),
        pd.DataFrame(rng.integers(0, 255, (n, 320)).astype(float), columns=TOF),
        pd.DataFrame({'sequence_id': 'SEQ_TEST', 'sequence_counter': np.arange(n), 'subject': 'SUBJ_TEST'}),
    ], axis=1)


def gravity_consistency_error(seq):
    """Max deviation between low-motion acceleration and quaternion gravity."""
    g = quat.rotate(quat.conjugate(seq.rot), np.broadcast_to(UP, seq.acc.shape))
    return np.abs(seq.acc - g).max()


class QuaternionTests(unittest.TestCase):
    def test_rotate_matches_matrix(self):
        q = random_quaternions(20)
        v = np.random.default_rng(1).normal(size=(20, 3))
        np.testing.assert_allclose(quat.rotate(q, v), np.einsum('nij,nj->ni', quat.to_matrix(q), v), atol=1e-12)

    def test_continuity_removes_sign_flips(self):
        q = random_quaternions(30)
        flipped = q.copy()
        flipped[10:20] *= -1
        fixed = quat.make_continuous(flipped)
        self.assertTrue((np.sum(fixed[1:] * fixed[:-1], axis=1) >= 0).all())
        np.testing.assert_allclose(np.abs(np.sum(fixed * q, axis=1)), 1, atol=1e-12)

    def test_6d_is_sign_invariant(self):
        q = random_quaternions(10)
        np.testing.assert_allclose(quat.to_6d(q), quat.to_6d(-q), atol=1e-12)

    def test_rotation_vector_angle(self):
        q = quat.from_axis_angle(np.array([[0, 0, 1.0]]), np.array([0.3]))
        np.testing.assert_allclose(quat.rotation_vector(q), [[0, 0, 0.3]], atol=1e-12)


class CleaningTests(unittest.TestCase):
    def test_order_and_input_untouched(self):
        frame = recording().sample(frac=1, random_state=0)
        original = frame.copy(deep=True)
        seq = clean_sequence(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(seq.meta['sequence_counter'].tolist(), list(range(40)))

    def test_tof_proximity_semantics(self):
        frame = recording(5)
        frame['tof_1_v0'] = [0, 254, -1, np.nan, 100]
        seq = clean_sequence(frame)
        p = seq.tof[:, 0, 0]
        self.assertAlmostEqual(p[0], 1.0)            # touching -> 1
        self.assertLess(p[1], 0.01)                  # far -> ~0
        self.assertEqual(p[2], 0.0)                  # nothing in range -> as far as possible
        self.assertTrue(0 < p[3] < 1)                # short gap interpolated
        self.assertEqual(seq.tof_valid[:, 0, 0].tolist(), [True, True, False, False, True])

    def test_absent_sensors_stay_nan_with_flags(self):
        frame = recording(10)
        frame[THM + TOF] = np.nan
        seq = clean_sequence(frame)
        self.assertTrue(seq.thm_missing and seq.tof_missing and not seq.rot_missing)
        self.assertTrue(np.isnan(seq.thm).all() and np.isnan(seq.tof).all())

    def test_rotation_gap_is_interpolated_not_identity(self):
        frame = recording(20)
        truth = clean_sequence(frame).rot
        frame.loc[8:10, ROT] = np.nan
        seq = clean_sequence(frame)
        self.assertFalse(seq.rot_observed[8:11].any())
        # Interpolated orientation stays close to the true smooth path.
        self.assertGreater(np.abs(np.sum(seq.rot[8:11] * truth[8:11], axis=1)).min(), 0.98)

    def test_rotation_sign_flip_fixed(self):
        frame = recording(20)
        frame.loc[5:9, ROT] *= -1
        seq = clean_sequence(frame)
        self.assertTrue((np.sum(seq.rot[1:] * seq.rot[:-1], axis=1) > 0).all())

    def test_missing_rotation_flag(self):
        frame = recording(10)
        frame[ROT] = np.nan
        seq = clean_sequence(frame)
        self.assertTrue(seq.rot_missing)
        np.testing.assert_allclose(seq.rot, np.tile([1, 0, 0, 0], (10, 1)))

    def test_inference_mode_repairs_instead_of_crashing(self):
        frame = recording(10)
        frame.loc[3, 'acc_x'] = np.nan
        frame.loc[4, 'sequence_counter'] = 3
        frame = frame.drop(columns=['tof_5_v63'])
        seq = clean_sequence(frame)
        self.assertTrue(np.isfinite(seq.acc).all())
        self.assertEqual(seq.n, 9)
        self.assertTrue(seq.issues)

    def test_strict_mode_raises(self):
        for change in ['acc', 'counter', 'id']:
            frame = recording(10)
            if change == 'acc':
                frame.loc[3, 'acc_x'] = np.nan
            elif change == 'counter':
                frame.loc[4, 'sequence_counter'] = 3
            else:
                frame.loc[0, 'sequence_id'] = 'OTHER'
            with self.subTest(change=change), self.assertRaises(DataError):
                clean_sequence(frame, strict=True)

    def test_empty_rejected(self):
        with self.assertRaises(DataError):
            clean_sequence(recording().iloc[:0])


class OrientationTests(unittest.TestCase):
    def test_frame_changes_keep_physics_consistent(self):
        seq = clean_sequence(recording())
        self.assertLess(gravity_consistency_error(seq), 1e-9)
        for change in (MIRROR, UPSIDE_DOWN):
            with self.subTest(change=change.kind):
                self.assertLess(gravity_consistency_error(apply_frame_change(seq, change)), 1e-9)

    def test_frame_changes_are_involutions(self):
        seq = clean_sequence(recording())
        for change in (MIRROR, UPSIDE_DOWN):
            twice = apply_frame_change(apply_frame_change(seq, change), change)
            np.testing.assert_allclose(twice.acc, seq.acc, atol=1e-12)
            np.testing.assert_allclose(np.abs(np.sum(twice.rot * seq.rot, axis=1)), 1, atol=1e-9)
            np.testing.assert_array_equal(twice.tof_valid, seq.tof_valid)
            np.testing.assert_allclose(twice.thm, seq.thm)

    def test_mirror_flips_axis_and_swaps_sensors(self):
        seq = clean_sequence(recording())
        m = apply_frame_change(seq, MIRROR)
        np.testing.assert_allclose(m.acc[:, MIRROR.axis], -seq.acc[:, MIRROR.axis])
        order = list(MIRROR.sensor_order)
        np.testing.assert_allclose(m.thm, seq.thm[:, order])

    def test_canonicalize_rules(self):
        seq = clean_sequence(recording())
        same = canonicalize(seq, subject='SUBJ_TEST', handedness=1)
        np.testing.assert_allclose(same.acc, seq.acc)
        left = canonicalize(seq, subject='SUBJ_TEST', handedness=0)
        np.testing.assert_allclose(left.acc[:, MIRROR.axis], -seq.acc[:, MIRROR.axis])
        flipped = canonicalize(seq, subject='SUBJ_019262', handedness=1)
        self.assertFalse(np.allclose(flipped.acc, seq.acc))


class FeatureTests(unittest.TestCase):
    def test_gravity_removal_recovers_motion(self):
        n = 60
        motion = np.zeros((n, 3))
        motion[20:30, 0] = 2.0
        seq = clean_sequence(recording(n, motion=motion))
        f = sequence_features(seq)
        np.testing.assert_allclose(f[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']].to_numpy(), motion, atol=1e-5)
        g, from_quat = gravity_in_device_frame(seq)
        self.assertTrue(from_quat)
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), GRAVITY)

    def test_fallback_gravity_without_rotation(self):
        frame = recording(60)
        frame[ROT] = np.nan
        seq = clean_sequence(frame)
        g, from_quat = gravity_in_device_frame(seq)
        self.assertFalse(from_quat)
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), GRAVITY, rtol=1e-6)
        f = sequence_features(seq)
        self.assertTrue(np.isfinite(f['linear_acc_mag']).all())
        self.assertTrue(f['angular_vel_x'].isna().all())
        self.assertEqual(f['rot_missing'].iloc[0], 1.0)

    def test_same_columns_in_all_sensor_situations(self):
        columns = None
        for drop_rot in (False, True):
            for drop_extra in (False, True):
                frame = recording(30)
                if drop_rot:
                    frame[ROT] = np.nan
                if drop_extra:
                    frame[THM + TOF] = np.nan
                f = preprocess(frame).features
                if columns is None:
                    columns = list(f.columns)
                self.assertEqual(list(f.columns), columns)
                self.assertTrue(np.isfinite(f[['acc_x', 'linear_acc_mag', 'time_frac']].to_numpy()).all())

    def test_angular_velocity_constant_spin(self):
        n = 30
        rate = 0.05  # rad per row about device z
        q = quat.from_axis_angle(np.tile([0, 0, 1.0], (n, 1)), rate * np.arange(n))
        frame = recording(n)
        frame[ROT] = q
        frame[ACC] = quat.rotate(quat.conjugate(q), np.broadcast_to(UP, (n, 3)))
        f = sequence_features(clean_sequence(frame), FeatureConfig(smooth_sigma=0))
        np.testing.assert_allclose(f['angular_distance'].to_numpy()[1:], rate, atol=1e-6)
        np.testing.assert_allclose(f['angular_vel_z'].to_numpy()[1:], rate * 10, atol=1e-4)

    def test_labels_do_not_change_features(self):
        frame = recording()
        a = preprocess(frame).features
        frame['gesture'] = 'Neck - scratch'
        frame['orientation'] = 'Seated Straight'
        b = preprocess(frame).features
        pd.testing.assert_frame_equal(a, b)


class ShapingTests(unittest.TestCase):
    def test_fixed_length_keeps_end(self):
        x = np.arange(10, dtype=float)[:, None]
        out, mask = fixed_length(x, 4)
        np.testing.assert_array_equal(out[:, 0], [6, 7, 8, 9])
        out, mask = fixed_length(x, 12)
        np.testing.assert_array_equal(mask, [0, 0] + [1] * 10)
        np.testing.assert_array_equal(out[2:, 0], np.arange(10))

    def test_scaler_ignores_nan_and_flags(self):
        frame = pd.DataFrame({'a': [1.0, 3.0, np.nan], 'tof_missing': [1.0, 0.0, 1.0]})
        s = Scaler().fit(frame)
        x = s.transform(frame)
        np.testing.assert_allclose(x[:, 0], [-1, 1, 0])
        np.testing.assert_allclose(x[:, 1], frame['tof_missing'])

    def test_tree_summary_tail(self):
        f = pd.DataFrame({'v': np.r_[np.zeros(6), np.ones(2)], 'time_frac': np.linspace(0, 1, 8)})
        row = tree_summary(f)
        self.assertEqual(row['v__tail_mean'], 1.0)
        self.assertEqual(row['length'], 8.0)
        self.assertNotIn('time_frac__mean', row)

    def test_folds_group_subjects(self):
        rows = []
        for s in range(20):
            for g in ['A', 'B', 'C']:
                rows.append({'sequence_id': f'{s}{g}', 'subject': f'S{s}', 'gesture': g})
        folds = make_folds(pd.DataFrame(rows))
        self.assertTrue((folds.groupby('subject').fold.nunique() == 1).all())
        self.assertEqual(sorted(folds.fold.unique()), [0, 1, 2, 3, 4])


class AugmentTests(unittest.TestCase):
    def test_rotations_keep_physics_consistent(self):
        seq = clean_sequence(recording())
        rng = np.random.default_rng(0)
        self.assertLess(gravity_consistency_error(wrist_rotation(seq, rng, 7)), 1e-9)
        self.assertLess(gravity_consistency_error(yaw_rotation(seq, rng, 60)), 1e-9)

    def test_augment_runs_and_keeps_shapes(self):
        seq = clean_sequence(recording(120))
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = augment(seq, rng, AugmentConfig())
            self.assertEqual(a.acc.shape[0], a.tof.shape[0])
            self.assertEqual(len(a.meta), a.n)
            f = sequence_features(a)
            self.assertTrue(np.isfinite(f['acc_x']).all())

    def test_time_stretch_length(self):
        seq = clean_sequence(recording(100))
        out = time_stretch(seq, np.random.default_rng(0), 0.2)
        self.assertTrue(80 <= out.n <= 120)


class LabelTests(unittest.TestCase):
    def test_score(self):
        score, binary, macro = competition_score(['Neck - scratch', 'Wave hello'], ['Neck - scratch', 'Text on phone'])
        self.assertEqual((score, binary, macro), (1.0, 1.0, 1.0))
        score, _, _ = competition_score(['Neck - scratch'], ['Wave hello'])
        self.assertEqual(score, 0.0)

    def test_phase_and_initial_state(self):
        self.assertEqual(phase_of('Performs gesture'), 'gesture')
        self.assertEqual(phase_of('Hand at target location'), 'pause')
        self.assertEqual(phase_of('Moves hand to target location'), 'transition')
        self.assertEqual(initial_state(pd.Series(['Relaxes and moves hand to target location'])), 'relaxed')

    def test_sequence_labels(self):
        meta = pd.DataFrame({'gesture': ['Wave hello'] * 2, 'orientation': ['Seated Straight'] * 2,
                             'behavior': ['Moves hand to target location', 'Performs gesture']})
        labels = sequence_labels(meta)
        self.assertEqual(labels['score_class'], 'non_target')
        self.assertEqual(labels['is_target'], 0)
        self.assertEqual(labels['combined'], 'Wave hello | Seated Straight | ready')


if __name__ == '__main__':
    unittest.main()
