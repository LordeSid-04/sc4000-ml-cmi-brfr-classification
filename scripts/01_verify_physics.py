"""Check every physical assumption of the pipeline on the real training data.

Answers the open questions of guide section 11 with numbers, and chooses the
left-hand mirror and upside-down corrections by matching subjects against the
right-handed population instead of trusting write-ups.

    python scripts/01_verify_physics.py [--data PATH]

Writes outputs/verification/physics.json and figures next to it.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import quaternion as quat  # noqa: E402
from preprocessing.cleaning import clean_sequence  # noqa: E402
from preprocessing.constants import ACC, GRAVITY, ROT, THM, TOF  # noqa: E402
from preprocessing.features import gravity_in_device_frame  # noqa: E402
from preprocessing.data_loading import load_train  # noqa: E402
from preprocessing.labels import phase_of  # noqa: E402
from preprocessing.orientation import IMAGE_OPS, UPSIDE_DOWN_SUBJECTS, FrameChange, image_op  # noqa: E402

OUT = Path('outputs/verification')


def describe(series):
    q = series.quantile([0, .05, .25, .5, .75, .9, .95, .99, 1])
    return {f'p{int(k * 100)}': float(v) for k, v in q.items()} | {'mean': float(series.mean())}


def basic_facts(train, demo):
    seq = train.groupby('sequence_id').agg(
        length=('sequence_counter', 'size'), gesture=('gesture', 'first'),
        subject=('subject', 'first'), orientation=('orientation', 'first'),
        sequence_type=('sequence_type', 'first'))
    r = {
        'rows': int(len(train)), 'sequences': int(len(seq)),
        'subjects': int(seq.subject.nunique()), 'gestures': int(seq.gesture.nunique()),
        'columns': int(train.shape[1]),
        'target_share': float((seq.sequence_type == 'Target').mean()),
        'sequences_per_gesture': seq.gesture.value_counts().to_dict(),
        'orientations': seq.orientation.value_counts().to_dict(),
        'length': describe(seq.length),
        'length_kept_whole': {str(L): float((seq.length <= L).mean()) for L in (64, 96, 103, 128, 160, 200)},
        'handedness_counts': demo['handedness'].value_counts().to_dict(),
        'sequences_per_subject': describe(seq.groupby('subject').size()),
        'demographic_columns': list(demo.columns),
    }
    # Consistency checks
    per_seq = train.groupby('sequence_id')
    r['max_distinct_per_sequence'] = {c: int(per_seq[c].nunique().max()) for c in ['gesture', 'subject', 'orientation', 'sequence_type']}
    r['duplicate_counters'] = int(train.duplicated(['sequence_id', 'sequence_counter']).sum())
    steps = per_seq['sequence_counter'].diff().dropna()
    r['counter_steps_not_1'] = int((steps != 1).sum())
    r['behavior_values'] = train['behavior'].value_counts().to_dict()
    r['phase_values'] = train['phase'].value_counts().to_dict()
    first_behavior = per_seq['behavior'].first()
    r['first_behavior_values'] = first_behavior.value_counts().to_dict()
    gesture_rows = train['behavior'].map(phase_of).eq('gesture').groupby(train['sequence_id']).sum()
    r['sequences_without_gesture_rows'] = int((gesture_rows == 0).sum())
    pos = train.groupby('sequence_id').cumcount() / train.groupby('sequence_id')['sequence_counter'].transform('size')
    start = pos[train['phase'] == 'Gesture'].groupby(train['sequence_id']).min()
    r['gesture_start_fraction'] = describe(start)
    combos = train.groupby('sequence_id').agg(subject=('subject', 'first'), gesture=('gesture', 'first'),
                                               orientation=('orientation', 'first'))
    combos['start'] = first_behavior
    r['max_repeats_subject_gesture_orientation'] = int(combos.groupby(['subject', 'gesture', 'orientation']).size().max())
    r['subject_gesture_orientation_combos'] = describe(combos.groupby(['subject', 'gesture', 'orientation']).size())
    return r, seq


def missingness(train):
    r = {}
    g = train.groupby('sequence_id')
    for name, cols in {'acc': ACC, 'rot': ROT, 'thm': THM, 'tof': TOF}.items():
        nan = train[cols].isna()
        row_any = nan.any(axis=1)
        row_all = nan.all(axis=1)
        whole = row_all.groupby(train['sequence_id']).all()
        some = row_any.groupby(train['sequence_id']).any()
        r[name] = {
            'cells_nan_pct': float(100 * nan.to_numpy().mean()),
            'rows_any_nan_pct': float(100 * row_any.mean()),
            'sequences_fully_missing': int(whole.sum()),
            'sequences_partly_missing': int((some & ~whole).sum()),
        }
    for k in range(1, 6):
        r[f'thm_{k}_rows_nan_pct'] = float(100 * train[f'thm_{k}'].isna().mean())
        r[f'tof_{k}_rows_nan_pct'] = float(100 * train[f'tof_{k}_v0'].isna().mean())
    tof = train[TOF].to_numpy(dtype=np.float32)
    finite = np.isfinite(tof)
    r['tof_minus_one_pct_of_finite'] = float(100 * (tof[finite] == -1).mean())
    valid = tof[finite & (tof >= 0)]
    r['tof_valid_range'] = [float(valid.min()), float(valid.max())]
    r['tof_other_negative_values'] = int((finite & (tof < 0) & (tof != -1)).sum())
    r['imu_only_sequences'] = int((g[THM + TOF].apply(lambda s: s.isna().all().all())).sum())
    return r


def quaternion_checks(train):
    acc = train[ACC].to_numpy(dtype=np.float64)
    q = train[ROT].to_numpy(dtype=np.float64)
    norm = np.linalg.norm(q, axis=1)
    ok = np.isfinite(q).all(axis=1) & (norm > 1e-8)
    qn = q[ok] / norm[ok, None]
    a = acc[ok]
    world = quat.rotate(qn, a)
    world_conj = quat.rotate(quat.conjugate(qn), a)
    r = {
        'acc_magnitude_mean': float(np.nanmean(np.linalg.norm(acc, axis=1))),
        'quaternion_norm': describe(pd.Series(norm[ok])),
        'world_mean_q_v_qconj': world.mean(axis=0).round(3).tolist(),
        'world_mean_qconj_v_q': world_conj.mean(axis=0).round(3).tolist(),
    }
    r['convention_device_to_world'] = bool(abs(world.mean(axis=0)[2] - GRAVITY) < abs(world_conj.mean(axis=0)[2] - GRAVITY))
    # Sign flips between consecutive rows of the same sequence
    sid = train['sequence_id'].to_numpy()
    same = (sid[1:] == sid[:-1]) & ok[1:] & ok[:-1]
    qq = np.where(ok[:, None], q / np.where(norm > 0, norm, 1)[:, None], np.nan)
    dot = np.sum(qq[1:] * qq[:-1], axis=1)
    flips = same & (dot < 0)
    r['sign_flip_rows'] = int(flips.sum())
    r['sign_flip_sequences'] = int(len(np.unique(sid[1:][flips])))
    r['sign_flip_rows_with_large_jump'] = int((same & (dot < -0.5)).sum())
    return r


def per_sequence_stats(train, demo):
    """Gesture-phase averages of gravity direction, thermopiles and ToF images."""
    hand = demo.set_index('subject')['handedness']
    rows = []
    for sid, s in train.groupby('sequence_id', sort=True):
        seq = clean_sequence(s)
        g, from_quat = gravity_in_device_frame(seq)
        gesture_rows = np.array(seq.meta['behavior'].map(phase_of).eq('gesture').to_numpy(), dtype=bool)
        if not gesture_rows.any():
            gesture_rows[:] = True
        gdir = (g[gesture_rows] / GRAVITY).mean(axis=0)
        with np.errstate(invalid='ignore'):
            thm = np.nanmean(seq.thm[gesture_rows], axis=0) if not seq.thm_missing else np.full(5, np.nan)
            tof = np.nanmean(seq.tof[gesture_rows], axis=0) if not seq.tof_missing else np.full((5, 64), np.nan)
        subject = s['subject'].iloc[0]
        rows.append({'sequence_id': sid, 'subject': subject, 'gesture': s['gesture'].iloc[0],
                     'handedness': int(hand.get(subject, -1)), 'from_quat': from_quat,
                     'gdir': gdir, 'thm': thm, 'tof': tof})
    return rows


def mirror_gdir(g, axis):
    g = g.copy()
    if axis is not None:
        g[axis] = -g[axis]
    return g


def rotate180_gdir(g, axis):
    g = -g.copy()
    g[axis] = -g[axis]
    return g


def reference(rows, key, select):
    out = {}
    for gesture in sorted({r['gesture'] for r in rows}):
        vals = [r[key] for r in rows if r['gesture'] == gesture and select(r)]
        vals = [v for v in vals if np.isfinite(v).all()]
        if vals:
            out[gesture] = np.mean(vals, axis=0)
    return out


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gdir_score(rows, transform, ref):
    return float(np.mean([cosine(transform(r['gdir']), ref[r['gesture']]) for r in rows]))


def thm_error(rows, perm, ref):
    errs = []
    for r in rows:
        if not np.isfinite(r['thm']).all() or r['gesture'] not in ref:
            continue
        x = r['thm'][list(perm)]
        x = x - x.mean()
        y = ref[r['gesture']] - ref[r['gesture']].mean()
        errs.append(np.mean((x - y) ** 2))
    return float(np.mean(errs)) if errs else float('nan')


def tof_sensor_error(rows, perm, ref):
    errs = []
    for r in rows:
        if not np.isfinite(r['tof']).all() or r['gesture'] not in ref:
            continue
        x = r['tof'][list(perm)].mean(axis=1)
        errs.append(np.mean((x - ref[r['gesture']].mean(axis=1)) ** 2))
    return float(np.mean(errs)) if errs else float('nan')


def best_image_ops(rows, perm, ref):
    """For each target sensor, the image symmetry that best matches the reference."""
    result = {}
    for k in range(5):
        scores = {}
        for op in IMAGE_OPS:
            errs = []
            for r in rows:
                if not np.isfinite(r['tof']).all() or r['gesture'] not in ref:
                    continue
                img = image_op(r['tof'][perm[k]].reshape(8, 8), op).reshape(64)
                errs.append(np.mean((img - ref[r['gesture']][k]) ** 2))
            scores[op] = float(np.mean(errs)) if errs else float('nan')
        result[k + 1] = scores
    return result


def choose_permutation(rows, ref_thm, ref_tof):
    results = []
    for perm in itertools.permutations(range(5)):
        results.append({'perm': [p + 1 for p in perm],
                        'thm_error': thm_error(rows, perm, ref_thm),
                        'tof_error': tof_sensor_error(rows, perm, ref_tof)})
    base_t = next(r for r in results if r['perm'] == [1, 2, 3, 4, 5])
    for r in results:
        r['combined'] = r['thm_error'] / base_t['thm_error'] + r['tof_error'] / base_t['tof_error']
    return sorted(results, key=lambda r: r['combined'])


def apply_gdir_mirror_rows(rows, axis, perm=None):
    out = []
    for r in rows:
        r = dict(r)
        r['gdir'] = mirror_gdir(r['gdir'], axis)
        if perm is not None:
            r['thm'] = r['thm'][list(perm)]
            r['tof'] = r['tof'][list(perm)]
        out.append(r)
    return out


def orientation_analysis(rows):
    report = {}
    right = [r for r in rows if r['handedness'] == 1 and r['subject'] not in UPSIDE_DOWN_SUBJECTS]
    left = [r for r in rows if r['handedness'] == 0 and r['subject'] not in UPSIDE_DOWN_SUBJECTS]
    ref_g = reference(right, 'gdir', lambda r: True)
    ref_thm = reference(right, 'thm', lambda r: True)
    ref_tof = reference(right, 'tof', lambda r: True)

    # 1) Which device axis does left-hand mirroring flip?
    axis_scores = {name: gdir_score(left, lambda g, a=a: mirror_gdir(g, a), ref_g)
                   for name, a in [('none', None), ('x', 0), ('y', 1), ('z', 2)]}
    report['left_mirror_axis_scores'] = axis_scores
    report['right_self_score'] = gdir_score(right, lambda g: g, ref_g)
    best_axis = max(['x', 'y', 'z'], key=lambda k: axis_scores[k])
    axis = 'xyz'.index(best_axis)
    report['left_mirror_axis'] = best_axis

    # 2) Which sensors trade places?
    perms = choose_permutation(left, ref_thm, ref_tof)
    report['left_permutations_top10'] = perms[:10]
    report['left_permutation_identity'] = next(r for r in perms if r['perm'] == [1, 2, 3, 4, 5])
    perm = [p - 1 for p in perms[0]['perm']]
    # 3) Which image flip per sensor?
    ops = best_image_ops(left, perm, ref_tof)
    report['left_image_op_errors'] = ops
    best_ops = [min(ops[k + 1], key=ops[k + 1].get) for k in range(5)]
    report['left_image_ops'] = best_ops

    # 4) Per-subject agreement with the population after mirroring.
    canon = [dict(r, gdir=mirror_gdir(r['gdir'], axis if r['handedness'] == 0 else None)) for r in rows]
    ref_all = reference([r for r in canon if r['subject'] not in UPSIDE_DOWN_SUBJECTS], 'gdir', lambda r: True)
    subj = {}
    for s in sorted({r['subject'] for r in canon}):
        rs = [r for r in canon if r['subject'] == s]
        subj[s] = {
            'handedness': rs[0]['handedness'],
            'as_is': gdir_score(rs, lambda g: g, ref_all),
            **{f'rot180_{n}': gdir_score(rs, lambda g, a=a: rotate180_gdir(g, a), ref_all) for n, a in [('x', 0), ('y', 1), ('z', 2)]},
        }
    report['subject_scores'] = subj
    ranked = sorted(subj, key=lambda s: subj[s]['as_is'])
    report['lowest_agreement_subjects'] = [{'subject': s, **subj[s]} for s in ranked[:8]]
    flagged = [s for s in subj if max(subj[s]['rot180_x'], subj[s]['rot180_y'], subj[s]['rot180_z']) > subj[s]['as_is'] + 0.2]
    report['subjects_better_rotated_180'] = flagged

    # 5) Upside-down correction: axis and sensor permutation for the named subjects.
    ud = [r for r in canon if r['subject'] in UPSIDE_DOWN_SUBJECTS]
    if ud:
        rot_scores = {n: gdir_score(ud, lambda g, a=a: rotate180_gdir(g, a), ref_all) for n, a in [('x', 0), ('y', 1), ('z', 2)]}
        rot_scores['none'] = gdir_score(ud, lambda g: g, ref_all)
        report['upside_down_rotation_scores'] = rot_scores
        # Thermopile / ToF of these subjects after their own mirroring.
        canon_full = []
        for r in rows:
            r2 = dict(r)
            if r['handedness'] == 0:
                r2['thm'], r2['tof'] = r['thm'][perm], np.stack([image_op(r['tof'][perm[k]].reshape(8, 8), best_ops[k]).reshape(64) for k in range(5)])
            canon_full.append(r2)
        ref_thm_c = reference([r for r in canon_full if r['subject'] not in UPSIDE_DOWN_SUBJECTS], 'thm', lambda r: True)
        ref_tof_c = reference([r for r in canon_full if r['subject'] not in UPSIDE_DOWN_SUBJECTS], 'tof', lambda r: True)
        ud_full = [r for r in canon_full if r['subject'] in UPSIDE_DOWN_SUBJECTS]
        ud_perms = choose_permutation(ud_full, ref_thm_c, ref_tof_c)
        report['upside_down_permutations_top10'] = ud_perms[:10]
        report['upside_down_permutation_identity'] = next(r for r in ud_perms if r['perm'] == [1, 2, 3, 4, 5])
        ud_perm = [p - 1 for p in ud_perms[0]['perm']]
        report['upside_down_image_op_errors'] = best_image_ops(ud_full, ud_perm, ref_tof_c)
    return report


def figures(rows, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axis = 'xyz'.index(report['left_mirror_axis'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, title, mirror in [(axes[0], 'Raw device frame', False), (axes[1], 'After left-hand mirroring', True)]:
        for s in sorted({r['subject'] for r in rows}):
            rs = [r for r in rows if r['subject'] == s]
            g = np.mean([mirror_gdir(r['gdir'], axis if (mirror and r['handedness'] == 0) else None) for r in rs], axis=0)
            color = 'crimson' if s in UPSIDE_DOWN_SUBJECTS else ('tab:orange' if rs[0]['handedness'] == 0 else 'tab:blue')
            ax.scatter(g[0], g[1], c=color, s=30, edgecolor='k', lw=0.3)
            if s in UPSIDE_DOWN_SUBJECTS:
                ax.annotate(s, g[:2], fontsize=7)
        ax.axhline(0, c='grey', lw=0.5); ax.axvline(0, c='grey', lw=0.5)
        ax.set_title(title); ax.set_xlabel('mean gravity direction x'); ax.set_ylabel('mean gravity direction y')
    fig.suptitle('Per-subject gravity direction during the gesture (blue right, orange left, red flagged upside-down)')
    fig.tight_layout()
    fig.savefig(OUT / 'subject_gravity_direction.png', dpi=140)
    plt.close(fig)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train, demo = load_train(args.data)
    train = train.sort_values(['sequence_id', 'sequence_counter'], kind='stable').reset_index(drop=True)
    print('loaded', train.shape, flush=True)
    report = {}
    report['basic'], _ = basic_facts(train, demo)
    print('basic done', flush=True)
    report['missing'] = missingness(train)
    print('missingness done', flush=True)
    report['quaternion'] = quaternion_checks(train)
    print('quaternion done', flush=True)
    rows = per_sequence_stats(train, demo)
    print('per-sequence stats done', flush=True)
    report['orientation'] = orientation_analysis(rows)
    figures(rows, report['orientation'])
    (OUT / 'physics.json').write_text(json.dumps(to_jsonable(report), indent=2), encoding='utf-8')
    o = report['orientation']
    print(json.dumps(to_jsonable({
        'convention_device_to_world': report['quaternion']['convention_device_to_world'],
        'world_mean': report['quaternion']['world_mean_q_v_qconj'],
        'sign_flips': [report['quaternion']['sign_flip_rows'], report['quaternion']['sign_flip_sequences']],
        'handedness_counts': report['basic']['handedness_counts'],
        'left_mirror_axis_scores': o['left_mirror_axis_scores'],
        'left_perm_best': o['left_permutations_top10'][:3],
        'left_perm_identity': o['left_permutation_identity'],
        'left_image_ops': o['left_image_ops'],
        'lowest_subjects': o['lowest_agreement_subjects'][:5],
        'better_rotated': o['subjects_better_rotated_180'],
        'upside_down_rotation_scores': o.get('upside_down_rotation_scores'),
        'upside_down_perm_best': o.get('upside_down_permutations_top10', [])[:3],
    }), indent=1))


if __name__ == '__main__':
    main()
