"""Exploratory data analysis: the six questions that decide the preprocessing.

    python scripts/01_explore_data.py

1. Size and structure   - sequences, rows, subjects, sequence lengths
2. Labels               - sequences per gesture, target vs non-target, rare targets
3. Behaviour phases     - which phases a sequence goes through, how long each is
4. Sensors              - raw signals of a few gestures side by side
5. Missing data         - how often thermopile / ToF are missing, and for whom
6. Demographics         - handedness, age, height, sex

Figures go to docs/figures/eda/ (shown in the README); every number is
written to outputs/eda/summary.json.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing.data_loading import load_train  # noqa: E402

FIG = ROOT / 'docs' / 'figures' / 'eda'
OUT = ROOT / 'outputs' / 'eda'
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
summary = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'{name}.png', dpi=120)
    plt.close(fig)


train, demo = load_train()
train = train.sort_values(['sequence_id', 'sequence_counter']).reset_index(drop=True)
acc_cols = ['acc_x', 'acc_y', 'acc_z']
rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_cols = [c for c in train.columns if c.startswith('tof_')]

seq = (train.groupby('sequence_id')
       .agg(length=('sequence_counter', 'size'), gesture=('gesture', 'first'),
            sequence_type=('sequence_type', 'first'), subject=('subject', 'first'),
            orientation=('orientation', 'first'))
       .reset_index().merge(demo, on='subject', how='left'))

# 1. Size and structure ------------------------------------------------------
lengths = seq.length
summary['1_size'] = {
    'sequences': len(seq), 'rows': len(train), 'subjects': int(seq.subject.nunique()),
    'gestures': int(seq.gesture.nunique()), 'orientations': int(seq.orientation.nunique()),
    'sequences_per_subject_median': float(seq.groupby('subject').size().median()),
    'length_min': int(lengths.min()), 'length_median': float(lengths.median()),
    'length_mean': round(float(lengths.mean()), 1), 'length_p95': float(lengths.quantile(.95)),
    'length_p99': float(lengths.quantile(.99)), 'length_max': int(lengths.max()),
    'share_longer_than_128': round(float((lengths > 128).mean()), 4),
}
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(lengths.clip(upper=300), bins=75, color='tab:blue')
for q, label in [(.5, 'median'), (.95, 'p95')]:
    v = lengths.quantile(q)
    ax.axvline(v, ls='--', c='k', lw=1)
    ax.text(v + 3, ax.get_ylim()[1] * .85, f'{label} = {v:.0f}')
ax.axvline(128, c='tab:red', lw=1.5)
ax.text(131, ax.get_ylim()[1] * .6, 'L = 128 (chosen)', color='tab:red')
ax.set_xlabel('rows per sequence (clipped at 300; max is %d)' % lengths.max())
ax.set_ylabel('sequences')
ax.set_title('1. Sequence length')
save(fig, '1_sequence_length')

# 2. Labels -------------------------------------------------------------------
counts = seq.groupby(['gesture', 'sequence_type']).size().reset_index(name='n').sort_values('n')
target = counts[counts.sequence_type == 'Target']
summary['2_labels'] = {
    'per_gesture': {r.gesture: int(r.n) for r in counts.itertuples()},
    'target_gestures': int(len(target)), 'non_target_gestures': int((counts.sequence_type != 'Target').sum()),
    'target_sequences': int((seq.sequence_type == 'Target').sum()),
    'non_target_sequences': int((seq.sequence_type != 'Target').sum()),
    'rarest_target': target.iloc[0].gesture, 'rarest_target_n': int(target.iloc[0].n),
    'largest_target': target.iloc[-1].gesture, 'largest_target_n': int(target.iloc[-1].n),
}
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(counts.gesture, counts.n,
        color=counts.sequence_type.map({'Target': 'tab:red', 'Non-Target': 'tab:gray'}))
for y, n in enumerate(counts.n):
    ax.text(n + 5, y, str(n), va='center', fontsize=8)
ax.set_xlabel('sequences')
ax.set_title('2. Sequences per gesture (red = BFRB target, grey = non-target)')
save(fig, '2_class_balance')

# 3. Behaviour phases -----------------------------------------------------------
order = ['Relaxes and moves hand to target location', 'Moves hand to target location',
         'Hand at target location', 'Performs gesture']
beh = train.groupby(['sequence_id', 'behavior']).size().unstack(fill_value=0)
pos = train.groupby('sequence_id').cumcount()
rel = pos / train.groupby('sequence_id')['sequence_counter'].transform('size')
gest = train.phase == 'Gesture'
gesture_start = rel[gest].groupby(train.sequence_id[gest]).min()
last_row_is_gesture = train.groupby('sequence_id').phase.last().eq('Gesture')
summary['3_phases'] = {
    'phase_values': sorted(train.phase.unique().tolist()),
    'behavior_rows_median_per_sequence': {b: float(beh[b][beh[b] > 0].median()) for b in order if b in beh},
    'sequences_with_behavior': {b: int((beh[b] > 0).sum()) for b in order if b in beh},
    'gesture_phase_share_of_rows': round(float(gest.mean()), 3),
    'gesture_starts_at_fraction_median': round(float(gesture_start.median()), 3),
    'gesture_starts_at_fraction_p10_p90': [round(float(gesture_start.quantile(q)), 3) for q in (.1, .9)],
    'sequences_ending_in_gesture': round(float(last_row_is_gesture.mean()), 4),
    'sequences_without_gesture_rows': int(len(seq) - len(gesture_start)),
}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
data = [beh[b][beh[b] > 0] for b in order if b in beh]
axes[0].boxplot(data, orientation='horizontal', showfliers=False)
axes[0].set_yticks(range(1, len(data) + 1), [b.replace(' target location', '...') for b in order if b in beh], fontsize=8)
axes[0].set_xlabel('rows per sequence')
axes[0].set_title('3a. Length of each behaviour')
axes[1].hist(gesture_start, bins=50, color='tab:orange')
axes[1].set_xlabel('where the gesture phase starts (0 = start, 1 = end of sequence)')
axes[1].set_title('3b. Gesture starts mid-sequence and runs to the end')
save(fig, '3_phases')

# 4. Sensors --------------------------------------------------------------------
show = ['Above ear - pull hair', 'Neck - scratch', 'Text on phone', 'Wave hello']
show = [g for g in show if g in set(seq.gesture)]
has_tof = train.groupby('sequence_id').tof_1_v0.apply(lambda s: s.notna().all())
fig, axes = plt.subplots(4, len(show), figsize=(4 * len(show), 9), sharex='col')
for c, g in enumerate(show):
    sid = seq[(seq.gesture == g) & seq.sequence_id.map(has_tof)].sort_values('sequence_id').sequence_id.iloc[0]
    s = train[train.sequence_id == sid].reset_index(drop=True)
    g_idx = s.index[s.phase == 'Gesture']
    tof_mean = s[tof_cols].where(s[tof_cols] >= 0)
    tof_mean = pd.DataFrame({f'tof_{i}': tof_mean[[f'tof_{i}_v{p}' for p in range(64)]].mean(axis=1)
                             for i in range(1, 6)})
    for r, (frame, name) in enumerate([(s[acc_cols], 'acceleration (m/s²)'), (s[rot_cols], 'rotation (quaternion)'),
                                       (s[thm_cols], 'thermopile (°C)'), (tof_mean, 'ToF mean distance')]):
        ax = axes[r, c]
        ax.plot(frame.to_numpy(), lw=1)
        if len(g_idx):
            ax.axvspan(g_idx.min(), g_idx.max(), color='orange', alpha=.15)
        if c == 0:
            ax.set_ylabel(name, fontsize=8)
        if r == 0:
            ax.set_title(f'{g}\n({sid})', fontsize=9)
        if c == len(show) - 1:
            ax.legend(frame.columns, fontsize=6, loc='upper left')
    axes[-1, c].set_xlabel('row')
fig.suptitle('4. Raw sensors for four gestures (orange = gesture phase)')
save(fig, '4_sensors')

# 5. Missing data ---------------------------------------------------------------
per_seq = train.groupby('sequence_id').agg(
    rot_all=('rot_w', lambda s: s.isna().all()), rot_any=('rot_w', lambda s: s.isna().any()),
    thm_all=('thm_1', lambda s: s.isna().all()), thm_any=('thm_1', lambda s: s.isna().any()),
    tof_all=('tof_1_v0', lambda s: s.isna().all()), tof_any=('tof_1_v0', lambda s: s.isna().any()))
per_seq = per_seq.join(seq.set_index('sequence_id').subject)
by_subject = per_seq[per_seq.tof_all | per_seq.thm_all].groupby('subject').size().sort_values(ascending=False)
tof_vals = train[tof_cols].to_numpy(dtype=np.float32)
row_nan = {g: float(train[cols].isna().any(axis=1).mean()) for g, cols in
           [('acc', acc_cols), ('rot', rot_cols), ('thm', thm_cols), ('tof', tof_cols)]}
dead_sensor = {}
for i in range(1, 6):
    dead_sensor[f'thm_{i}'] = int(train.groupby('sequence_id')[f'thm_{i}'].apply(lambda s: s.isna().all()).sum())
    dead_sensor[f'tof_{i}'] = int(train.groupby('sequence_id')[f'tof_{i}_v0'].apply(lambda s: s.isna().all()).sum())
summary['5_missing'] = {
    'rows_with_any_nan': {k: round(v, 4) for k, v in row_nan.items()},
    'sequences_rotation_fully_missing': int(per_seq.rot_all.sum()),
    'sequences_rotation_partly_missing': int((per_seq.rot_any & ~per_seq.rot_all).sum()),
    'sequences_thm_fully_missing': int(per_seq.thm_all.sum()),
    'sequences_thm_partly_missing': int((per_seq.thm_any & ~per_seq.thm_all).sum()),
    'sequences_tof_fully_missing': int(per_seq.tof_all.sum()),
    'sequences_tof_partly_missing': int((per_seq.tof_any & ~per_seq.tof_all).sum()),
    'sequences_dead_sensor_whole_recording': dead_sensor,
    'tof_pixels_minus_one_share': round(float(np.mean(tof_vals == -1)), 4),
    'subjects_with_fully_missing_thm_or_tof': {k: int(v) for k, v in by_subject.items()},
}
del tof_vals
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].bar(dead_sensor.keys(), dead_sensor.values(),
            color=['tab:orange' if k.startswith('thm') else 'tab:blue' for k in dead_sensor])
axes[0].set_ylabel('sequences where the sensor is dead throughout')
axes[0].set_title('5a. Dead sensors per recording')
axes[0].tick_params(axis='x', rotation=45)
axes[1].bar(by_subject.index, by_subject.values, color='tab:gray')
axes[1].set_ylabel('sequences with THM/ToF fully missing')
axes[1].set_title('5b. Which subjects lose THM/ToF')
axes[1].tick_params(axis='x', rotation=60, labelsize=7)
save(fig, '5_missing')

# 6. Demographics ---------------------------------------------------------------
summary['6_demographics'] = {
    'handedness_1_right_0_left': {int(k): int(v) for k, v in demo.handedness.value_counts().items()},
    'adult_child_1_adult': {int(k): int(v) for k, v in demo.adult_child.value_counts().items()},
    'sex_1_male': {int(k): int(v) for k, v in demo.sex.value_counts().items()},
    'age_min_median_max': [int(demo.age.min()), float(demo.age.median()), int(demo.age.max())],
    'height_cm_min_median_max': [float(demo.height_cm.min()), float(demo.height_cm.median()), float(demo.height_cm.max())],
    'left_handed_sequences': int((seq.handedness == 0).sum()),
}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, col, labels in [(axes[0], 'handedness', {0: 'left', 1: 'right'}),
                        (axes[1], 'sex', {0: 'female', 1: 'male'})]:
    vc = demo[col].map(labels).value_counts()
    ax.bar(vc.index, vc.values, color='tab:purple')
    ax.set_title(f'6. {col} (subjects)')
axes[2].hist(demo.age, bins=20, color='tab:green')
axes[2].set_title('age (years)')
axes[3].hist(demo.height_cm, bins=20, color='tab:green')
axes[3].set_title('height (cm)')
save(fig, '6_demographics')

(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
print(f'\nFigures: {FIG}\nNumbers: {OUT / "summary.json"}')
