"""Figures for the preprocessing report, drawn from the saved results.

    python scripts/07_report_figures.py

Reads outputs/ablation/trees.json, outputs/verification/physics.json,
data/processed/sequences.csv and the raw training data; writes PNGs to
docs/figures/results/.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import quaternion as quat  # noqa: E402
from preprocessing.step1_cleaning import clean_sequence  # noqa: E402
from preprocessing.constants import GRAVITY, ROT  # noqa: E402
from preprocessing.data_loading import load_train  # noqa: E402
from preprocessing.step3_features import gravity_in_device_frame  # noqa: E402
from preprocessing.step2_orientation import UPSIDE_DOWN_SUBJECTS  # noqa: E402

OUT = Path('docs/figures/results')
# Reference palette, first slots (validated as a set): blue, orange, aqua.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'legend.frameon': False,
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})


def ablation_steps(results):
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    labels = ['A\nraw', 'B\ncleaning', 'C\nkeep end', 'D\nupside-\ndown fix', 'E\nmirror\nleft hand',
              'F\ngravity\nremoval', 'G\nengineered\n(+smooth)', 'H\nfinal\n(no smooth)']
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for key, color, name in [('full', BLUE, 'All sensors'), ('imu', ORANGE, 'IMU only')]:
        mean = np.array([results[r][f'{key}_mean'] for r in rows])
        std = np.array([results[r][f'{key}_std'] for r in rows])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, lw=0)
        ax.plot(x, mean, color=color, lw=2, marker='o', ms=6, mec='white', mew=1.5, label=name)
        for i in (0, len(rows) - 1):
            ax.annotate(f'{mean[i]:.3f}', (x[i], mean[i]), textcoords='offset points',
                        xytext=(0, 9), ha='center', color=INK, fontsize=8.5)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel('Competition score (5-fold mean)')
    ax.set_ylim(0.63, 0.85)
    ax.legend(loc='upper left', ncol=2)
    ax.set_title('Each preprocessing step added in turn (band = ±1 std across folds)', loc='left')
    fig.savefig(OUT / 'fig_ablation_steps.png')
    plt.close(fig)


def leave_one_out(results):
    base = {f['fold']: f for f in results['G']['folds']}
    variants = [('H', 'remove smoothing (= final H)'),
                ('G-no_gravity', 'remove gravity removal'),
                ('G-no_continuity', 'remove quaternion continuity'),
                ('G-no_upside_down', 'remove upside-down fix'),
                ('G-drop_upside_down', 'delete upside-down subjects'),
                ('G-no_mirror', 'remove left-hand mirroring'),
                ('G-tof_near', 'ToF -1 as "touching" (old)'),
                ('G-no_regions', 'remove ToF 3x3 regions')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, key, color, title in [(axes[0], 'full', BLUE, 'All sensors'), (axes[1], 'imu', ORANGE, 'IMU only')]:
        for yi, (name, _) in zip(y, variants):
            deltas = np.array([f[key] - base[f['fold']][key] for f in results[name]['folds']])
            ax.barh(yi, deltas.mean(), color=color, height=0.55, alpha=0.85)
            ax.scatter(deltas, np.full(5, yi), s=12, color=INK, zorder=3, lw=0)
            ax.annotate(f'{deltas.mean() + 0.0:+.3f}'.replace('-0.000', '0.000'), (max(deltas.max(), 0) + 0.002, yi), va='center', fontsize=7.5, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Score change vs full pipeline G')
        ax.set_xlim(-0.03, 0.03)
    axes[0].set_yticks(y, [v[1] for v in variants], fontsize=8)
    fig.suptitle('Leave-one-out: bar = mean change, dots = the 5 folds (left of 0 = the step helps)',
                 x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_leave_one_out.png')
    plt.close(fig)


def lengths(seqs):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.hist(seqs['rows'].clip(upper=300), bins=np.arange(25, 305, 5), color=BLUE, edgecolor='white', lw=0.5)
    ax.axvline(128, color=INK, lw=1.2, ls='--')
    kept = (seqs['rows'] <= 128).mean()
    ax.annotate(f'L = 128 keeps {kept:.0%} of recordings whole', (130, ax.get_ylim()[1] * 0.85), fontsize=8.5, color=INK)
    ax.set_xlabel('Rows per recording (values above 300 shown at 300)')
    ax.set_ylabel('Recordings')
    ax.set_title('Recording length: median 59, 95th percentile 127, max 700', loc='left')
    fig.savefig(OUT / 'fig_lengths.png')
    plt.close(fig)


def subject_gravity(train, demo):
    hand = demo.set_index('subject')['handedness']
    rows = []
    for sid, s in train.groupby('sequence_id', sort=False):
        seq = clean_sequence(s)
        g, _ = gravity_in_device_frame(seq)
        rows.append({'subject': s['subject'].iloc[0], 'gx': g[:, 0].mean() / GRAVITY, 'gy': g[:, 1].mean() / GRAVITY})
    per = pd.DataFrame(rows).groupby('subject').mean()
    per['hand'] = hand.reindex(per.index).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
    for ax, corrected in [(axes[0], False), (axes[1], True)]:
        gx, gy = per['gx'].to_numpy().copy(), per['gy'].to_numpy().copy()
        ud = per.index.isin(UPSIDE_DOWN_SUBJECTS)
        left = (per['hand'] == 0).to_numpy()
        if corrected:
            gx[ud], gy[ud] = -gx[ud], -gy[ud]          # 180 deg about z
            gx[left] = -gx[left]                         # mirror x
        for mask, color, name in [(~left & ~ud, BLUE, 'right-handed'), (left, ORANGE, 'left-handed'),
                                  (ud, AQUA, 'upside-down (2)')]:
            ax.scatter(gx[mask], gy[mask], s=34, color=color, edgecolor='white', lw=1, label=name, zorder=3)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title('Raw device frame' if not corrected else 'After upside-down fix + mirroring', loc='left')
        ax.set_xlabel('Mean gravity direction, x')
    axes[0].set_ylabel('Mean gravity direction, y')
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc='lower right', fontsize=8)
    fig.suptitle('One dot per subject: corrections move outliers into the main group',
                 x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_subject_gravity.png')
    plt.close(fig)


def sign_flip(train):
    for sid, s in train.groupby('sequence_id', sort=False):
        q = s[ROT].to_numpy(dtype=float)
        if not np.isfinite(q).all():
            continue
        dot = np.sum(q[1:] * q[:-1], axis=1)
        if (dot < -0.5).sum() >= 2 and len(s) < 90:
            break
    fixed = quat.make_continuous(q / np.linalg.norm(q, axis=1, keepdims=True))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    names = ['w', 'x', 'y', 'z']
    colors = [BLUE, ORANGE, AQUA, '#4a3aa7']
    for ax, data, title in [(axes[0], q, 'Raw quaternion'), (axes[1], fixed, 'After sign continuity')]:
        for k in range(4):
            ax.plot(data[:, k], color=colors[k], lw=1.6, label=f'rot_{names[k]}')
        ax.set_title(title, loc='left')
        ax.set_xlabel('Row')
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f'Same orientation, opposite sign: fake jumps in {sid}', x=0.01, ha='left',
                 fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_sign_flip.png')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(Path('outputs/ablation/trees.json').read_text())
    ablation_steps(results)
    leave_one_out(results)
    lengths(pd.read_csv('data/processed/sequences.csv'))
    train, demo = load_train()
    train = train.sort_values(['sequence_id', 'sequence_counter'], kind='stable')
    sign_flip(train)
    subject_gravity(train, demo)
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
