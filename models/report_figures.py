"""Report figures, drawn only from the committed CSVs in results/ (no model files needed).

  python -m models.report_figures            # -> results/<section>/figures/*.png (200 dpi)

Colours follow one validated palette: networks blue, summary-feature (tree) models orange, hybrids aqua; marker
shape repeats the family so identity never depends on colour alone. Every axis is labelled with the metric and
the score being shown ("combined" = the competition metric over all-sensor and motion-only inputs pooled)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from models.paths import ABLATIONS, COMPARISON, ERRORS, HYBRIDS, TUNING, VALIDATION, figures  # noqa: E402

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
FAMILY = {'net': (BLUE, 'o', 'Sequence network'), 'tree': (ORANGE, 's', 'Summary-feature model'),
          'hyb': (AQUA, 'D', 'Hybrid (network + LightGBM)')}
NAMES = {
    'n1_mirror': ('N1: tuned CNN-BiGRU, left-handers mirrored', 'net'),
    'h1_cnn_gru_tuned_lgbm_tuned': ('H1: weighted blend', 'hyb'),
    'h2_cnn_gru_tuned_lgbm_tuned': ('H2: logistic stacker', 'hyb'),
    'h3_cnn_gru_tuned_lgbm_tuned': ('H3: LightGBM on network outputs', 'hyb'),
    'cnn_gru_tuned': ('CNN-BiGRU (tuned)', 'net'), 'cnn_gru': ('CNN-BiGRU', 'net'), 'gru': ('BiGRU', 'net'),
    'transformer': ('Transformer', 'net'), 'cnn': ('1D-CNN', 'net'),
    'lgbm_tuned': ('LightGBM (tuned)', 'tree'), 'lgbm': ('LightGBM', 'tree'), 'xgb': ('XGBoost', 'tree'),
    'mlp': ('MLP on summary features', 'tree'), 'rf': ('Random forest', 'tree')}
SHORT = {'Above ear - pull hair': 'Above ear: pull hair', 'Forehead - pull hairline': 'Forehead: pull hairline',
         'Forehead - scratch': 'Forehead: scratch', 'Eyebrow - pull hair': 'Eyebrow: pull hair',
         'Eyelash - pull hair': 'Eyelash: pull hair', 'Neck - pinch skin': 'Neck: pinch skin',
         'Neck - scratch': 'Neck: scratch', 'Cheek - pinch skin': 'Cheek: pinch skin', 'non_target': 'Non-BFRB gesture'}
BLUES = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.labelsize': 9, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'grid.linestyle': '-', 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 8.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white'})


def save(fig, section, name, note=None):
    if note:
        fig.text(0.01, -0.5 / fig.get_size_inches()[1], note, fontsize=7.5, color=MUTED, ha='left', va='top')
    path = figures(section) / f'{name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


def family_legend(ax, fams=('net', 'tree', 'hyb'), loc='lower right'):
    ax.legend(handles=[Line2D([], [], color=FAMILY[f][0], marker=FAMILY[f][1], ls='', ms=7, label=FAMILY[f][2])
                       for f in fams], loc=loc)


SECTION_OF = {'main': COMPARISON, 'tuning': TUNING, 'ablations': ABLATIONS, 'n1': ABLATIONS, 'routing': ABLATIONS,
              'hybrids': HYBRIDS, 'ensembles_exploratory': HYBRIDS, 'sanity': VALIDATION, 'arch_default': COMPARISON,
              'tuning_nn': TUNING, 'tuning_gb': TUNING}


def table(tag):
    return pd.read_csv(SECTION_OF[tag] / f'table_{tag}.csv').set_index('model')


def compare(tag):
    return pd.read_csv(SECTION_OF[tag] / f'compare_{tag}.csv').set_index('comparison')


def analysis(model, name):
    return pd.read_csv(ERRORS / model / name, index_col=0 if name.startswith(('per_class', 'confusion')) else None)


def fig_model_comparison():
    t = pd.concat([table('main'), table('hybrids'), table('n1')])
    t = t[~t.index.duplicated()].loc[[m for m in NAMES if m in t.index]].sort_values('combined')
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (m, r) in enumerate(t.iterrows()):
        c, mk, _ = FAMILY[NAMES[m][1]]
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, solid_capstyle='round', alpha=0.45)
        ax.plot(r.combined, i, marker=mk, color=c, ms=8 if m != 'n1_mirror' else 10, mec='white', mew=1.5)
        ax.text(r.hi + 0.002, i, f'{r.combined:.3f}', va='center', fontsize=8,
                color=INK if m == 'n1_mirror' else INK2, fontweight='bold' if m == 'n1_mirror' else 'normal')
    ax.set_yticks(range(len(t)), [NAMES[m][0] for m in t.index])
    ax.get_yticklabels()[-1].set_fontweight('bold')
    ax.set_xlabel('Competition score, combined (½ binary F1 + ½ macro F1), 5-fold CV, mean of 3 seeds')
    ax.set_xlim(t.lo.min() - 0.01, t.hi.max() + 0.02)
    ax.grid(axis='y', visible=False)
    ax.set_title('Sequence networks beat summary-feature models; mirroring left-handers (N1) is best')
    family_legend(ax)
    save(fig, COMPARISON, 'model_comparison', 'Bars: 95% confidence interval from resampling the 81 people (2,000 resamples).')


def fig_sensor_settings():
    t = pd.concat([table('main'), table('hybrids'), table('n1')])
    t = t[~t.index.duplicated()].loc[[m for m in NAMES if m in t.index]].sort_values('combined')
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (m, r) in enumerate(t.iterrows()):
        ax.plot([r.imu, r.full], [i, i], color=GRID, lw=3, solid_capstyle='round', zorder=1)
        ax.plot(r.full, i, 'o', color=BLUE, ms=8, mec='white', mew=1.5, zorder=3)
        ax.plot(r.imu, i, 'o', color=ORANGE, ms=8, mec='white', mew=1.5, zorder=3)
    ax.set_yticks(range(len(t)), [NAMES[m][0] for m in t.index])
    ax.set_xlabel('Competition score on each half of the validation data (5-fold CV, mean of 3 seeds)')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='', ms=7, label='All sensors (IMU + thermopile + ToF)'),
                       Line2D([], [], color=ORANGE, marker='o', ls='', ms=7, label='Motion only (IMU), as in half the test set')],
              loc='upper center', bbox_to_anchor=(0.4, -0.12), ncol=2)
    gap = (t.full - t.imu)
    ax.set_title(f'Losing thermopile + ToF costs every model {gap.min():.2f}-{gap.max():.2f}')
    save(fig, COMPARISON, 'all_sensors_vs_motion_only')


def fig_effects():
    def row(tag, comp, label, kind):
        c = compare(tag).loc[comp]
        return dict(label=label, kind=kind, **c[['delta', 'lo', 'hi', 'p_holm', 'folds_won']].to_dict(),
                    sig=bool(c['significant']))
    rows = [row('n1', 'n1_mirror - cnn_gru_tuned', 'N1: mirror left-handed wearers', 'novelty'),
            row('ablations', 'cnn_gru - abl_noaug', 'Physics-aware augmentation', 'component'),
            row('tuning_nn', 'cnn_gru_tuned - cnn_gru', 'Nested tuning: CNN-BiGRU', 'component'),
            row('routing', 'cnn_gru_tuned - cnn_gru_tuned:blank', 'N3: dedicated motion-only model', 'novelty'),
            row('ablations', 'cnn_gru - abl_notofreg', 'ToF 3x3 region features', 'component'),
            row('hybrids', 'h1_cnn_gru_tuned_lgbm_tuned - cnn_gru_tuned', 'H1: blend with LightGBM', 'component'),
            row('ablations', 'cnn_gru - abl_nophase', 'N2: phase auxiliary loss', 'novelty'),
            row('tuning_gb', 'lgbm_tuned - lgbm', 'Nested tuning: LightGBM', 'component')]
    if (COMPARISON / 'compare_arch_default.csv').exists():
        rows.append(row('arch_default', 'gru - cnn_gru', 'Architecture: BiGRU vs CNN-BiGRU (both default)', 'component'))
    d = pd.DataFrame(rows)[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(0, color=INK2, lw=1)
    for i, r in d.iterrows():
        c = BLUE if r.kind == 'novelty' else INK2
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, solid_capstyle='round', alpha=0.5)
        ax.plot(r.delta, i, 'o', ms=8, color=c if r.sig else 'white', mec=c, mew=1.8)
        p = '<0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}'
        ax.text(max(r.hi, 0) + 0.0012, i, f'{r.delta:+.4f}   p(Holm) {p}   wins {r.folds_won} folds',
                va='center', fontsize=7.6, color=INK2)
    ax.set_yticks(range(len(d)), d.label)
    ax.set_xlim(d.lo.min() - 0.002, d.hi.max() + 0.017)
    ax.set_xlabel('Change in combined competition score (paired, 95% CI from resampling people)')
    ax.grid(axis='y', visible=False)
    ax.set_title('What each design choice is worth: N1 and augmentation matter most, N2 does nothing')
    fig.text(0.01, -0.12, 'Filled = significant after Holm correction within its family of comparisons. 3 seeds per model.',
             fontsize=7.5, color=MUTED, ha='left', va='top')
    ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='', ms=7, label='Novelty claim'),
                       Line2D([], [], color=INK2, marker='o', ls='', ms=7, label='Component / method'),
                       Line2D([], [], color=INK2, marker='o', mfc='white', ls='', ms=7, label='Not significant (Holm)')],
              loc='upper center', bbox_to_anchor=(0.45, -0.13), ncol=3)
    save(fig, ABLATIONS, 'component_effects')


def fig_per_class():
    runs = [('lgbm_tuned', 'LightGBM (tuned)', ORANGE), ('cnn_gru_tuned', 'CNN-BiGRU (tuned)', BLUE),
            ('n1_mirror', 'N1 (mirrored)', AQUA)]
    pc = {m: analysis(m, 'per_class_f1.csv') for m, _, _ in runs}
    classes = list(pc['n1_mirror'].index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    h = 0.26
    for ax, setting, title in ((axes[0], 'full', 'All sensors'), (axes[1], 'imu', 'Motion only')):
        for j, (m, lab, c) in enumerate(runs):
            y = np.arange(len(classes)) + (j - 1) * h
            ax.barh(y, pc[m].loc[classes, setting], height=h * 0.86, color=c, label=lab)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_xlabel('F1 score per class (5-fold CV)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(classes)), [SHORT[c] for c in classes])
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs[::-1], ls[::-1], loc='lower center', ncol=3, bbox_to_anchor=(0.55, -0.06))
    fig.suptitle('Which BFRB is it? Networks win every class; eyebrow vs eyelash pulling stays hardest',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.tight_layout()
    save(fig, ERRORS, 'per_class_f1')


def fig_confusion(model='n1_mirror'):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    labels = [SHORT[c] for c in analysis(model, 'confusion_full.csv').index]
    for ax, s, title in ((axes[0], 'full', 'All sensors'), (axes[1], 'imu', 'Motion only')):
        cm = analysis(model, f'confusion_{s}.csv').values.astype(float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
        for i in range(9):
            for j in range(9):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f'{cm[i, j]:.2f}'.lstrip('0') if cm[i, j] < 1 else '1', ha='center', va='center',
                            fontsize=6.3, color='white' if cm[i, j] > 0.5 else INK)
        ax.set_xticks(range(9), labels, rotation=55, ha='right', fontsize=7)
        ax.set_yticks(range(9), labels if s == 'full' else [], fontsize=7)
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel('Predicted class')
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('True class')
    for t in axes[1].get_yticklines():
        t.set_visible(False)
    fig.suptitle(f'Where the best model ({NAMES[model][0].split(":")[0]}) still errs: neighbouring sites on the face and neck',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, ERRORS, 'confusion_matrix_n1', 'Rows sum to 1 (recall per true class); cells below 0.05 left blank.')


def fig_handedness():
    a = analysis('cnn_gru_tuned', 'per_subject.csv').set_index('subject')
    b = analysis('n1_mirror', 'per_subject.csv').set_index('subject')
    d = a[['combined', 'handedness']].join(b[['combined']], rsuffix='_n1')
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for _, r in d[d.handedness == 1].iterrows():
        ax.plot([0, 1], [r.combined, r.combined_n1], color=GRID, lw=1, zorder=1)
    for _, r in d[d.handedness == 0].iterrows():
        ax.plot([0, 1], [r.combined, r.combined_n1], color=BLUE, lw=1.6, alpha=0.8, zorder=2)
        ax.plot([0, 1], [r.combined, r.combined_n1], 'o', color=BLUE, ms=5, mec='white', mew=1, zorder=3)
    for hand, c, name in ((0, BLUE, 'left-handed'), (1, INK2, 'right-handed')):
        m = d[d.handedness == hand][['combined', 'combined_n1']].mean()
        ax.plot([-0.08, 1.08], [m.combined, m.combined_n1], ls='', marker='_', ms=0)
        ax.text(1.06, m.combined_n1, f'{name} mean {m.combined_n1:.3f}', va='center', fontsize=8, color=c)
        ax.text(-0.06, m.combined, f'{m.combined:.3f}', va='center', ha='right', fontsize=8, color=c)
    ax.set_xticks([0, 1], ['CNN-BiGRU (tuned)', 'N1: same model,\nleft-handers mirrored'])
    ax.set_xlim(-0.35, 1.6)
    ax.set_ylabel('Combined competition score per person')
    ax.grid(axis='x', visible=False)
    gap0 = d[d.handedness == 1].combined.mean() - d[d.handedness == 0].combined.mean()
    gap1 = d[d.handedness == 1].combined_n1.mean() - d[d.handedness == 0].combined_n1.mean()
    ax.set_title(f'N1 closes the left-hander gap ({gap0:.3f} -> {gap1:.3f})')
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, marker='o', ms=5, label='Left-handed person (10)'),
                       Line2D([], [], color=GRID, lw=1.5, label='Right-handed person (71)')],
              loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=2)
    save(fig, ABLATIONS, 'n1_handedness')


def fig_seed_averaging():
    s = pd.read_csv(COMPARISON / 'seed_averaging.csv')
    s = s[s.model.isin(NAMES)].sort_values('averaged').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, r in s.iterrows():
        c, mk, _ = FAMILY[NAMES[r.model][1]]
        ax.plot([r.single_min, r.single_max], [i, i], color=c, lw=5, alpha=0.18, solid_capstyle='round')
        ax.annotate('', xy=(r.averaged, i), xytext=(r.single_mean, i),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.4, shrinkA=4, shrinkB=4))
        ax.plot(r.single_mean, i, marker=mk, color='white', mec=c, mew=1.6, ms=7)
        ax.plot(r.averaged, i, marker=mk, color=c, mec='white', mew=1.2, ms=8)
        ax.text(max(r.averaged, r.single_max) + 0.002, i, f'+{r.gain:.3f}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(s)), [NAMES[m][0] for m in s.model])
    ax.set_xlabel('Combined competition score (5-fold CV)')
    ax.grid(axis='y', visible=False)
    ax.set_title('Averaging 3 seeds adds +0.012-0.020 to every neural network (incl. the MLP), ~0 to trees')
    ax.legend(handles=[Line2D([], [], color=INK2, marker='o', mfc='white', ls='', ms=7, label='Single seed (mean of 3)'),
                       Line2D([], [], color=INK2, marker='o', ls='', ms=7, label='Average of 3 seeds'),
                       Line2D([], [], color=INK2, lw=5, alpha=0.2, label='Range of single seeds')], loc='lower right')
    save(fig, COMPARISON, 'seed_averaging')


def fig_metric_breakdown():
    b = pd.read_csv(COMPARISON / 'metric_breakdown.csv')
    b = b[b.model.isin(NAMES)].sort_values('full_macro_f1').reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1.6], 'wspace': 0.12})
    for ax, part, title, xl in ((axes[0], 'binary', 'Is it a BFRB?  binary F1', (0.93, 1.0)),
                                (axes[1], 'macro', 'Which BFRB?  macro F1, 9 classes', (0.42, 0.78))):
        for i, r in b.iterrows():
            c, mk, _ = FAMILY[NAMES[r.model][1]]
            ax.plot([r[f'imu_{part}_f1'], r[f'full_{part}_f1']], [i, i], color=GRID, lw=3, zorder=1)
            ax.plot(r[f'full_{part}_f1'], i, marker=mk, color=c, ms=7.5, mec='white', mew=1.2, zorder=3)
            ax.plot(r[f'imu_{part}_f1'], i, marker=mk, color='white', mec=c, mew=1.5, ms=7, zorder=3)
        ax.set_xlim(*xl)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('F1 (5-fold CV)')
    axes[0].set_yticks(range(len(b)), [NAMES[m][0] for m in b.model])
    axes[1].legend(handles=[Line2D([], [], color=INK2, marker='o', ls='', ms=7, label='All sensors'),
                            Line2D([], [], color=INK2, marker='o', mfc='white', ls='', ms=7, label='Motion only')],
                   loc='upper center', bbox_to_anchor=(0.2, -0.14), ncol=2)
    fig.suptitle('Detecting a BFRB is nearly solved; telling BFRBs apart is where models differ',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    save(fig, COMPARISON, 'binary_vs_macro_f1', 'Competition score = ½ binary F1 + ½ macro F1. Colour/shape = model family.')


def fig_posture():
    runs = [('cnn_gru_tuned', 'CNN-BiGRU (tuned)', BLUE), ('n1_mirror', 'N1 (mirrored)', AQUA)]
    o = {m: analysis(m, 'by_orientation.csv').set_index('orientation') for m, _, _ in runs}
    order = o['n1_mirror'].sort_values('full').index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, s, title in ((axes[0], 'full', 'All sensors'), (axes[1], 'imu', 'Motion only')):
        for j, (m, lab, c) in enumerate(runs):
            ax.barh(np.arange(len(order)) + (j - 0.5) * 0.36, o[m].loc[order, s], height=0.31, color=c, label=lab)
        ax.set_xlim(0.7, 0.92)
        ax.set_title(title)
        ax.set_xlabel('Competition score (5-fold CV)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(order)), [f'{p} (n={o["n1_mirror"].loc[p, "n"]})' for p in order], fontsize=7.8)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=2, bbox_to_anchor=(0.6, -0.08))
    fig.suptitle('Lying on the side is hardest; N1 helps in every posture', x=0.01, ha='left',
                 fontsize=10.5, fontweight='bold')
    fig.tight_layout()
    save(fig, ERRORS, 'posture')


def fig_ensembles():
    if not (HYBRIDS / 'table_ensembles_exploratory.csv').exists():
        return
    t = table('ensembles_exploratory').sort_values('combined')
    lab = {'n1_mirror': 'N1 alone', 'ens_networks': 'CNN-BiGRU (tuned) + BiGRU + Transformer + CNN',
           'ens_n1_cnn_gru_tuned': 'N1 + CNN-BiGRU (tuned)', 'ens_n1_gru_transformer': 'N1 + BiGRU + Transformer'}
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    base = t.loc['n1_mirror', 'combined']
    ax.axvline(base, color=GRID, lw=1.2)
    for i, (m, r) in enumerate(t.iterrows()):
        c = BLUE if m != 'n1_mirror' else INK2
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, alpha=0.45, solid_capstyle='round')
        ax.plot(r.combined, i, 'o', color=c, ms=8, mec='white', mew=1.5)
        ax.text(r.hi + 0.002, i, f'{r.combined:.3f} ({r.combined - base:+.3f})', va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(t)), [lab.get(m, m) for m in t.index])
    ax.set_xlim(t.lo.min() - 0.005, t.hi.max() + 0.025)
    ax.set_xlabel('Combined competition score (5-fold CV), equal-weight averages, nothing fitted')
    ax.grid(axis='y', visible=False)
    ax.set_title('Mixing architectures adds more than mixing views (exploratory)')
    save(fig, HYBRIDS, 'ensembles_exploratory', 'Exploratory: ensembles chosen after seeing CV, so treat the ranking as indicative.')


def fig_tuning():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, m, title in ((axes[0], 'cnn_gru', 'CNN-BiGRU (12 trials, GPU)'), (axes[1], 'lgbm', 'LightGBM (20 trials, CPU)')):
        t = pd.read_csv(TUNING / f'params/{m}_trials.csv')
        done, pruned = t[t.state == 'COMPLETE'], t[t.state == 'PRUNED']
        ax.plot(done.number, done.value.cummax(), color=INK2, lw=1.5, drawstyle='steps-post', label='Best so far')
        ax.plot(done.number, done.value, 'o', color=BLUE if m == 'cnn_gru' else ORANGE, ms=6, mec='white',
                label='Completed trial')
        ax.plot(pruned.number, [ax.get_ylim()[0]] * len(pruned), 'x', color=MUTED, ms=5, label='Stopped early')
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Inner-fold combined score')
    axes[1].legend(loc='lower right', fontsize=7.8)
    best = {m: json.load(open(TUNING / f'params/{m}.json'))['inner_combined_score'] for m in ('cnn_gru', 'lgbm')}
    fig.suptitle(f'Nested tuning on outer folds 1-4 (best inner score: network {best["cnn_gru"]:.3f}, '
                 f'LightGBM {best["lgbm"]:.3f})', x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.tight_layout()
    save(fig, TUNING, 'tuning_history', 'Inner folds are smaller than outer folds, so inner scores sit below the reported CV.')


def dot_ci(ax, rows, labels, colors, ref=None, fmt='{:.3f}', bold=None):
    """Horizontal dot + 95% CI rows (rows: DataFrame with combined/lo/hi), value printed right of the CI."""
    if ref is not None:
        ax.axvline(ref, color=GRID, lw=1.2, zorder=0)
    for i, ((m, r), c) in enumerate(zip(rows.iterrows(), colors)):
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2, alpha=0.45, solid_capstyle='round')
        ax.plot(r.combined, i, 'o', color=c, ms=8, mec='white', mew=1.5)
        ax.text(r.hi + 0.0015, i, fmt.format(r.combined), va='center', fontsize=8,
                color=INK if m == bold else INK2, fontweight='bold' if m == bold else 'normal')
    ax.set_yticks(range(len(rows)), labels)
    ax.grid(axis='y', visible=False)


def fig_architectures():
    t = table('main').loc[['cnn', 'transformer', 'cnn_gru', 'gru']]
    c = compare('arch_default')
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    dot_ci(ax, t, [NAMES[m][0] + ' (default settings)' for m in t.index], [BLUE] * 4)
    ax.set_xlim(t.lo.min() - 0.004, t.hi.max() + 0.012)
    ax.set_xlabel('Combined competition score (5-fold CV, mean of 3 seeds), 95% CI')
    ax.set_title('At equal (default) settings the four architectures are statistically tied')
    pmin = c.p_holm.min()
    save(fig, COMPARISON, 'architecture_comparison',
         f'BiGRU vs each other core, paired bootstrap with Holm correction: smallest adjusted p = {pmin:.2f} '
         '(none significant).')


def fig_tuning_search_space():
    spaces = {'cnn_gru': [('lr', 'learning rate', True), ('d', 'width (channels)', False),
                          ('drop', 'dropout', False), ('epochs', 'epochs', False), ('phase', 'phase-loss weight', False)],
              'lgbm': [('num_leaves', 'leaves', True), ('learning_rate', 'learning rate', False),
                       ('n_estimators', 'trees', False), ('colsample_bytree', 'feature fraction', False),
                       ('min_child_samples', 'min child samples', True), ('reg_lambda', 'L2 penalty', True)]}
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 4.2))
    for row, (m, params) in enumerate(spaces.items()):
        t = pd.read_csv(TUNING / f'params/{m}_trials.csv')
        done, pruned = t[t.state == 'COMPLETE'], t[t.state == 'PRUNED']
        best = done.loc[done.value.idxmax()]
        c = BLUE if m == 'cnn_gru' else ORANGE
        for j in range(6):
            ax = axes[row, j]
            if j >= len(params):
                ax.axis('off')
                continue
            p, lab, log = params[j]
            col = f'params_{p}'
            ax.plot(pruned[col], pruned.value, 'o', mfc='white', mec=MUTED, ms=5, mew=1.1)
            ax.plot(done[col], done.value, 'o', color=c, ms=6, mec='white', mew=1)
            ax.plot(best[col], best.value, '*', color=INK, ms=11, mec='white', mew=0.8)
            if log:
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3) if p in ('lr', 'num_leaves', 'min_child_samples') else (1,), numticks=5))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(lab, fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(('CNN-BiGRU' if m == 'cnn_gru' else 'LightGBM') + '\ninner-fold score', fontsize=8)
            else:
                ax.set_yticklabels([])
        lo, hi = t.value.min(), t.value.max()
        for ax in axes[row, :len(params)]:
            ax.set_ylim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.15)
    fig.legend(handles=[Line2D([], [], color=INK2, marker='o', ls='', ms=6, label='Completed trial'),
                        Line2D([], [], color=MUTED, marker='o', mfc='white', ls='', ms=6,
                               label='Stopped early (partial score, first inner fold)'),
                        Line2D([], [], color=INK, marker='*', ls='', ms=10, label='Selected setting')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Hyper-parameter search: each trial\'s inner-fold score against each setting (Optuna TPE)',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, TUNING, 'search_space')


def fig_default_vs_tuned():
    t = table('tuning')
    pairs = [('cnn_gru', 'cnn_gru_tuned', 'CNN-BiGRU', BLUE, 'tuning_nn'), ('lgbm', 'lgbm_tuned', 'LightGBM', ORANGE, 'tuning_gb')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, col, title in ((axes[0], 'combined', 'All 5 folds (combined score)'),
                           (axes[1], 'fold0', 'Fold 0 only: never seen by tuning')):
        for i, (d, tu, lab, c, tag) in enumerate(pairs):
            a, b = t.loc[d, col], t.loc[tu, col]
            ax.annotate('', xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle='-|>', color=c, lw=1.5,
                                                                    shrinkA=5, shrinkB=5))
            ax.plot(a, i, 'o', color='white', mec=c, mew=1.6, ms=8)
            ax.plot(b, i, 'o', color=c, mec='white', mew=1.2, ms=9)
            txt = f'{b - a:+.4f}'
            if col == 'combined':
                r = compare(tag).iloc[0]
                txt += f'  (p Holm {r.p_holm:.3f}, {r.folds_won})' if r.p_holm >= 0.001 else f'  (p Holm <0.001, {r.folds_won})'
            ax.text(max(a, b) + 0.007, i, txt, va='center', fontsize=7.8, color=INK2)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(0.76, 0.905)
        ax.set_xlabel('Competition score (5-fold CV)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks([0, 1], ['CNN-BiGRU', 'LightGBM'])
    axes[0].set_ylim(-0.6, 1.6)
    fig.legend(handles=[Line2D([], [], color=INK2, marker='o', mfc='white', ls='', ms=7, label='Default settings'),
                        Line2D([], [], color=INK2, marker='o', ls='', ms=7, label='Tuned (nested Optuna)')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle('Tuning helps the network (also on the untouched fold), not LightGBM', x=0.01, ha='left',
                 fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, TUNING, 'default_vs_tuned')


def fig_ablation_scores():
    t = table('ablations').loc[['cnn_gru', 'abl_nophase', 'abl_notofreg', 'abl_noaug']]
    lab = {'cnn_gru': 'Full model (CNN-BiGRU, default)', 'abl_noaug': 'without augmentation',
           'abl_nophase': 'without phase auxiliary loss (N2)', 'abl_notofreg': 'without ToF 3x3 region features'}
    c = compare('ablations')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw={'width_ratios': [1.25, 1]})
    dot_ci(axes[0], t, [lab[m] for m in t.index], [INK2] + [BLUE] * 3, ref=t.loc['cnn_gru', 'combined'],
           bold='cnn_gru')
    axes[0].set_xlim(t.lo.min() - 0.003, t.hi.max() + 0.012)
    axes[0].set_xlabel('Combined score, 95% CI')
    for i, m in enumerate(t.index):
        r = t.loc[m]
        axes[1].plot([r.imu, r.full], [i, i], color=GRID, lw=3, zorder=1)
        axes[1].plot(r.full, i, 'o', color=BLUE, ms=7, mec='white', mew=1.2, zorder=3)
        axes[1].plot(r.imu, i, 'o', color=ORANGE, ms=7, mec='white', mew=1.2, zorder=3)
        if m != 'cnn_gru':
            p = c.loc[f'cnn_gru - {m}', 'p_holm']
            axes[1].text(0.868, i, 'n.s.' if p >= 0.05 else ('p<0.001' if p < 0.001 else f'p={p:.3f}'),
                         va='center', fontsize=7.6, color=INK2)
    axes[1].set_xlim(0.765, 0.885)
    axes[1].set_xlabel('All sensors (blue) / motion only (orange)')
    axes[1].grid(axis='y', visible=False)
    fig.suptitle('Ablations: removing augmentation hurts most; the phase loss (N2) makes no difference',
                 x=0.01, ha='left', fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, ABLATIONS, 'ablation_scores', 'p = Holm-corrected paired bootstrap vs the full model, combined score.')


def fig_hybrids():
    t = table('hybrids').loc[['lgbm_tuned', 'cnn_gru_tuned', 'h3_cnn_gru_tuned_lgbm_tuned',
                              'h2_cnn_gru_tuned_lgbm_tuned', 'h1_cnn_gru_tuned_lgbm_tuned']]
    w = json.load(open(HYBRIDS / 'blend_weights.json'))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    dot_ci(ax, t, [NAMES[m][0] for m in t.index], [FAMILY[NAMES[m][1]][0] for m in t.index],
           ref=t.loc['cnn_gru_tuned', 'combined'], bold='h1_cnn_gru_tuned_lgbm_tuned')
    ax.set_xlim(t.lo.min() - 0.005, t.hi.max() + 0.015)
    ax.set_xlabel('Combined competition score (5-fold CV), 95% CI')
    ax.set_title('Combining the network with LightGBM: the simple blend (H1) is best, by a small margin')
    ax.legend(handles=[Line2D([], [], color=FAMILY[f][0], marker='o', ls='', ms=7, label=FAMILY[f][2])
                       for f in ('net', 'tree', 'hyb')], loc='center', bbox_to_anchor=(0.36, 0.62))
    save(fig, HYBRIDS, 'hybrids', f'H1 weight on the network (chosen on other folds): {w["full"]:.2f} with all sensors, '
         f'{w["imu"]:.2f} motion only. All hybrids are fitted on out-of-fold predictions only.')


def fig_sanity():
    real = table('main').loc[['cnn_gru_tuned', 'lgbm_tuned']]
    sh = table('sanity')
    t = pd.concat([sh.loc[['sanity_shuffled_lgbm', 'sanity_shuffled_cnn_gru']], real.iloc[::-1]])
    lab = ['LightGBM, labels shuffled', 'CNN-BiGRU, labels shuffled', 'LightGBM (tuned), real labels',
           'CNN-BiGRU (tuned), real labels']
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.axvspan(0.37, 0.40, color=GRID, alpha=0.8, lw=0)
    ax.text(0.385, 3.55, 'chance level\nfor this metric', ha='center', va='top', fontsize=7.6, color=INK2)
    dot_ci(ax, t, lab, [MUTED, MUTED, ORANGE, BLUE])
    ax.set_xlim(0.3, 0.88)
    ax.set_ylim(-0.6, 3.7)
    ax.set_xlabel('Combined competition score (5-fold CV)')
    ax.set_title('Leakage check: with shuffled labels both models fall to chance')
    save(fig, VALIDATION, 'sanity_check', 'Chance: random guessing at the class frequencies scores about 0.37; always '
         'predicting a BFRB about 0.39.')


if __name__ == '__main__':
    for f in (fig_model_comparison, fig_sensor_settings, fig_effects, fig_per_class, fig_confusion, fig_handedness,
              fig_seed_averaging, fig_metric_breakdown, fig_posture, fig_ensembles, fig_tuning, fig_architectures,
              fig_tuning_search_space, fig_default_vs_tuned, fig_ablation_scores, fig_hybrids, fig_sanity):
        try:
            f()
        except (FileNotFoundError, KeyError) as e:      # partial runs: draw what exists
            print(f'skipped {f.__name__}: missing {e}')
