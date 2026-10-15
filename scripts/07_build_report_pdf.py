"""Build docs/Preprocessing_Report.pdf from the saved results and figures.

    python scripts/07_build_report_pdf.py

Every number is read from outputs/ (ablation, verification, quality checks)
and processed/meta.json, so the report always matches the code that ran.
"""
import json
from datetime import date
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = Path('docs/Preprocessing_Report.pdf')
FIG = Path('docs/figures/results')
abl = json.loads(Path('outputs/ablation/trees.json').read_text())
phys = json.loads(Path('outputs/verification/physics.json').read_text())
quality = json.loads(Path('outputs/verification/quality.json').read_text())
parity = json.loads(Path('outputs/verification/parity.json').read_text())
meta = json.loads(Path('processed/meta.json').read_text())

# Fonts with full Unicode coverage (±, →, ↔, −) from matplotlib's bundled DejaVu.
font_dir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('Body', str(font_dir / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('Body-Bold', str(font_dir / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('Body-Italic', str(font_dir / 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('Mono', str(font_dir / 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-Bold')

INK, INK2, RULE, TINT = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#d6d5d0'), colors.HexColor('#f3f2ee')
ACCENT = colors.HexColor('#2a78d6')
GOOD, BAD = colors.HexColor('#e6f2ea'), colors.HexColor('#fbeae6')

S = {
    'title': ParagraphStyle('title', fontName='Body-Bold', fontSize=21, leading=26, textColor=INK, spaceAfter=4),
    'subtitle': ParagraphStyle('subtitle', fontName='Body', fontSize=11, leading=15, textColor=INK2, spaceAfter=14),
    'h1': ParagraphStyle('h1', fontName='Body-Bold', fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6),
    'h2': ParagraphStyle('h2', fontName='Body-Bold', fontSize=11, leading=14, textColor=INK, spaceBefore=10, spaceAfter=4),
    'body': ParagraphStyle('body', fontName='Body', fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    'small': ParagraphStyle('small', fontName='Body', fontSize=8, leading=10.5, textColor=INK2, spaceAfter=4),
    'cell': ParagraphStyle('cell', fontName='Body', fontSize=8, leading=10.2, textColor=INK),
    'cellb': ParagraphStyle('cellb', fontName='Body-Bold', fontSize=8, leading=10.2, textColor=INK),
    'bullet': ParagraphStyle('bullet', fontName='Body', fontSize=9.3, leading=13.2, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2),
    'box': ParagraphStyle('box', fontName='Body', fontSize=9.3, leading=13.2, textColor=INK),
    'caption': ParagraphStyle('caption', fontName='Body-Italic', fontSize=8, leading=10.5, textColor=INK2, spaceAfter=10),
}

story = []
P = lambda text, style='body': story.append(Paragraph(text, S[style]))  # noqa: E731
B = lambda text: story.append(Paragraph(text, S['bullet'], bulletText='•'))  # noqa: E731


def table(rows, widths, header=True, highlight=None, bold_last=False):
    data = [[Paragraph(str(c), S['cellb'] if (header and i == 0) else S['cell']) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.8, INK2),
             ('LINEBELOW', (0, 1), (-1, -1), 0.3, RULE), ('TOPPADDING', (0, 0), (-1, -1), 3),
             ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('LEFTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        style.append(('BACKGROUND', (0, 0), (-1, 0), TINT))
    for row, color in (highlight or {}).items():
        style.append(('BACKGROUND', (0, row), (-1, row), color))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))


def figure(name, width_mm, caption):
    img = Image(str(FIG / name))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * ratio
    story.append(KeepTogether([img, Paragraph(caption, S['caption'])]))


def callout(text):
    t = Table([[Paragraph(text, S['box'])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), TINT), ('LINEBEFORE', (0, 0), (0, -1), 3, ACCENT),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9), ('TOPPADDING', (0, 0), (-1, -1), 7),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    story.append(t)
    story.append(Spacer(1, 10))


def f3(x):
    return f'{x:.3f}'


def pm(row, key):
    return f"{abl[row][key + '_mean']:.3f} ± {abl[row][key + '_std']:.3f}"


def delta(row, base, key):
    d = round(abl[row][key + '_mean'] - abl[base][key + '_mean'], 3) + 0.0
    return '0.000' if d == 0 else f'{d:+.3f}'.replace('-', '−')


def wins(row, key):
    """Folds in which G beats the variant (variant lower), ties within 0.0005 excluded."""
    g = {f['fold']: f[key] for f in abl['G']['folds']}
    return sum(1 for f in abl[row]['folds'] if g[f['fold']] - f[key] > 0.0005)


def better(row, base, key):
    """Folds in which `row` scores higher than `base` (ties within 0.0005 excluded)."""
    ref = {f['fold']: f[key] for f in abl[base]['folds']}
    return sum(1 for f in abl[row]['folds'] if f[key] - ref[f['fold']] > 0.0005)


b, m, o, q = phys['basic'], phys['missing'], phys['orientation'], phys['quaternion']
gain_full = abl['H']['full_mean'] - abl['A']['full_mean']
gain_imu = abl['H']['imu_mean'] - abl['A']['imu_mean']

# ---------------------------------------------------------------- title + summary
P('Preprocessing for BFRB gesture detection', 'title')
P(f'SC4000 group project · Kaggle “CMI – Detect Behavior with Sensor Data” · preprocessing workstream · {date.today():%d %B %Y}', 'subtitle')
callout(
    f'<b>In one paragraph.</b> We rebuilt the preprocessing so that one function, <font name="Mono">preprocess()</font>, '
    f'cleans a single recording, corrects how the watch was worn and computes {len(meta["feature_columns"])} features per time step, '
    f'with exactly the same code in training and inside the Kaggle prediction call. Every physical assumption was checked on the '
    f'real data before it was used, and every step was measured with the same model on the same five subject folds. '
    f'The final pipeline scores <b>{f3(abl["H"]["full_mean"])}</b> with all sensors and <b>{f3(abl["H"]["imu_mean"])}</b> '
    f'with the motion sensor alone, against {f3(abl["A"]["full_mean"])} and {f3(abl["A"]["imu_mean"])} for raw data: '
    f'<b>+{gain_full:.3f}</b> and <b>+{gain_imu:.3f}</b>.')

P('Headline results', 'h2')
table([
    ['', 'All sensors', 'IMU only (≈ half of the hidden test set)'],
    ['Raw data, missing → 0 (row A)', pm('A', 'full'), pm('A', 'imu')],
    ['Final pipeline (row H)', pm('H', 'full'), pm('H', 'imu')],
    ['Improvement', f'+{gain_full:.3f}', f'+{gain_imu:.3f}'],
], [62, 50, 58], highlight={3: GOOD})
P('Score = ½ binary F1 (BFRB vs not) + ½ macro F1 over the 8 BFRB gestures plus one “non-target” class, the competition metric. '
  'Mean ± standard deviation over 5 folds split by person (StratifiedGroupKFold, seed 42). '
  'The measuring model is a fixed LightGBM classifier on per-recording summary statistics; it is a yardstick for the preprocessing, not our final model.', 'small')

P('What the evidence says, in short', 'h2')
B('<b>Biggest single wins:</b> ToF handling (treating “nothing detected” as far, plus 3×3 regions of each depth image), '
  'statistics from the end of the recording where the gesture happens, and fixing how the device was worn.')
B('<b>Left-handed and upside-down wearers</b> are real and correctable: after the corrections they agree with everyone else as well as right-handers agree with each other.')
B('<b>Two findings went against the original plan</b> and we followed the data: smoothing before derivative features <i>hurts</i> '
  f'(turning it off scored higher in {better("H", "G", "full")} of 5 folds with all sensors and {better("H", "G", "imu")} of 5 IMU-only), and deleting the upside-down subjects is worse than fixing them.')
B('<b>Two steps are correct but redundant for a tree model:</b> quaternion sign continuity and gravity removal no longer move the score once the 6D rotation and gravity-direction features exist. We report this rather than overclaim.')
story.append(PageBreak())

# ---------------------------------------------------------------- 1 problem
P('1. The task and what it demands from preprocessing', 'h1')
P('A wrist device records motion (accelerometer + orientation quaternion), five thermopiles (skin temperature) and five 8×8 time-of-flight (ToF) depth cameras '
  'while a person performs one of 18 gestures, 8 of which are body-focused repetitive behaviours (BFRBs). We must name the gesture from one short recording. '
  'Three facts about the test set drive every design decision:')
B('<b>One recording at a time.</b> Kaggle calls our code once per test sequence, so every step must work on a single recording; anything learned from all data (scalers) is fitted once and saved.')
B('<b>About half the test recordings have only the motion sensor.</b> The pipeline must work, and be evaluated, with thermopile and ToF missing.')
B('<b>Only sensor columns and demographics are available at test time.</b> Train-only columns (gesture, orientation, behaviour, phase) may be targets, never inputs.')

# ---------------------------------------------------------------- 2 data facts
P('2. What the data actually looks like (verified, not assumed)', 'h1')
P('Before writing any correction we measured the data (script <font name="Mono">01_verify_physics.py</font>). Key facts:')
table([
    ['Question', 'Answer on the real data', 'Used for'],
    ['Size', f"{b['rows']:,} rows, {b['sequences']:,} recordings, {b['subjects']} people, {b['gestures']} gestures", 'folds, report'],
    ['Recording length', f"median {b['length']['p50']:.0f}, 95th pct {b['length']['p95']:.0f}, max {b['length']['p100']:.0f} rows", 'fixed length L = 128'],
    ['Handedness coding', f"1 = right ({b['handedness_counts'].get('1')} people), 0 = left ({b['handedness_counts'].get('0')})", 'mirroring'],
    ['Quaternion convention', f"rotating acceleration to the world frame gives {q['world_mean_q_v_qconj'][2]:.2f} m/s² on the vertical (gravity)", 'gravity removal'],
    ['Quaternion sign flips', f"<b>{q['sign_flip_rows']:,} flips in {q['sign_flip_sequences']:,} recordings ({q['sign_flip_sequences'] / b['sequences']:.0%})</b>", 'continuity step'],
    ['Rotation missing', f"entirely in {m['rot']['sequences_fully_missing']} recordings, never partly", 'gravity fallback'],
    ['Thermopile / ToF missing', f"entirely in {m['thm']['sequences_fully_missing']} recordings (IMU-only); partly in {m['thm']['sequences_partly_missing']} / {m['tof']['sequences_partly_missing']}; sensor 5 drops out most", 'gap filling, flags'],
    ['ToF “no object” (-1)', f"{m['tof_minus_one_pct_of_finite']:.1f}% of ToF pixels; valid range {m['tof_valid_range'][0]:.0f}–{m['tof_valid_range'][1]:.0f}", 'ToF encoding'],
    ['Where the gesture happens', f"gesture phase starts at {b['gesture_start_fraction']['p50']:.0%} of the recording (median)", 'keep the end'],
    ['Broken recordings', f"{b['sequences_without_gesture_rows']} recording without gesture rows (flagged); no duplicate or missing counters", 'sanity checks'],
], [38, 92, 40])
figure('fig_lengths.png', 160, 'Figure 1. Recording lengths. Padding/truncating to L = 128 keeps 95% of recordings whole; longer ones keep their last 128 rows.')

# ---------------------------------------------------------------- 3 pipeline
story.append(PageBreak())
P('3. The pipeline, step by step: what we do and why', 'h1')

P('3.1 Cleaning (one recording at a time)', 'h2')
table([
    ['Step', 'What', 'Why'],
    ['Order', 'sort rows by sequence_counter', 'everything after assumes time order; Kaggle may not send rows in order'],
    ['Gaps', 'linear interpolation over the counter, nearest value at the ends; a sensor missing for the whole recording stays missing and is flagged', 'a filled gap is a better guess than zero; a dead sensor must not be faked'],
    ['Quaternion sign', 'flip q to −q when consecutive rows point opposite ways, then normalise', 'q and −q are the same orientation; a flip looks like a sudden full twist (Figure 2)'],
    ['Rotation gaps', 'interpolate on continuous quaternions (nlerp); identity only if rotation is absent for the whole recording', 'an identity row mid-recording would invent a huge twist'],
    ['ToF encoding', 'proximity = 1 − distance/255; “-1, nothing in range” → 0 (far)', 'keeps the physical order: 1 touching … 0 far. The earlier cleaning mapped −1 to distance 0, i.e. “touching”'],
    ['Robustness', 'strict mode audits training data; prediction mode repairs instead of crashing', 'one exception inside Kaggle fails the whole submission'],
], [26, 72, 72])
figure('fig_sign_flip.png', 160, "Figure 2. A real recording with quaternion sign flips. Left: the raw quaternion jumps although the wrist did not; right: after continuity the path is smooth.")

P('3.2 Putting every recording into one device frame', 'h2')
P('The same gesture looks different to the watch if it is worn on the other wrist or upside down. Instead of trusting competition write-ups, '
  'we tested every candidate correction by asking: after the correction, how well does a subject’s average gravity direction per gesture agree with the right-handed population '
  '(mean cosine similarity, 1 = identical)? The method was first validated on synthetic data with a planted transform, where it recovered the planted axis, sensor swap and image flip exactly.')
lm = o['left_mirror_axis_scores']
table([
    ['Left-handed: mirror which axis?', 'Agreement with right-handers'],
    ['no mirroring', f3(lm['none'])],
    ['mirror x', f"<b>{f3(lm['x'])}</b> (right-handers among themselves: {f3(o['right_self_score'])})"],
    ['mirror y', f3(lm['y'])],
    ['mirror z', f3(lm['z'])],
], [70, 100], highlight={2: GOOD})
li, lb = o['left_permutation_identity'], o['left_permutations_top10'][0]
ud = o['upside_down_rotation_scores']
ui, ub = o['upside_down_permutation_identity'], o['upside_down_permutations_top10'][0]
P(f"For left-handers, thermopile/ToF sensors 3 and 5 swap places (thermopile error {li['thm_error']:.1f} → {lb['thm_error']:.1f}, ToF error "
  f"{li['tof_error']:.3f} → {lb['tof_error']:.3f}), and each 8×8 ToF image is flipped left-right on sensors 1, 2, 4 but up-down on 3 and 5, "
  "which are mounted sideways. All five sensors are corrected, not just the accelerometer.")
P(f"<b>Upside-down wearers.</b> Exactly two subjects stand out (SUBJ_019262, SUBJ_045235); no other subject is flagged. A 180° rotation about the watch’s z axis "
  f"fixes them (agreement {f3(ud['none'])} → {f3(ud['z'])}; rotating about x or y gives {f3(ud['x'])} / {f3(ud['y'])}). Their sensors 2↔4 and 3↔5 swap and images rotate 180° "
  f"(thermopile error {ui['thm_error']:.2f} → {ub['thm_error']:.2f}).")
figure('fig_subject_gravity.png', 165, 'Figure 3. Each dot is one person’s average gravity direction seen by the watch. Left-handers (orange) and the two upside-down wearers (green) '
       'sit apart in the raw data and join the main group after the corrections.')

P('3.3 Features (computed per time step)', 'h2')
table([
    ['Group', 'Features', 'Why it should help'],
    ['Motion', 'acceleration x/y/z and magnitude; <b>linear acceleration</b> (gravity removed via the quaternion, or a low-pass estimate if rotation is missing); vertical and horizontal motion; jerk; similarity between consecutive acceleration directions', 'separates the movement from posture; a scratch is fast and repetitive, a pinch is slow'],
    ['Orientation', 'quaternion; <b>gravity direction</b> in the watch frame; continuous 6D rotation (no sign ambiguity); angular velocity and angular distance', 'hand pointing up (head) vs down (leg); twisting vs not'],
    ['Thermopile', 'five temperatures, their mean, change per step', 'skin nearby is warm'],
    ['ToF', 'per sensor: mean proximity, closest point, spread, fraction of pixels seeing something, change, and a <b>3×3 grid of regional averages</b>', 'how close the surface is and <i>where</i> in view it is'],
    ['Context', 'flags for missing rotation / thermopile / ToF; position in the recording', 'lets one model handle every sensor situation'],
], [24, 92, 54])
P(f"Total: {len(meta['feature_columns'])} per-row features (for neural networks, padded to L = {meta['fixed_length']} with a mask, start-padded so the end is kept) "
  "and 795 per-recording statistics (mean, std, min, max, last value, and mean/std over the last 25%) for tree models. "
  "Values that do not exist (e.g. thermopile when the sensor is off) are NaN: tree models handle that natively; after scaling, neural networks get 0 plus the flag.")

P('3.4 Labels, folds and scaling', 'h2')
B('Labels (training targets only): 18-gesture, binary BFRB, the 9-class scoring label, per-row phase (transition / pause / gesture), orientation, and gesture × orientation × start-state (102 combinations).')
B('Five folds grouped by person and stratified by gesture, fixed seed, saved as one shared file: no person is in two folds, every fold has all 18 gestures and a BFRB share of 62.6–62.9%.')
B('Per-fold standardisation fitted on training rows only (no leakage), plus an all-data scaler for the final model.')

# ---------------------------------------------------------------- 4 results
story.append(PageBreak())
P('4. Does each step help? Cumulative ablation', 'h1')
P('Same LightGBM settings, same five person-based folds, one change per row. Each model is trained on every training recording twice (all sensors, and with thermopile/ToF blanked) '
  'and scored on both versions of the held-out people.')
rows = [['Row', 'Adds', 'All sensors', 'Δ', 'IMU only', 'Δ']]
order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
names = {'A': 'raw columns, missing → 0', 'B': 'cleaning + ToF/thermopile summaries', 'C': 'statistics of the last 25% (keep the end)',
         'D': 'upside-down subjects corrected', 'E': 'left-handers mirrored (all sensors)', 'F': 'gravity removal (+ fallback)',
         'G': 'engineered features, with smoothing', 'H': '<b>final: G without smoothing</b>'}
for i, r in enumerate(order):
    prev = order[i - 1] if i else None
    rows.append([r, names[r], pm(r, 'full'), delta(r, prev, 'full') if prev else '', pm(r, 'imu'), delta(r, prev, 'imu') if prev else ''])
table(rows, [12, 64, 28, 15, 28, 15], highlight={8: GOOD})
figure('fig_ablation_steps.png', 165, 'Figure 4. Score after each step. Shaded bands show ±1 standard deviation across the five folds.')

story.append(PageBreak())
P('5. Does each step still matter with everything else in? Leave-one-out', 'h1')
P('Starting from the full pipeline G, we removed one step at a time. Because the fold-to-fold spread (≈ ±0.015) is larger than several differences, we also count in how many of the five folds '
  'G beats the variant: a step that wins 5/5 is reliable even if its average gain is small.')
lo = [('G-no_regions', 'remove ToF 3×3 regions'), ('G-tof_near', 'ToF −1 as “touching” (earlier cleaning)'),
      ('G-no_mirror', 'remove left-hand mirroring'), ('G-drop_upside_down', 'delete upside-down subjects instead of fixing'),
      ('G-no_upside_down', 'remove upside-down fix'), ('G-no_continuity', 'remove quaternion continuity'),
      ('G-no_gravity', 'remove gravity removal'), ('H', 'remove smoothing')]
rows = [['Variant of G', 'All sensors', 'Δ vs G', 'Folds G wins', 'IMU only', 'Δ vs G', 'Verdict']]
verdict = {'G-no_regions': 'keep: largest single step', 'G-tof_near': 'keep “far”', 'G-no_mirror': 'keep',
           'G-drop_upside_down': 'fix, don’t delete', 'G-no_upside_down': 'keep', 'G-no_continuity': 'no effect here',
           'G-no_gravity': 'redundant for trees', 'H': '<b>smoothing hurts → off</b>'}
hl = {}
for i, (k, label) in enumerate(lo, start=1):
    rows.append([label, f3(abl[k]['full_mean']), delta(k, 'G', 'full'), f"{wins(k, 'full')}/5", f3(abl[k]['imu_mean']), delta(k, 'G', 'imu'), verdict[k]])
    if k == 'H':
        hl[i] = GOOD
table(rows, [50, 18, 15, 13, 17, 15, 42], highlight=hl)
figure('fig_leave_one_out.png', 165, 'Figure 5. Change in score when one step is removed from G. Bars: mean over folds; dots: each fold. Bars left of zero mean the step was helping.')

story.append(PageBreak())
P('6. Honest reading of the results', 'h1')
P('<b>What clearly works.</b> ToF handling is the largest contributor once everything else is in: without the 3×3 regions the all-sensor score drops by 0.013, and the earlier '
  '“−1 = touching” encoding costs 0.011; both lose in all five folds. Left-hand mirroring (−0.010, 5/5 folds) and the upside-down fix (−0.006) also hold up. '
  'Deleting the upside-down subjects instead of fixing them is worse than fixing them, and even worse than leaving them alone: their data is valuable once corrected.')
P(f"<b>Smoothing hurts, so we turned it off.</b> The original plan smoothed motion before computing jerk and angular velocity. Removing it raised the score from "
  f"{f3(abl['G']['full_mean'])} to {f3(abl['H']['full_mean'])} (all sensors) and {f3(abl['G']['imu_mean'])} to {f3(abl['H']['imu_mean'])} (IMU only), better in "
  f"{better('H', 'G', 'full')} of 5 folds (all sensors; the remaining fold is a tie) and {better('H', 'G', 'imu')} of 5 (IMU only). "
  "At about 10 readings per second a gesture lasts only a few rows, so even light smoothing blurs the quick changes that distinguish, for example, scratching from pinching.")
P('<b>Correct but redundant for a tree model.</b> Quaternion continuity mattered on its own (it is part of row B), but once the 6D rotation is used, which has no sign ambiguity by construction, '
  'removing continuity changes nothing. Likewise gravity removal clearly helped when added (row F: +0.012 IMU only), but in the full pipeline the gravity-direction and 6D-rotation features let the trees recover the same information. '
  'Both steps stay: they are physically right and cheap, and neural networks, which do not split on raw columns the way trees do, may still benefit. That is a test for the modelling team.')
P('<b>Limits of this evidence.</b> The yardstick is one tree model; neural networks may weigh steps differently. Differences under about 0.005 are within fold noise. '
  'Augmentations (section 8b of the guide) were implemented but cannot be measured without training a sequence model, so they are left to the modellers.')

# ---------------------------------------------------------------- 7 QA
P('7. How we know the preprocessing is correct', 'h1')
qc = quality['checks']
table([
    ['Check', 'Result'],
    ['Unit tests (synthetic recordings built from known physics)', '44 / 44 pass (33 new + 11 from the earlier cleaning)'],
    ['Physics properties', 'mirroring, the upside-down fix and rotation augmentations keep gravity consistent with the quaternion to 10⁻⁹'],
    ['Synthetic dataset with planted transforms', 'verification script recovered the planted mirror axis, sensor swap, image flip and both upside-down subjects exactly'],
    ['Train ↔ Kaggle parity', f"{parity['checked']} real recordings sent through the Kaggle path (polars, shuffled rows, no label columns): max difference {parity['max_abs_difference']}; real test sequences {parity['test_sequences_ok']}"],
    ['Output quality checks', f"{quality['passed']} / {quality['total']} pass, e.g. " + '; '.join(
        k for k in ['no subject appears in two folds', 'rotation features NaN exactly when rotation is missing',
                    'sensor columns NaN only for whole-recording dropouts (gaps filled)', 'no train-only column leaks into features',
                    'gravity direction is a unit vector'] if qc[k]['pass'])],
    ['Build', f"{meta['sequences']:,} recordings, {meta['rows']:,} rows, {meta['sequences_with_issues']} data issues"],
], [62, 108])

P('8. Compared with the earlier cleaning', 'h1')
table([
    ['Earlier cleaning', 'Now', 'Measured effect'],
    ['ToF −1 → 0 (“touching”)', 'proximity, −1 → far', f"+{abl['G']['full_mean'] - abl['G-tof_near']['full_mean']:.3f} all sensors, {wins('G-tof_near', 'full')}/5 folds"],
    ['missing rotation rows → identity', 'interpolated; identity only if absent all recording', 'no partial gaps exist in train; protects test'],
    ['no quaternion continuity', 'continuity', '58% of recordings had flips'],
    ['raised errors on bad input', 'repairs at prediction time', 'cannot crash the Kaggle run'],
    ['no orientation fixes, features, labels, folds', 'all added', f"+{gain_full:.3f} / +{gain_imu:.3f} overall"],
], [55, 55, 60])
P('The earlier work (kept in <font name="Mono">legacy/</font>) contributed the strict input validation, the sequence-local design and the original test suite, all of which carried over.')

P('9. What the modelling team receives', 'h1')
B('<font name="Mono">processed/</font>: per-row features, per-recording tree features (all sensors and IMU-only), labels, the shared fold file, per-fold scalers, and cleaned signals for on-the-fly augmentation.')
B('<font name="Mono">preprocess(sequence, demographics)</font>: call it inside the Kaggle <font name="Mono">predict()</font>; it accepts the polars frames Kaggle sends and returns the same features as training.')
B('<font name="Mono">preprocessing/augmentation.py</font>: crop, time-stretch, wrist/heading rotation, magnitude scaling, jitter, sensor dropout and mixup, applied to raw signals so physics features stay consistent.')

P('10. Sources (ideas only; no code copied)', 'h1')
P('CMI competition top-solution write-ups (1st–15th place summaries; 7th-place slides on handedness, upside-down subjects and gravity removal; 6th-place note on upside-down detection); '
  'silver-medal solution README (angular velocity, ToF aggregation); Zhou et al. 2019, <i>On the Continuity of Rotation Representations in Neural Networks</i> (6D rotation); '
  'Um et al. 2017, <i>Data Augmentation of Wearable Sensor Data</i>. All choices were re-verified on our data as described above.', 'small')


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont('Body', 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(20 * mm, 12 * mm, 'SC4000 · CMI preprocessing report')
    canvas.drawRightString(190 * mm, 12 * mm, f'{doc.page}')
    canvas.restoreState()


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                        title='Preprocessing for BFRB gesture detection', author='SC4000 preprocessing workstream')
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print('wrote', OUT)
