# SC4000 · CMI – Detect Behavior with Sensor Data: preprocessing

This repository turns the raw sensor recordings of the Kaggle competition
[CMI – Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)
into clean, consistent features that the modelling team trains on. Below: what the data looks like,
what is wrong with it, how we clean it, which features we build and why, and how much each step helps.

**Contents:** [1. Background](#1-background) · [2. Why the raw data needs cleaning](#2-why-the-raw-data-needs-cleaning) ·
[3. Cleaning](#3-cleaning) · [4. Putting everyone in the same position](#4-putting-everyone-in-the-same-position) ·
[5. Features](#5-features) · [6. Labels, folds and model input](#6-labels-folds-and-model-input) ·
[7. Validation and evidence](#7-validation-and-evidence) · [8. Output files](#8-output-files) ·
[9. Using it: two ways](#9-using-it-two-ways) · [10. Code and rebuilding](#10-code-and-rebuilding)

---

## 1. Background

People wore a wrist device (Helios) while doing one of **18 gestures**. Eight of them are
**BFRBs** (body-focused repetitive behaviours such as hair pulling, skin pinching and scratching). Ten are
everyday gestures, such as drinking, waving or texting. The task is to recognise the gesture from the sensors.

| Sensor | Columns | What it measures |
|---|---|---|
| Accelerometer | `acc_x/y/z` | movement + gravity, in m/s² |
| Rotation | `rot_w/x/y/z` | how the wrist is turned, as a quaternion |
| Thermopiles (5) | `thm_1..5` | temperature of whatever the sensor faces, in °C |
| Time-of-flight (5) | `tof_1..5_v0..v63` | distance images, 8×8 pixels per sensor |

- **Data size:** 8,151 recordings, 574,945 rows (about 10 rows per second) and 81 people. Recordings are
  29–700 rows long, with a median of 59.
- **What happens in a recording:** the hand moves to the target (for example the head), pauses, then
  performs the gesture. **The gesture is always the last part.**
- **The test set:** it contains **new people**, and in **about half of it only the accelerometer and
  rotation work** (no temperature or distance).
- **The score:** half is "BFRB or not?" and half is "which BFRB?".

## 2. Why the raw data needs cleaning

| Problem in the raw data | How common | Why it hurts a model |
|---|---|---|
| Distance pixels read `-1` when nothing is in range | **58%** of all ToF pixels | `-1` sits below every real distance, so a model reads "nothing there" as "closer than anything" |
| A quaternion can jump between `q` and `−q` (the same rotation) | **13,644** jumps in **4,730** recordings (58%) | looks like a sudden full twist of the wrist that never happened |
| Whole sensors missing for a recording | rotation: 50 recordings, temperature: 96, distance: 96, sensor 5 alone: ~480 | NaN breaks most models; filling with a fake value invents a reading |
| **10 left-handed** people wear the device on the other wrist | 1,020 recordings (12.5%) | their movements are a mirror image of everyone else's |
| **2 people wore the device upside down** | SUBJ_019262, SUBJ_045235 | all their signals are rotated 180° |
| Gravity (9.8 m/s²) dominates the accelerometer | every row | the same gesture looks different sitting vs lying down, because gravity points along a different axis |
| Recordings have different lengths | 29–700 rows | neural networks need a fixed input size |

## 3. Cleaning

`preprocessing/step1_cleaning.py`, applied to **one recording at a time**. It uses no labels and no other
people, so it can run unchanged inside Kaggle's `predict()`.

| # | Step | What it does | Why |
|---|---|---|---|
| 1 | **Order rows** | sort by `sequence_counter`, drop duplicates | a safety step. The training data is already ordered, but Kaggle input is not guaranteed to be |
| 2 | **Fill gaps** | a gap inside a recording is filled by linear interpolation. A sensor that is missing for the **whole** recording stays empty, with a flag (`rot_missing`, `thm_missing`, `tof_missing`) | never invent a reading. The model is told the sensor is off instead of being fed fake numbers |
| 3 | **Fix quaternion sign flips** | flip `q` to `−q` whenever it jumps away from the previous row | removes the fake "full twist" jumps (13,644 of them) |
| 4 | **Distance → proximity** | `proximity = 1 − distance / 255`, so 1 = touching and 0 = far. `-1` becomes **0 proximity** | an explicit no-object modelling convention; missing NaN readings remain distinct and invalid-pixel flags are retained |

The cleaning **repairs** bad input instead of crashing, because one crash on Kaggle fails the whole
submission. For checking the training data there is also a `strict=True` mode that stops on anything unexpected.

## 4. Putting everyone in the same position

`preprocessing/step2_orientation.py` contains **exploratory corrections, disabled by default**.
Their settings were chosen using labels from all subjects, so the validated dataset does not apply them.
The following describes the optional functions, not the default processing.

| Fix | Who | What it does |
|---|---|---|
| **Upside-down fix** | the 2 people who wore the device upside down | rotate all signals 180° back. Sensors 2↔4 and 3↔5 swap places and distance images turn 180° |
| **Left-hand mirroring** | the 10 left-handed people | mirror the x-axis so they look like right-handers. Applied to **every** sensor: acceleration, rotation, sensor positions (3↔5 swap) and distance images (flipped left–right on sensors 1, 2, 4, up–down on 3, 5, which are mounted sideways) |

These settings were **measured on the data, not guessed**. We compared each person's movement pattern with
the right-handed majority and picked the transformation that matches best:
- **Left-handers:** agreement rises from **0.38 to 0.64**, the same as right-handers have with each other (0.64).
- **Upside-down people:** agreement rises from **−0.20 to 0.62**.

*Limitation:* the upside-down fix only knows these 2 training people. A new upside-down person in the test
set would not be detected (this harms no one else).

## 5. Features

`preprocessing/step3_features.py` turns **every time step** (one row, about 0.1 s) of a recording into
**117 features**. A feature that cannot exist, for example temperature when the sensor is off, is `NaN`.
Section 9 shows the two ways models use them: per time step (neural networks) or summarised per recording
(tree models).

| Group | # | Features | What it captures, and why it helps |
|---|---|---|---|
| **Raw acceleration** | 4 | `acc_x/y/z`, `acc_mag` | the base movement signal |
| **Gravity removed** | 4 | `linear_acc_x/y/z`, `linear_acc_mag` | the hand's own movement without gravity, so the same gesture looks the same sitting or lying. Gravity is computed from the rotation, or estimated from the slow part of the signal when rotation is missing |
| **Wrist tilt** | 3 | `gravity_dir_x/y/z` | which way the wrist points. A hand at the head is tilted differently from a hand at the knee |
| **Up/down vs sideways** | 2 | `linear_acc_vertical`, `linear_acc_horizontal` | movement direction in the room, independent of how the watch sits |
| **Rotation** | 10 | `rot_w/x/y/z`, `rot6d_1..6` | wrist orientation. The 6D version has no `q`/`−q` ambiguity and is easier for networks to learn |
| **Jerk** | 4 | `jerk_x/y/z`, `jerk_mag` | how fast acceleration changes. Scratching and pulling are short, jerky movements |
| **Direction change** | 1 | `acc_cos_sim` | how much the movement direction changes between rows (back-and-forth motion) |
| **Turning speed** | 4 | `angular_vel_x/y/z`, `angular_distance` | how fast the wrist rotates, for example twisting while pulling hair |
| **Temperature** | 11 | `thm_1..5`, `thm_mean`, `thm_1..5_diff` | skin is warm. Shows when the hand is next to the body and when it arrives there |
| **Distance per sensor** | 25 | `tof_k_mean/max/std/valid_frac/mean_diff` | how close something is, the closest point, and how it changes |
| **Distance regions** | 45 | `tof_k_r00..r22` (3×3 grid per sensor) | **where** in the image the object is (centre vs edge); a candidate spatial feature |
| **Sensor flags** | 3 | `rot_missing`, `thm_missing`, `tof_missing` | tell the model which sensors were available |
| **Position** | 1 | `time_frac` | 0 = start, 1 = end of the recording |
| | **117** | | |

Smoothing is switched off. Earlier exploratory comparisons suggested a decrease, but that comparison
is not a separately held-out causal result.

## 6. Labels, folds and model input

`preprocessing/step4_labels.py` and `step5_shaping.py`.

- **Labels:** `gesture_id` (18 classes, the main target), `score_id` (the 9 classes Kaggle scores: 8 BFRBs +
  "non-target"), `is_target` (BFRB or not), and a per-row `phase` (0 moving, 1 pausing, 2 gesture). Labels
  are training targets only, never model inputs.
- **Folds:** 5 folds, split **by person**, so nobody is in train and validation at the same time. The test
  set has new people, so this is the only honest way to validate. Every fold contains all 18 gestures.
- **Tree models** get one row per recording (795 features, section 9).
- **Neural networks** get a fixed length of **128 rows**, which keeps 95% of recordings whole. Longer recordings
  keep their **last** 128 rows, shorter ones are padded at the start, and a mask marks the real rows. Features are
  normalised with a scaler fitted only on the training folds.
- **Augmentation** (`augmentation.py`, training only): random crop, speed change, small rotations, noise, and
  **switching off temperature and distance 50% of the time**, so models learn to cope with the motion-only
  half of the test set.

## 7. Validation and evidence

The earlier orientation axes, sensor permutations and image flips were selected using gesture labels
from the complete dataset before cross-validation. The previous 0.815 / 0.740 scores and orientation
ablation gains were exploratory, not strictly held-out evidence, and are withdrawn as validation claims.

The validated dataset now uses `PipelineConfig()` with `mirror_left=False` and
`fix_upside_down=False`. Neither globally label-tuned correction is applied to training, validation
or inference. The optional functions remain available for exploration. Using them in a claimed held-out
experiment requires independent physical evidence or fitting only on each fold's training subjects.

After rebuilding, run `python scripts/04_validate_trees.py`. It evaluates the conservative pipeline
using the shared subject-grouped folds and saves parameters, per-fold scores and aggregate results to
`data/processed/validation.json`. The features and reference model settings are fixed for this run.
This is a grouped-CV reference, not an untouched final test score; repeated tuning on these folds can
still overfit them.

Reference run on 15 October 2026 (LightGBM 4.7.0, five subject-grouped folds):

| Inputs | Mean fold score | Fold standard deviation | Pooled out-of-fold score |
|---|---:|---:|---:|
| All sensors | 0.7964 | 0.0081 | 0.7963 |
| Motion only | 0.7302 | 0.0138 | 0.7305 |

Fold spread is not a confidence interval. Full settings and per-fold results are in `validation.json`.

## 8. Output files

Everything is in `data/processed/`. All files list the recordings in the **same order**.

| File | For | Contents |
|---|---|---|
| `sequences.csv` | all models | one row per recording: labels, person, `fold`, sensor flags, row position in `features.npz` |
| `tree_full.parquet` | tree models | 8,151 × 795 features, all sensors |
| `tree_imu.parquet` | tree models | the same with temperature/distance blanked (motion-only training and validation) |
| `features.npz` | neural networks | `features` (574,945 rows × 117) and `phase` (per row) |
| `scalers/` | neural networks | normalisation per fold (`fold0-4.json`) and for the final model (`all.json`) |
| `canonical/` | augmentation | the cleaned, position-corrected sensor signals |
| `folds.csv`, `meta.json` | reference | person → fold; feature names and settings |
| `validation.json` | evaluation | fixed grouped-CV parameters and scores (generated by the validation script) |

`features.npz` and `canonical/` are excluded from git. **There is currently no published Release download.**
Rebuild them from the original competition ZIP with the command in section 10. The runner also creates
`outputs/release/processed_large_files.zip` locally for optional sharing; creating this archive does not
upload a GitHub Release. Keep large files paired with metadata and scalers from the same build.
The tree tables and scalers in this branch have been rebuilt with orientation corrections disabled.
Do not reuse an older cached `features.npz` or `canonical/` folder with them; rebuild those arrays first.

## 9. Using it: two ways

The same 117 features come in two shapes. Use either one, or train both and average their predictions
(worth testing: the two model types tend to make different mistakes).

| | **A. Tree models** (LightGBM, XGBoost, CatBoost) | **B. Neural networks** (1D-CNN, GRU/LSTM, Transformer) |
|---|---|---|
| Shape | **1 row per recording × 795 features** | **128 time steps × 117 features** per recording |
| Files | `tree_full.parquet` + `tree_imu.parquet` | `features.npz` + `scalers/` (+ `canonical/` for augmentation) |
| How the 795 / 128 come about | each of the 113 sensor features is summarised 7 ways (mean, std, min, max, mean and std of the last 25%, last value) = 791, + 3 flags + length | the last 128 time steps, padded at the start if shorter, with a mask |
| Missing sensors (`NaN`) | handled by the tree itself | the scaler turns them into 0; the flags tell the model |
| Motion-only test half | train on `tree_full` and `tree_imu` stacked | `augment()` switches temperature/distance off 50% of the time |
| Effort | minutes to train; reference scores are recorded in `validation.json` | slower, but can learn the timing of a gesture |

Both use the same labels and folds from `sequences.csv`. Always train on `fold != k` and validate on
`fold == k`. Score with `competition_score()`:

```python
from preprocessing.step4_labels import competition_score
score, binary_f1, macro_f1 = competition_score(true_gesture_names, predicted_gesture_names)
```

### A. Tree models

```python
import pandas as pd
import lightgbm as lgb
from preprocessing.constants import GESTURES
from preprocessing.step4_labels import competition_score

seqs = pd.read_csv('data/processed/sequences.csv')
X_full = pd.read_parquet('data/processed/tree_full.parquet').drop(columns='sequence_id')
X_imu = pd.read_parquet('data/processed/tree_imu.parquet').drop(columns='sequence_id')
y, fold = seqs['gesture_id'], seqs['fold']

params = dict(learning_rate=0.05, n_estimators=500, num_leaves=31, min_child_samples=20,
              colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
              random_state=42, verbose=-1)              # our reference settings (section 7)

for k in range(5):
    train, valid = fold != k, fold == k
    model = lgb.LGBMClassifier(**params)
    model.fit(pd.concat([X_full[train], X_imu[train]]), pd.concat([y[train], y[train]]))
    for name, X in [('all sensors', X_full), ('motion only', X_imu)]:
        pred = model.predict(X[valid])
        score, _, _ = competition_score([GESTURES[g] for g in y[valid]], [GESTURES[g] for g in pred])
        print(f'fold {k} {name}: {score:.3f}')
```

### B. Neural networks

```python
import json
import numpy as np
import pandas as pd
from preprocessing.dataset import sequence_slices
from preprocessing.step5_shaping import Scaler, fixed_length

seqs = pd.read_csv('data/processed/sequences.csv')
meta = json.load(open('data/processed/meta.json'))
features = np.load('data/processed/features.npz')['features']
slices = sequence_slices(seqs)                                   # where each recording's rows are

k = 0                                                            # fold to validate on
scaler = Scaler.load(f'data/processed/scalers/fold{k}.json')     # fitted on the other folds only

def model_input(i):
    """Recording i (row i of sequences.csv) -> (128, 117) input and (128,) mask."""
    rows = pd.DataFrame(features[slices[i]], columns=meta['feature_columns'])
    return fixed_length(scaler.transform(rows), 128)

x, mask = model_input(0)
label = seqs.loc[0, 'gesture_id']
```

**Augmentation** (training only, a new random variant each call):

```python
from preprocessing.dataset import CanonicalStore
from preprocessing.augmentation import augment
from preprocessing.pipeline import featurize

store = CanonicalStore('data/processed/canonical')               # same order as sequences.csv
feats, tof_images = featurize(augment(store.get(0), np.random.default_rng()))
x, mask = fixed_length(scaler.transform(feats), 128)
```

`tof_images` gives the distance sensors as `(rows, 5, 8, 8)` images, if you want a CNN on them.

### On Kaggle

Upload the `preprocessing/` folder as a Kaggle Dataset. Inside `predict()`, one call gives exactly the
training features (checked automatically, section 10). Then use whichever shape your model needs:

```python
from preprocessing.pipeline import preprocess
from preprocessing.step5_shaping import tree_summary

out = preprocess(sequence, demographics)                         # clean + position fix + 117 features
tree_row = tree_summary(out.features)                            # A: 795 features for tree models
scaler = Scaler.load('data/processed/scalers/all.json')          # B: final model uses the all-data scaler
x, mask = fixed_length(scaler.transform(out.features), 128)      #    (128, 117) for neural networks
```

## 10. Code and rebuilding

```
preprocessing/               the pipeline, in the order it runs
  step1_cleaning.py            section 3
  step2_orientation.py         section 4
  step3_features.py            section 5
  step4_labels.py              labels and competition_score()
  step5_shaping.py             tree summaries, fixed length, scaler, folds
  pipeline.py                  preprocess(): steps 1-3 for one recording (used on Kaggle)
  augmentation.py              training augmentations
  dataset.py                   runs the pipeline over the whole training set
  constants.py, data_loading.py, quaternion.py   helpers
scripts/
  01_build_dataset.py          builds data/processed/
  02_parity_check.py           Kaggle preprocess() == training features
  03_quality_checks.py         23 checks on the built data
  04_validate_trees.py         fixed subject-grouped reference evaluation
tests/test_preprocessing.py  34 unit tests
run_pipeline.py              runs everything above
```

Setup and rebuild:

Download the original competition ZIP from Kaggle's Data page first (joining the competition may be
required), then substitute its location below. A separate Kaggle command-line installation is not needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_pipeline.py --data "C:\path\to\cmi-detect-behavior-with-sensor-data.zip"
.\.venv\Scripts\python.exe scripts/04_validate_trees.py
```

`run_pipeline.py` runs the tests, builds the data, checks that the Kaggle path gives identical features, and
runs 23 quality checks: no person in two folds, all gestures in every fold, no label leakage, valid scaling,
and more. It replaces `data/processed/` **only if every check passes**, in about 5 minutes.
The separate tree-validation command trains five reference models and updates `validation.json`.
