# SC4000 · CMI – Detect Behavior with Sensor Data: preprocessed data for modelling

Everything you need to train models for the Kaggle competition
[CMI – Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).
The raw sensor data is already cleaned and turned into features. You load the files and train.

**The task:** a wrist device records motion, temperature and distance while a person does one of 18
gestures. Predict the gesture. 8 gestures are BFRBs (hair pulling, skin pinching, ...), 10 are everyday gestures.

---

## 1. Setup (once)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

1. **Small data files** (labels, folds, tree features, scalers) come with the repo in `data/processed/`.
2. **Big data files** (`features.npz`, `canonical/`): download `processed_large_files.zip` from this
   repo's **Releases** page and unzip it **in the repo folder**. The files land in `data/processed/`.
   You only need these for neural networks.
3. **Raw Kaggle data** (only needed to rebuild or to test the Kaggle submission):
   `kaggle competitions download -c cmi-detect-behavior-with-sensor-data -p data/raw`

## 2. What is in `data/processed/`

| File | Use it for | What it is |
|---|---|---|
| `sequences.csv` | **everything** | one row per recording (8,151): `sequence_id`, `subject`, labels, `fold`, and `start_row`/`rows` to find its rows in `features.npz` |
| `tree_full.parquet` | **tree models** (LightGBM, XGBoost, ...) | one row per recording, 795 features, all sensors |
| `tree_imu.parquet` | **tree models** | the same, but with temperature and distance sensors blanked (see section 5) |
| `features.npz` | **neural networks** | `features`: 574,945 rows × 117 features (one row per time step); `phase`: per-row phase label |
| `scalers/fold0-4.json`, `all.json` | **neural networks** | mean/std to normalise the 117 features |
| `canonical/` | **augmentation** (optional) | the cleaned sensor signals, so you can make new training variants |
| `folds.csv` | cross-validation | which subject is in which fold (also in `sequences.csv`) |
| `meta.json` | reference | the 117 feature names, fixed length `L = 128`, gesture list |

All rows of `sequences.csv`, `tree_*.parquet` and `canonical/` are in the **same order**.

## 3. Labels and the score

Columns in `sequences.csv`:

| Column | Values | Notes |
|---|---|---|
| `gesture_id` | 0–17 | **main target.** `0–7` are the BFRBs, `8–17` the others. Names: `preprocessing.constants.GESTURES` |
| `gesture` | text | the same as text |
| `score_id` | 0–8 | the 9 classes Kaggle scores (8 BFRBs + 1 "non_target") |
| `is_target` | 0 / 1 | BFRB or not |
| `orientation`, `initial_state` | text | body position and starting state. Optional extra targets, **never** inputs |

**Kaggle score** = 0.5 × F1(BFRB vs not) + 0.5 × macro F1 over the 9 classes. Use
`preprocessing.step4_labels.competition_score(true_gestures, predicted_gestures)`, which returns
`(score, binary_f1, macro_f1)`.

**Phase** (in `features.npz`, one per row): `0` moving the hand, `1` pausing at the target, `2` doing the
gesture. The gesture is always at the **end** of a recording. Phase is an optional extra target for neural networks.

## 4. Cross-validation: always use the `fold` column

`fold` (0–4) puts each **subject** in exactly one fold. The test set contains unseen people, so a subject
must never be in train and validation at the same time. Train on `fold != k`, validate on `fold == k`.
**Do not make your own random split**: scores will look too good and won't match Kaggle.

## 5. Important: half the test set has motion sensors only

About half of the hidden test recordings have only the motion sensors (IMU: acceleration + rotation).
Temperature (`thm`) and distance (`tof`) are switched off. Your model must work in both cases:

- **Trees:** train on `tree_full` **and** `tree_imu` stacked together (as below), and report both scores.
- **Neural networks:** use the `augment()` function (section 6). It blanks temperature/distance 50% of the time.

Missing sensors appear as `NaN` in the features, and the flags `rot_missing`, `thm_missing`, `tof_missing` say which.

## 6. Examples

### Tree model (LightGBM)

```python
import pandas as pd
import lightgbm as lgb
from preprocessing.constants import GESTURES
from preprocessing.step4_labels import competition_score

seqs = pd.read_csv('data/processed/sequences.csv')
X_full = pd.read_parquet('data/processed/tree_full.parquet').drop(columns='sequence_id')
X_imu = pd.read_parquet('data/processed/tree_imu.parquet').drop(columns='sequence_id')
y, fold = seqs['gesture_id'], seqs['fold']

k = 0
train, valid = fold != k, fold == k
model = lgb.LGBMClassifier(verbose=-1)
model.fit(pd.concat([X_full[train], X_imu[train]]), pd.concat([y[train], y[train]]))

for name, X in [('all sensors', X_full), ('IMU-only', X_imu)]:
    pred = model.predict(X[valid])
    score, binary_f1, macro_f1 = competition_score(
        [GESTURES[i] for i in y[valid]], [GESTURES[i] for i in pred])
    print(f'fold {k} {name}: {score:.3f}')
```

Trees handle `NaN` themselves, so no scaling is needed. Our reference LightGBM scores about **0.815 (all
sensors)** and **0.740 (IMU-only)** averaged over the 5 folds. That's the baseline to beat.

### Neural network input for one recording

```python
import json
import numpy as np
import pandas as pd
from preprocessing.dataset import sequence_slices
from preprocessing.step5_shaping import Scaler, fixed_length

seqs = pd.read_csv('data/processed/sequences.csv')
meta = json.load(open('data/processed/meta.json'))
z = np.load('data/processed/features.npz')
features, phase = z['features'], z['phase']
slices = sequence_slices(seqs)                                   # row range of each recording
scaler = Scaler.load('data/processed/scalers/fold0.json')        # use the scaler of the fold you validate on
L = meta['fixed_length']                                         # 128

i = 0                                                            # recording number (row in sequences.csv)
rows = pd.DataFrame(features[slices[i]], columns=meta['feature_columns'])
x, mask = fixed_length(scaler.transform(rows), L)                # x: (128, 117), mask: 1 = real row, 0 = padding
p, _ = fixed_length(phase[slices[i]], L)                         # per-row phase target (optional)
label = seqs.loc[i, 'gesture_id']
```

- `scaler.transform` normalises each feature and turns `NaN` into 0.
- `fixed_length` keeps the **last** 128 rows (the gesture is at the end) and pads short recordings at the start.
- Use `scalers/fold{k}.json` when validating on fold `k`, and `scalers/all.json` for the final model trained on everything.

### Augmentation (neural networks, training only)

```python
import numpy as np
from preprocessing.dataset import CanonicalStore
from preprocessing.augmentation import augment
from preprocessing.pipeline import featurize

store = CanonicalStore('data/processed/canonical')               # cleaned signals, same order as sequences.csv
rng = np.random.default_rng()

clean = store.get(i)                                             # recording i
feats, tof_images = featurize(augment(clean, rng))               # random variant -> (rows, 117) features
x, mask = fixed_length(scaler.transform(feats), L)
```

`augment()` randomly crops, speeds up/slows down, rotates and adds noise, and **blanks temperature/distance
50% of the time** (section 5). Call it again each epoch for new variants. Never augment validation data.
`tof_images` gives `(rows, 5, 8, 8)` distance images if you want a CNN on the ToF sensors.

### Kaggle submission

On Kaggle, upload the `preprocessing/` folder as a Kaggle Dataset and add it to your notebook. Kaggle
calls `predict()` once per recording. `preprocess()` produces exactly the same features as the training
files (this is checked automatically, section 8):

```python
import sys
sys.path.append('/kaggle/input/<your-dataset-name>')            # folder that contains preprocessing/

import pandas as pd
import polars as pl
from preprocessing.constants import GESTURES
from preprocessing.pipeline import preprocess
from preprocessing.step5_shaping import tree_summary

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    out = preprocess(sequence, demographics)                     # cleaning + orientation + features
    # tree model:
    row = pd.DataFrame([tree_summary(out.features)])[tree_columns]   # tree_columns = X_full.columns from training
    return GESTURES[int(model.predict(row)[0])]
    # neural network instead:
    # x, mask = fixed_length(scaler.transform(out.features), 128)
```

## 7. The 117 features (per time step)

| Group | Columns |
|---|---|
| Acceleration | `acc_x/y/z`, `acc_mag` |
| Acceleration without gravity | `linear_acc_x/y/z`, `linear_acc_mag`, `linear_acc_vertical`, `linear_acc_horizontal` |
| Wrist tilt | `gravity_dir_x/y/z` |
| Rotation | `rot_w/x/y/z`, `rot6d_1..6` |
| Movement | `jerk_x/y/z`, `jerk_mag`, `acc_cos_sim`, `angular_vel_x/y/z`, `angular_distance` |
| Temperature | `thm_1..5`, `thm_mean`, `thm_1..5_diff` |
| Distance (5 sensors) | `tof_k_mean/max/std/valid_frac/mean_diff` and 3×3 regions `tof_k_r00..r22` |
| Flags | `rot_missing`, `thm_missing`, `tof_missing` |
| Position | `time_frac` (0 = start, 1 = end of recording) |

The tree files summarise each feature over the recording: `__mean`, `__std`, `__min`, `__max`,
`__tail_mean`, `__tail_std` (last 25%) and `__last`, plus the flags and `length`.

What the cleaning already did for you: rows ordered, gaps filled, rotation sign flips fixed, distance
`-1` ("nothing there") turned into "far", the 2 subjects who wore the device upside down rotated back, and
left-handed subjects mirrored so they look like right-handers. **You don't need to clean anything.**

## 8. Code layout and rebuilding

```
data/raw/                 Kaggle ZIP (not in git)
data/processed/           the files above
preprocessing/            the pipeline, in the order it runs
  step1_cleaning.py         order rows, fill gaps, fix rotation, ToF -> proximity
  step2_orientation.py      upside-down fix, left-hand mirroring
  step3_features.py         the 117 features
  step4_labels.py           labels and competition_score()
  step5_shaping.py          fixed_length(), tree_summary(), Scaler, folds
  pipeline.py               preprocess(): steps 1-3 for one recording (used on Kaggle)
  augmentation.py           augment() for training
  dataset.py                builds the training set, CanonicalStore, sequence_slices()
scripts/
  01_build_dataset.py       builds data/processed/
  02_parity_check.py        checks Kaggle preprocess() == training features
  03_quality_checks.py      22 checks on the built data
tests/                    32 unit tests
run_pipeline.py           runs all of the above in order
```

To rebuild `data/processed/` from the raw data (about 6 minutes):

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

It runs the tests, builds the data, and checks it: Kaggle parity, plus 22 quality checks such as no
subject in two folds, all 18 gestures in every fold, and no label leakage. It only replaces
`data/processed/` if **every** check passes. You only need this if the preprocessing code changes.
