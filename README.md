# SC4000: CMI sensor data cleaning

This handoff contains sequence-local cleaning code for CMI Detect Behavior with Sensor Data. Download the original competition ZIP separately. Raw and generated data are excluded from Git.

## Quick start

Use Python 3.10 or newer. Open a terminal in the repository folder.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe preprocess_cmi.py --zip "C:\path\to\cmi-detect-behavior-with-sensor-data.zip" --output processed_data
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python preprocess_cmi.py --zip /path/to/cmi-detect-behavior-with-sensor-data.zip --output processed_data
```

Replace the ZIP path with your own downloaded file. No extraction or Kaggle credentials are required by this script. Allow several minutes and roughly 1 GB of free disk space for generated output; actual size depends on the dataset. Choose a new output folder when rerunning: existing nonempty folders are rejected. A failed run can leave incomplete parts; use only outputs from a successfully completed run whose summary contains `"status": "complete"`, and start retries in a new folder.

Tested locally with Python 3.12.14, numpy 2.3.5 and pandas 3.0.1. The broader dependency ranges in requirements.txt are not a claim that every version combination was tested.

## Cleaning rules

| Input | Cleaning | Meaning of the observed flag |
|---|---|---|
| Acceleration | Keep finite readings; reject missing/nonfinite acceleration | True for all accepted readings |
| Rotation | Normalize usable quaternions; replace unavailable/zero-length quaternions with identity `[1,0,0,0]` | False for every component of an unusable quaternion |
| Temperature | Interpolate within the same recording by reading counter; carry nearest readings at edges; use zero for an entirely missing channel | False at all originally missing/nonfinite positions, including interpolated positions |
| Distance | Replace missing/nonfinite values and `-1` with zero; preserve genuine zero distances | False for replaced positions, True for usable original measurements |

These are explicit initial choices for modelling evaluation. They do not restore missing physical measurements or establish that a model will perform better. All changes are local to one whole sequence: no gesture labels, other subjects, or fitted population statistics are used. The script is suitable for offline prediction where the complete recording is available. Temperature interpolation uses future readings within that recording, so this is not a streaming transform.

## Outputs and how to read them

- `part_000.npz`, etc.: compressed numpy arrays containing `values`, `observed`, and `channels`.
- Matching `part_000_metadata.csv`, etc.: IDs and training metadata in the same row order.
- `sequence_index.csv`: sequence ID, part filename, zero-based start row, row count.
- `train_demographics.csv`: one row per subject, copied from the source; join by subject if needed.
- `cleaning_summary.json`: processing totals and selected example IDs.
- `example_*.csv`: selected raw/cleaned sensor values and flags for actual recordings. Examples are generated only for categories that exist in the input.

There are 332 sensor channels. `values` has shape `(number_of_rows, 332)` and dtype float32. The boolean `observed` array has exactly the same shape. Every part contains whole sequences. Float32 storage introduces small rounding differences from the original CSV.

Load a recording:

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('processed_data')
index = pd.read_csv(root / 'sequence_index.csv')
record = index.iloc[0]  # choose another row for another recording
start = int(record.start_row)
end = start + int(record.rows)
with np.load(root / record.part, allow_pickle=False) as part:
    values = part['values'][start:end].copy()
    observed = part['observed'][start:end].copy()
    channels = part['channels'].tolist()
metadata = pd.read_csv(root / record.part.replace('.npz', '_metadata.csv'))
sequence_metadata = metadata.iloc[start:end].reset_index(drop=True)
print(record.sequence_id, values.shape, sequence_metadata.gesture.iloc[0])
```

Distance summaries MUST exclude unobserved placeholders:

```python
tof_positions = [i for i, c in enumerate(channels) if c.startswith('tof_1_')]
distance = np.where(observed[:, tof_positions], values[:, tof_positions], np.nan)
valid_fraction = observed[:, tof_positions].mean(axis=1)
count = np.isfinite(distance).sum(axis=1)
mean_distance = np.divide(np.nansum(distance, axis=1), count,
                          out=np.full(len(values), np.nan), where=count > 0)
```

A zero count means the summary is unavailable. The identity rotation placeholder is not a measured orientation: do not use it for gravity removal unless rotation validity is checked. Temperature's flag identifies original observations, even when interpolation provided a plausible numeric input. Modellers may use interpolated temperatures, but should retain missingness information and exclude wholly absent channels from fitted statistics.

## Use the same cleaning function at prediction time

```python
from preprocess_cmi import clean_sequence

ordered_raw, channels, values, observed = clean_sequence(sequence_dataframe)
```

The input needs `sequence_id`, `sequence_counter`, and all 332 named sensor columns; unavailable sensors should be present as NaN columns. It does not need a gesture label or demographic columns. Column order is normalized. One call must contain exactly one recording. The CLI processes the supplied training ZIP; it expects each recording to occupy a contiguous block of source rows and rejects repeated noncontiguous sequences.

## Teammate's next steps

1. Split entire recordings by subject, keeping each subject in one split. Save the assignments.
2. Fit scaling or other learned transforms on training subjects only. Exclude padding and unavailable placeholders from fitted statistics.
3. Prepare per-sequence features for tree models or padded readings with a separate padding mask for sequence models.
4. Use `gesture` as the label. Other training-only helper columns (`phase`, `behavior`, `orientation`, `sequence_type`) are not prediction-time input features.
5. Save label mapping, scaling parameters and feature/padding choices. Reuse exactly the same transforms for validation and inference.

This cleaning handoff does not implement gravity removal, handedness mirroring, sensor dropout, scaling, model features, sequence truncation or labels. It also does not correct physically unusual readings other than the documented missing/invalid cases.

## Audit and verification

Optional full audit:

```powershell
.\.venv\Scripts\python.exe audit_training_data.py --zip "C:\path\to\cmi-detect-behavior-with-sensor-data.zip"
```

The supplied original archive contained 574,945 rows, 8,151 sequences, 81 subjects and 18 gestures. Acceleration was complete. Rotation was entirely unavailable in 50 sequences. Temperature had gaps in 508 sequences; distance had missing values in 435. Both temperature and distance were entirely missing in 96 sequences. Every sequence contained some invalid ToF pixels (`-1`). These are measured audit results for that archive, not hardcoded requirements of the cleaning function.

The 11-test suite covers sequence ordering, column alignment, missing-sensor cases, genuine zero distance, temperature interpolation, quaternion normalization, input nonmutation, label independence, invalid inputs, chunk and output-part boundaries, empty-input rejection, output overwrite protection, noncontiguous sequence rejection and subject/demographic matching. Real train/test parity, independent source comparisons and full-dataset saved-output reconciliation are also checked locally; see VERIFICATION.md for results.
