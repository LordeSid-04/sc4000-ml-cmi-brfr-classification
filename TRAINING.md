# Training, tuning and evaluating the models

```
models/
  common.py            data shaping, GPU scaling, competition metric, OOF files, bootstrap, Holm
  seqnet.py            shared blocks + base network          cnn.py  gru.py  transformer.py  cnn_gru.py (main)
  lgbm.py xgb.py rf.py mlp.py   summary-feature models        registry.py  name -> model
  hybrids.py           blend / stacker / trees-on-probabilities
  balance.py           per-person balancing at test time
  evaluate.py          table | compare | pairs | hybrid | ensemble | deploy | analysis | seeds | breakdown
  train/
    nn_trainer.py      loop shared by the networks             tree_trainer.py  loop shared by the tree-family models
    cnn.py gru.py transformer.py cnn_gru.py lgbm.py xgb.py rf.py mlp.py   one entry point per model
    augment.py         pre-computes augmented copies           tune.py          nested Optuna tuning
    run_all.py         the whole pipeline, in parallel
submission/            submit.py (inference code) and submission.ipynb (the Kaggle notebook)
```

## Folds are pinned to a file

`StratifiedGroupKFold` splits people differently in scikit-learn 1.7 and 1.8+, even with the same seed. The published
split is `results/06_validation/folds_used.csv`, and `scripts/01_build_dataset.py` (run by `run_pipeline.py`) **reuses it
automatically**. Pass `--new-folds` only for an experiment that need not match the published results.
`data/processed/` is not committed; it is rebuilt on demand.

**Never copy `data/` or `results/` from a fresh code download over a trained run:** they belong to that run.

## Run everything

```
python -m models.train.run_all --data-zip <competition zip or folder> --final --ablations --sanity
```

`--final` implies `--tune --n1` and adds the submitted ensemble (below). Re-running the same command resumes:
finished runs and finished tuning are skipped, except that a run which must keep its weights (`--save`) is retrained
when its weights are missing. A quick first pass: `--seeds 0` (one seed instead of three) and no `--ablations`.

## Pass 3: the final model

What is submitted: three networks trained on left-hand-mirrored data (N1), all with saved weights:
`n1_mirror` (tuned CNN-BiGRU), `n1_gru` (BiGRU) and `n1_transformer` (Transformer). Their equal-weight average is
`final`; at test time each person's predictions are balanced against the training protocol (`models/balance.py`).
`run_all --final` trains them, evaluates them and writes `runs/deploy.json`, which the submission reads.

**On Kaggle (free GPU T4 x2, about 2-2.5 hours):**

1. On GitHub switch to the `modelling` branch, **Code → Download ZIP**, and upload it as a **new version** of the
   `cmi-code` dataset.
2. **Create → New Notebook.** Settings: **Accelerator GPU T4 x2**, **Persistence: Files only** (optional).
   Add inputs: the **competition**, **`cmi-code`** (latest version) and your **Pass 2 notebook** (its output holds
   `repo/runs/` with every Pass 1-2 prediction and the tuned models).
3. Cell 1: start from the new code plus the Pass 2 predictions (never Pass 2's `data/` or `results/`):

```python
import glob, os, shutil
code = [p for p in glob.glob('/kaggle/input/**/run_pipeline.py', recursive=True) if '/cmi-code/' in p][0]
prev = [p for p in glob.glob('/kaggle/input/**/runs/cnn_gru_tuned/full_s2_f4.npy', recursive=True)][0]
shutil.copytree(os.path.dirname(code), '/kaggle/working/repo', dirs_exist_ok=True)
shutil.copytree(os.path.dirname(os.path.dirname(prev)), '/kaggle/working/repo/runs', dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('logs'))
DATA = os.path.dirname(glob.glob('/kaggle/input/**/train_demographics.csv', recursive=True)[0])
print('code:', code, '\nprevious runs:', os.path.dirname(os.path.dirname(prev)), '\ndata:', DATA)
```

4. Cell 2: **Save Version → Save & Run All (Commit)** so it keeps running with the browser closed:

```python
%cd /kaggle/working/repo
!python -m models.train.run_all --data-zip {DATA} --final --ablations --sanity --workers 4
```

   It rebuilds both datasets with the pinned folds, trains only what is missing (90 network runs: the three mirrored
   networks x 2 sensor settings x 5 folds x 3 seeds, all saved), re-evaluates everything and ends with
   `finished in ... min` and no `FAILED` line. Check that the log shows `deploying n1_mirror + n1_gru + n1_transformer`,
   the `final ensemble` table and the `per-person balancing` table.
5. Download `cmi_results.zip` (the results for the repository). `cmi_models.zip` / `repo/runs/` is what the
   submission needs; it can stay on Kaggle.

**Submit (CPU notebook, internet off):** Code → New Notebook from the competition page; **Accelerator None**,
**Internet Off**; inputs: `cmi-code` (latest version) and the **Pass 3 notebook**. One cell:

```python
import glob, os
code = [p for p in glob.glob('/kaggle/input/**/submission/submit.py', recursive=True) if '/cmi-code/' in p][0]
runs = [p for p in glob.glob('/kaggle/input/**/runs/deploy.json', recursive=True)][0]
os.environ['CMI_CODE'], os.environ['CMI_RUNS'] = os.path.dirname(os.path.dirname(code)), os.path.dirname(runs)
# os.environ['CMI_BAL'] = '0'          # submit without per-person balancing (to measure it on the leaderboard)
exec(open(code).read())
```

The first line printed must be `SUBMITTING: networks ['n1_mirror', 'n1_gru', 'n1_transformer'] (15, 15, ... checkpoints ...),
trees - (0 models), weight on networks {'full': 1.0, 'imu': 1.0}, balancing strength 4.0, mirrored inputs for [...]`
(the strength may differ). Then **Save Version → Save & Run All**, and **Submit Prediction** with that version.
Other choices without retraining: `CMI_NN=n1_mirror` (one network), `CMI_NN=cnn_gru_tuned CMI_GB=lgbm_tuned
CMI_W=0.6,0.8 CMI_BAL=0` (the first submission).

## How the results stay honest

| Rule | Where |
| --- | --- |
| Folds are split by person; the test set has new people | `sequences.csv` `fold`, built by the preprocessing |
| Nothing is chosen on a reported fold: fixed epochs, EMA weights of the last epoch | `nn_trainer.py` |
| Scalers are fitted on training recordings only; augmented copies of validation people are never trained on | `common.py`, `nn_trainer.py` |
| Tuning sees only outer folds 1-4 (inner subject folds); outer fold 0 is untouched, reported as `fold0` | `tune.py`, `evaluate.py table` |
| Hybrids are fitted on the other folds' out-of-fold predictions only | `hybrids.py` |
| 3 seeds, 95% subject-level bootstrap CIs (people are the independent units) | `evaluate.py table` |
| "A beats B" only if the Holm-corrected bootstrap p-value < 0.05 (plus folds won) | `evaluate.py compare` |
| Shuffled-label models must collapse to chance, otherwise something leaks | `run_all --sanity` |
| The Kaggle path produces exactly the training inputs (parity tests) | `tests/test_models.py`, `scripts/02_parity_check.py` |

## Results (plain files)

| Path | Content |
| --- | --- |
| `runs/<model>/<kind>_s<seed>_f<fold>.npy` | out-of-fold probabilities (kind: `full`, `imu`, `blank`) |
| `runs/<deployed>/*.pt`, `*.joblib`, `runs/deploy.json`, `runs/processed*/` | what the submission loads |
| `runs/logs/` | one log per job (kept with the Kaggle output, not in git) |
| `results/<NN_topic>/table_*.csv`, `compare_*.csv` | report tables and statistics, one folder per topic (`models/paths.py`) |
| `results/02_hyperparameter_tuning/params/` | tuned settings (read by the trainers) and every tuning trial |
| `results/05_error_analysis/<model>/` | per-class F1, confusion matrices, per-person and per-posture scores |
| `results/<NN_topic>/figures/` | the report figures (PNG), redrawn by `python -m models.report_figures` |
| `cmi_models.zip`, `cmi_results.zip` | `runs/` and `results/` zipped for download / upload as Kaggle Datasets |

## Single steps

```
python -m models.train.cnn_gru --variant full --folds 0 --seeds 0 --epochs 5     # one network, one fold
python -m models.train.lgbm --params results/02_hyperparameter_tuning/params/lgbm.json --name lgbm_tuned  # tuned tree model
python -m models.train.tune --model cnn_gru --trials 12
python -m models.evaluate compare cnn_gru_tuned lgbm_tuned cnn gru transformer xgb rf mlp
python -m models.evaluate analysis cnn_gru_tuned
python -m models.evaluate ensemble final n1_mirror n1_gru n1_transformer   # equal-weight ensemble -> runs/final
python -m models.evaluate deploy final n1_mirror n1_gru n1_transformer     # balancing (nested) + runs/deploy.json
python -m models.report_figures
python -m pytest tests/test_models.py
```
