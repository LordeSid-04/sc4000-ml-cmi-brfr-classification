# Training, tuning and evaluating the models

```
models/
  common.py            data shaping, GPU scaling, competition metric, OOF files, bootstrap, Holm
  seqnet.py            shared blocks + base network          cnn.py  gru.py  transformer.py  cnn_gru.py (main)
  lgbm.py xgb.py rf.py mlp.py   summary-feature models        registry.py  name -> model
  hybrids.py           blend / stacker / trees-on-probabilities
  evaluate.py          table | compare | hybrid | analysis
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
python -m models.train.run_all --data-zip <competition zip or folder> --tune --ablations --sanity --n1
```

Re-running the same command resumes: finished runs and finished tuning are skipped. A quick first pass:
`--seeds 0` (one seed instead of three) and no `--ablations`.

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
| `runs/<deployed>/*.pt`, `*.joblib`, `runs/blend_weights.json`, `runs/deploy.json` | what the submission loads |
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
python -m models.evaluate ensemble ens_a n1_mirror gru transformer   # equal-weight ensemble -> runs/ens_a
python -m models.report_figures
python -m pytest tests/test_models.py
```
