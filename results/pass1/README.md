# Pass 1 results (real competition data, 1 seed)

**Run:** Kaggle notebook, 2x Tesla T4, 24 Oct 2026, 324 min.
**Command:** `python -m models.train.run_all --data-zip <competition data> --seeds 0 --tune --sanity --workers 4`.
**Code:** `modelling` branch at commit `032736e`.

**Verified on 24 Oct 2026.** Every score in these tables was recomputed from the raw out-of-fold predictions
(`runs/`, kept in `cmi_models.zip`, not in git) and matches to 5 decimals. The confusion matrices' true-class counts equal
the dataset's (638-640 per BFRB, 3,038 non-target; 8,151 recordings, 81 people).

## Folds: read this before comparing numbers

The person-to-fold split used here is **`folds_used.csv`**. It differs from `data/processed/folds.csv` in this repo,
because `StratifiedGroupKFold` splits differently in scikit-learn 1.7 (Kaggle) and 1.8+ (where the committed file was
built), even with the same seed. Both splits are valid: every person is in one fold, and all 18 gestures are in every
fold. Numbers are only comparable within one split, so any rebuild that must match these results uses
`python scripts/01_build_dataset.py --folds-from results/pass1/folds_used.csv`.

## Contents

| File | What |
| --- | --- |
| `table_main.csv` | All models: all-sensor / motion-only / combined score, 95% subject-bootstrap CI, fold 0 |
| `compare_main.csv` | Tuned CNN-BiGRU vs every other model: paired bootstrap, Holm-corrected |
| `table_tuning.csv` | Default vs tuned CNN-BiGRU and LightGBM (fold 0 never seen by tuning) |
| `table_hybrids.csv` | H1 blend, H2 stacker, H3 trees-on-probabilities |
| `table_sanity.csv` | Shuffled-label models (chance level for this metric is about 0.37-0.40) |
| `params/` | Tuned settings and every Optuna trial |
| `analysis_cnn_gru_tuned/` | Per-class F1, confusion matrices (+ PNG), per-person and per-posture scores |
| `blend_weights.json`, `deploy.json` | The deployed pair and blend weight used for the first submission |
| `pipeline/` | Data-build evidence: run summary, parity check, 23 quality checks |
| `logs/`, `run1_console.log` | Every job's log and the Kaggle console log |
| `folds_used.csv` | The exact person-to-fold split |

## Known issues (addressed in Pass 2)

- **XGBoost failed.** It tried to use CUDA while the tree jobs had the GPU hidden (`logs/xgb.log`). Fixed: XGBoost now gets a GPU explicitly.
- **H2 (stacker) is provisional.** It was fitted before the input-scaling fix (lbfgs did not converge).
- **The tuned network's learning rate (0.0038) and width (256) are at the top of the search range.** This is a limitation of the search.
- **Only 1 seed, and no ablations yet.** Both come in Pass 2.

## Headline findings

| Finding | Evidence |
| --- | --- |
| Best model: H1 blend | 0.821 [0.808, 0.834] |
| Tuned CNN-BiGRU beats every other model | 0.811, significant vs all after Holm |
| Sequence networks beat summary-feature models | Main model vs LightGBM +0.041, 5/5 folds |
| Tuning helps the network but not LightGBM | Network: +0.006, fold 0 0.833 → 0.848; LightGBM: +0.002 |
| Left-handed wearers score lower | 0.738 vs 0.817 for right-handers |
| The 2 upside-down wearers are among the 3 worst people | `analysis_cnn_gru_tuned/per_subject.csv` |

These two wearer findings motivate the N1 mirroring experiment.
