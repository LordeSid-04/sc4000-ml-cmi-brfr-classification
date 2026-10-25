# SC4000 · CMI – Detect Behavior with Sensor Data

This is our course project for the Kaggle competition
[CMI – Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).
The task is to recognise 18 hand gestures, 8 of them body-focused repetitive behaviours (BFRBs), from a wrist
device with motion (IMU), thermopile and time-of-flight sensors. About half the hidden test set has motion sensors
only. The score is ½ binary F1 (BFRB or not) + ½ macro F1 (which BFRB).

> Keep this repository **private** (course rule: do not release code publicly).

## Results (5-fold cross-validation by person, 3 seeds)

| Model | Combined score [95% CI] |
| --- | --- |
| **N1: tuned CNN-BiGRU with left-handed wearers mirrored** | **0.838** [0.824, 0.850] |
| H1: tuned CNN-BiGRU + LightGBM blend (current submission) | 0.828 [0.814, 0.841] |
| Tuned CNN-BiGRU | 0.825 [0.812, 0.838] |
| LightGBM on 795 summary features | 0.773 [0.760, 0.785] |

The full tables, statistics, error analysis and 11 figures are in **[results/](results/README.md)**.

## Repository

```
preprocessing/        cleaning, orientation, 117 per-step features, labels, folds   -> docs/PREPROCESSING.md
models/               one file per model + shared code
  seqnet.py             shared network blocks; cnn.py gru.py transformer.py cnn_gru.py (main)
  lgbm.py xgb.py rf.py mlp.py   summary-feature models
  hybrids.py            blend / stacker / trees-on-network-outputs
  evaluate.py           tables, bootstrap CIs, Holm-corrected comparisons, error analysis
  report_figures.py     the figures in results/figures/
  train/                one entry point per model, shared trainers, tuning, run_all.py (whole pipeline)
submission/           Kaggle inference: submit.py and submission.ipynb
scripts/              dataset build, parity check, quality checks (run by run_pipeline.py)
tests/                preprocessing + model tests
results/              published results, figures, logs, the fold split
```

## Reproduce

Everything runs on a free Kaggle GPU notebook. **[TRAINING.md](TRAINING.md)** has the exact steps.

```
python run_pipeline.py --data <competition data>          # tests, build, parity + quality checks (~5 min)
python -m models.train.run_all --data-zip <competition data> --tune --ablations --sanity --n1
python -m models.report_figures                           # figures from results/*.csv
```

## How results are kept honest

- **Folds:** 5 folds split by person (the test set has new people), pinned in `results/folds_used.csv`.
- **No selection on reported folds:** the epoch count is fixed in advance, the EMA weights from the last epoch are reported, and nothing is chosen on a reported fold.
- **Nested tuning:** tuning uses only folds 1-4. Fold 0 is untouched and reported separately.
- **Hybrids** are fitted on other folds' out-of-fold predictions only.
- **Uncertainty:** 3 seeds; 95% confidence intervals from resampling people; Holm correction across comparisons.
- **Checks:** a shuffled-label control, parity between the training and Kaggle code paths, and 23 data-quality checks.

## Acknowledgements

All code in this repository was written by the team. We used no code from Kaggle participants. Ideas we adopted
from public competition write-ups:
- **Left-hand normalisation and gravity removal:** 7th-place talk
  ([Speaker Deck](https://speakerdeck.com/rist/cmi-detect-behavior-with-sensor-data)).
- **Late sensor fusion, sensor-availability-specific models, and CNN-RNN hybrids:** competition summary
  ([Medium](https://medium.com/@nlztrk/my-competition-summary-cmi-detect-behavior-with-sensor-data-8215c710e2f3)).
- **Augmentation choices:** top-solution notes
  ([Zenn](https://zenn.dev/ottantachinque/articles/2025-09-14_cmi-detect-behavior-with-sensor-data)).

Libraries: PyTorch, LightGBM, XGBoost, scikit-learn, Optuna, NumPy, pandas, Polars, Matplotlib.
