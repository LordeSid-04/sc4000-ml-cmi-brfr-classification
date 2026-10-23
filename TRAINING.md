# Training the models

```
models/
  common.py            data shaping, GPU scaling, competition metric, OOF files, subject bootstrap
  seqnet.py            shared blocks + base network          cnn.py  gru.py  transformer.py  cnn_gru.py (main)
  lgbm.py xgb.py rf.py mlp.py   summary-feature models        registry.py  name -> model
  hybrids.py           blend / stacker / trees-on-probabilities     evaluate.py  tables, CIs, hybrids
  train/
    nn_trainer.py      loop shared by the networks             tree_trainer.py  loop shared by the tree-family models
    cnn.py gru.py transformer.py cnn_gru.py lgbm.py xgb.py rf.py mlp.py   one entry point per model
    augment.py         pre-computes augmented copies           run_all.py       the whole pipeline, in parallel
submission/submit.py   the Kaggle inference notebook
```

**Everything at once** (from the repo root, after `pip install -r requirements.txt`):

```
python -m models.train.run_all --data-zip <competition zip or folder> --ablations
```

**One model** (e.g. while debugging): `python -m models.train.cnn_gru --variant full --folds 0 --seeds 0 --epochs 5`
(`--variant imu` for the motion-only model; trees: `python -m models.train.lgbm`).

**Results** are plain files and survive interruptions (finished runs are skipped on re-run):

| Path | Content |
| --- | --- |
| `runs/<model>/<kind>_s<seed>_f<fold>.npy` | out-of-fold probabilities (kind: `full`, `imu`, `blank`) |
| `runs/cnn_gru/*.pt`, `runs/lgbm/*.joblib`, `runs/blend_weights.json` | the deployed models (only these keep weights) |
| `runs/logs/` | one log per job |
| `results/*.csv`, `results/last_eval.log` | tables with 95% subject-bootstrap CIs |
| `cmi_models.zip`, `cmi_results.zip` | the two folders zipped for download / upload as a Kaggle Dataset |

After training, `python -m models.evaluate table|diff|hybrid ...` (see its docstring) produces every report number.
Tests: `python -m pytest tests/test_models.py`.
