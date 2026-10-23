"""Kaggle inference notebook (internet off). Attach two private datasets:
  cmi-code   = this repo (preprocessing/, models/, data/processed/{meta.json,scalers/,tree_full.parquet})
  cmi-models = the unzipped cmi_models.zip (runs/: checkpoints, fitted trees, blend_weights.json at its root)
Every recording is routed by its sensors: no thermopile AND no ToF -> motion-only models, else all-sensor models.
Prediction = w * mean(NN folds x seeds) + (1 - w) * mean(tree folds x seeds); the pair and w come from
`python -m models.evaluate hybrid <nn> <gb>` (runs/deploy.json, runs/blend_weights.json)."""
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

CODE = os.environ.get('CMI_CODE', '/kaggle/input/cmi-code')
RUNS = os.environ.get('CMI_RUNS', '/kaggle/input/cmi-models')
sys.path.insert(0, CODE)
from models.registry import NETS as ARCH                               # noqa: E402
from preprocessing.constants import GESTURES                           # noqa: E402
from preprocessing.pipeline import preprocess                          # noqa: E402
from preprocessing.step5_shaping import Scaler, fixed_length, tree_summary   # noqa: E402

torch.set_num_threads(4)
meta = json.load(open(f'{CODE}/data/processed/meta.json'))
L = meta['fixed_length']
W = json.load(open(f'{RUNS}/blend_weights.json'))
NN, GB = json.load(open(f'{RUNS}/deploy.json')).values()      # written by `evaluate hybrid <nn> <gb>`
SC = {k: Scaler.load(f'{CODE}/data/processed/scalers/fold{k}.json') for k in range(5)}
TREE_COLS = [c for c in pq.read_schema(f'{CODE}/data/processed/tree_full.parquet').names if c != 'sequence_id']
GBS = [joblib.load(f) for f in sorted(glob.glob(f'{RUNS}/{GB}/model_s*_f*.joblib'))]
NETS = {}
for v in ('full', 'imu'):
    NETS[v] = []
    for f in sorted(glob.glob(f'{RUNS}/{NN}/{v}_s*_f*.pt')):
        ck = torch.load(f, map_location='cpu')
        cfg = dict(ck['cfg'])
        net = ARCH[cfg.pop('core')](**cfg)
        net.load_state_dict(ck['sd'])
        NETS[v].append((net.eval(), int(f.rsplit('_f', 1)[1][:-3])))
assert GBS and NETS['full'] and NETS['imu'], 'models not found'


@torch.no_grad()
def predict(sequence, demographics) -> str:
    out = preprocess(sequence, demographics)                 # the same code that built the training features
    f = out.features
    v = 'imu' if f['thm_missing'].iloc[0] and f['tof_missing'].iloc[0] else 'full'
    pn = []
    for net, k in NETS[v]:
        x, m = fixed_length(SC[k].transform(f), L)           # fold-k scaler, exactly as in training
        pn.append(net(torch.tensor(x)[None], torch.tensor(m)[None])[0].softmax(1)[0].numpy())
    row = pd.DataFrame([tree_summary(f)])[TREE_COLS]
    pg = np.mean([g.predict_proba(row)[0] for g in GBS], axis=0)
    return GESTURES[int(np.argmax(W[v] * np.mean(pn, axis=0) + (1 - W[v]) * pg))]


try:
    import kaggle_evaluation.cmi_inference_server as cmi
    server = cmi.CMIInferenceServer(predict)
except ImportError:                                           # running outside Kaggle (tests)
    server = None
if server is not None:
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        server.serve()
    else:
        base = '/kaggle/input/cmi-detect-behavior-with-sensor-data'
        server.run_local_gateway(data_paths=(f'{base}/test.csv', f'{base}/test_demographics.csv'))
