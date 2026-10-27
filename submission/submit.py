"""Kaggle inference notebook (internet off). Inputs:
  cmi-code   = this repo (the code: preprocessing/, models/)
  cmi-models = runs/ from training (or the training notebook itself): checkpoints, deploy.json, processed/ and
               processed_mirror/ (the exact meta.json and scalers the models were trained with)
Both folders are found automatically under /kaggle/input (set CMI_CODE / CMI_RUNS to override).

What is submitted is read from runs/deploy.json (written by `python -m models.evaluate deploy` or `hybrid`):
  nn       networks; each network's checkpoints (folds x seeds) are averaged, then the networks are averaged
  gb       optional tree model (null = none), blended as  w * networks + (1 - w) * trees,  w from `weights`
  balance  per-person balancing strength (models/balance.py; 0 = off), quota = gestures per person in training
Every recording is routed by its sensors: no thermopile AND no ToF -> motion-only models, else all-sensor models.
A checkpoint trained on left-hand-mirrored data gets mirrored inputs (handedness comes with each test recording).

Override in a cell BEFORE running this file (nothing has to be re-uploaded):
  CMI_NN=n1_mirror,n1_gru  CMI_GB=lgbm_tuned (or none)  CMI_W=1 | 0.6 | 0.6,0.8 (all-sensor,motion-only)  CMI_BAL=0"""
import glob
import json
import os
import sys
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import torch


def find(name):
    """Folder holding `name` anywhere under /kaggle/input (Kaggle's mount paths vary by account and notebook)."""
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True), key=len)
    return os.path.dirname(hits[0]) if hits else None


CODE = os.environ.get('CMI_CODE') or os.path.dirname(find('submission/submit.py') or '/kaggle/input/cmi-code/x')
RUNS = os.environ.get('CMI_RUNS') or find('deploy.json') or '/kaggle/input/cmi-models'
sys.path.insert(0, CODE)
from models.balance import adjust                                      # noqa: E402
from models.registry import NETS as ARCH                               # noqa: E402
from preprocessing.constants import GESTURES                           # noqa: E402
from preprocessing.pipeline import PipelineConfig, preprocess          # noqa: E402
from preprocessing.step5_shaping import Scaler, fixed_length, tree_summary   # noqa: E402

torch.set_num_threads(4)
DEP = json.load(open(f'{RUNS}/deploy.json'))
if 'weights' not in DEP:                                      # older runs: {"nn": name, "gb": name} + blend_weights.json
    DEP = dict(nn=[DEP['nn']], gb=DEP['gb'], weights=json.load(open(f'{RUNS}/blend_weights.json')), balance=0)
NN = os.environ.get('CMI_NN', ','.join(DEP['nn'])).split(',')
GB = os.environ.get('CMI_GB', DEP['gb'] or 'none')
W = DEP['weights'] if GB != 'none' else {'full': 1.0, 'imu': 1.0}
if os.environ.get('CMI_W'):
    w = [float(x) for x in os.environ['CMI_W'].split(',')]
    W = {'full': w[0], 'imu': w[-1]}
BAL = float(os.environ.get('CMI_BAL', DEP.get('balance', 0)))
QUOTA = np.array(DEP.get('quota', [0.0] * 18))
USE_NN, USE_GB = max(W.values()) > 0, min(W.values()) < 1


def data_dir(mirror):
    d = f'{RUNS}/processed_mirror' if mirror else f'{RUNS}/processed'
    return d if os.path.isdir(d) else f'{CODE}/data/processed' + ('_mirror' if mirror else '')


META = {m: json.load(open(f'{data_dir(m)}/meta.json')) for m in (False, True) if os.path.isdir(data_dir(m))}
SC = {m: {k: Scaler.load(f'{data_dir(m)}/scalers/fold{k}.json') for k in range(5)} for m in META}
NETS = {(n, v): [] for n in NN for v in ('full', 'imu')} if USE_NN else {}
for (n, v), lst in NETS.items():
    for f in sorted(glob.glob(f'{RUNS}/{n}/{v}_s*_f*.pt')):
        ck = torch.load(f, map_location='cpu')
        cfg = dict(ck['cfg'])
        net = ARCH[cfg.pop('core')](**cfg)
        net.load_state_dict(ck['sd'])
        lst.append((net.eval(), int(f.rsplit('_f', 1)[1][:-3]), bool(ck.get('mirror_left', False))))
    assert lst, f'no {v} checkpoints in {RUNS}/{n}/ (only runs trained with --save have them)'
GBS = [joblib.load(f) for f in sorted(glob.glob(f'{RUNS}/{GB}/model_s*_f*.joblib'))] if USE_GB else []
assert not USE_GB or GBS, f'no fitted models in {RUNS}/{GB}/ (only runs trained with --save have them)'
assert not USE_GB or GB != 'none', 'CMI_W < 1 needs a tree model (CMI_GB)'
TREE_COLS = None
if USE_GB:                                                    # the column order the tree models were fitted with
    p = data_dir(False)
    TREE_COLS = json.load(open(f'{p}/tree_columns.json')) if os.path.exists(f'{p}/tree_columns.json') else \
        [c for c in pd.read_parquet(f'{p}/tree_full.parquet').columns if c != 'sequence_id']
COUNTS = defaultdict(lambda: np.zeros(18))                    # per person: gestures predicted so far
print(f'SUBMITTING: networks {NN if USE_NN else "-"} '
      f'({", ".join(str(len(c)) for c in NETS.values())} checkpoints per network x sensor setting), '
      f'trees {GB if USE_GB else "-"} ({len(GBS)} models), weight on networks {W}, balancing strength {BAL}, '
      f'mirrored inputs for {sorted({n for (n, _), c in NETS.items() if c[0][2]})}', flush=True)


@torch.no_grad()
def predict(sequence, demographics) -> str:
    feats = {}

    def features(mirror):                                     # the same code that built the training features
        if mirror not in feats:
            feats[mirror] = preprocess(sequence, demographics, PipelineConfig(mirror_left=mirror)).features
        return feats[mirror]

    f = features(False)
    v = 'imu' if f['thm_missing'].iloc[0] and f['tof_missing'].iloc[0] else 'full'
    p = 0.0
    if W[v] > 0:
        per_net = []
        for n in NN:
            pn = []
            for net, k, mirror in NETS[(n, v)]:
                x, m = fixed_length(SC[mirror][k].transform(features(mirror)), META[mirror]['fixed_length'])
                pn.append(net(torch.tensor(x)[None], torch.tensor(m)[None])[0].softmax(1)[0].numpy())
            per_net.append(np.mean(pn, axis=0))
        p = p + W[v] * np.mean(per_net, axis=0)
    if W[v] < 1:
        row = pd.DataFrame([tree_summary(f)])[TREE_COLS]
        p = p + (1 - W[v]) * np.mean([g.predict_proba(row)[0] for g in GBS], axis=0)
    person = None                                             # the wearer, for per-person balancing
    for t in (sequence, demographics):
        t = t.to_pandas() if hasattr(t, 'to_pandas') else t
        if person is None and t is not None and 'subject' in t and len(t):
            person = str(t['subject'].iloc[0])
    if person is not None:
        p = adjust(p, COUNTS[person], QUOTA, BAL)
    g = int(np.argmax(p))
    if person is not None:
        COUNTS[person][g] += 1
    return GESTURES[g]


try:
    import kaggle_evaluation.cmi_inference_server as cmi
    server = cmi.CMIInferenceServer(predict)
except ImportError:                                           # running outside Kaggle (tests)
    server = None
if server is not None:
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        server.serve()
    else:
        base = find('test_demographics.csv')
        server.run_local_gateway(data_paths=(f'{base}/test.csv', f'{base}/test_demographics.csv'))
