"""Fast CPU tests for the modelling code (no real data needed):  python -m pytest tests/test_models.py"""
import numpy as np
import pandas as pd
import pytest
import torch

from models.common import SID, blank, groups, prep, score_ids
from models.registry import NETS
from preprocessing.constants import GESTURES
from preprocessing.step4_labels import competition_score
from preprocessing.step5_shaping import Scaler, fixed_length

COLS = [f'acc_{i}' for i in range(6)] + [f'thm_{i}' for i in range(3)] + ['tof_1_mean', 'rot_missing', 'thm_missing',
                                                                       'tof_missing', 'time_frac']


def test_metric_matches_competition_score():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 18, 2000), rng.integers(0, 18, 2000)
    ref = competition_score([GESTURES[i] for i in y], [GESTURES[i] for i in p])[0]
    assert abs(score_ids(SID[y], SID[p]) - ref) < 1e-12


def test_groups_split_motion_and_proximity():
    imu, prox, flags = groups(COLS)
    assert [COLS[i] for i in flags] == ['thm_missing', 'tof_missing']
    assert set(imu) | set(prox) == set(range(len(COLS))) and not set(imu) & set(prox)
    assert 'rot_missing' in [COLS[i] for i in imu]


def test_gpu_scaling_equals_repo_scaler_and_fixed_length():
    rng = np.random.default_rng(1)
    L, frame = 16, pd.DataFrame(rng.normal(size=(40, len(COLS))), columns=COLS)
    frame.iloc[:, 6:9] = np.nan                                   # a missing sensor
    sc = Scaler().fit(frame)
    x, m = fixed_length(sc.transform(frame.iloc[:10]), L)         # a 10-row recording, left-padded
    raw = np.zeros((1, L, len(COLS)), np.float32)
    raw[0, L - 10:] = frame.iloc[:10].to_numpy()
    z, mk = prep(torch.tensor(raw), torch.tensor([10]), torch.tensor(sc.mean, dtype=torch.float32),
                 torch.tensor(sc.std, dtype=torch.float32))
    assert np.allclose(z[0].numpy(), x, atol=1e-5) and np.array_equal(mk[0].numpy(), m)


def test_blank_matches_missing_sensor_input():
    imu, prox, flags = groups(COLS)
    z = torch.randn(2, 8, len(COLS))
    m = torch.ones(2, 8)
    m[0, :3] = 0
    b = blank(z * m[..., None], m, prox, flags)
    assert (b[..., [i for i in prox if i not in flags]] == 0).all()
    assert torch.equal(b[..., flags][0, :3], torch.zeros(3, 2)) and (b[..., flags][0, 3:] == 1).all()


@pytest.mark.parametrize('core', list(NETS))
@pytest.mark.parametrize('motion_only', [False, True])
def test_every_network_forward_shapes_and_padding(core, motion_only):
    imu, prox, _ = groups(COLS)
    net = NETS[core](imu, [] if motion_only else prox, 16).eval()
    z = torch.randn(3, 16, len(COLS))
    m = torch.ones(3, 16)
    m[0, :8] = 0
    logits, ph = net(z * m[..., None], m)
    assert logits.shape == (3, 18) and ph.shape == (3, 16, 3) and torch.isfinite(logits).all()
