"""Name -> model. Networks take (imu_cols, prox_cols, L); tree-family builders take (seed, trees)."""
from models import lgbm, mlp, rf, xgb
from models.cnn import CNN
from models.cnn_gru import CNNGRU
from models.gru import GRU
from models.transformer import Transformer

NETS = {'cnn': CNN, 'gru': GRU, 'transformer': Transformer, 'cnn_gru': CNNGRU}
TREES = {'lgbm': lgbm.make, 'xgb': xgb.make, 'rf': rf.make, 'mlp': mlp.make}
