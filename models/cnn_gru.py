"""MAIN MODEL: residual SE-CNN blocks per sensor branch (local patterns) followed by a BiGRU (their order)."""
from models.gru import GRU


class CNNGRU(GRU):
    n_blocks = 3
