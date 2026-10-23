"""1D-CNN: three residual SE blocks per branch, no sequence layer (local motion patterns only)."""
from models.seqnet import SeqNet


class CNN(SeqNet):
    n_blocks = 3
