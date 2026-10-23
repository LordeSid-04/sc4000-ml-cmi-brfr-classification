"""BiGRU: 1x1 stem then a 2-layer bidirectional GRU, no convolutions (order of the motion only)."""
import torch.nn as nn

from models.seqnet import SeqNet


class GRU(SeqNet):
    n_blocks = 0

    def build_core(s, d, L, drop):
        s.rnn = nn.GRU(d, 128, 2, batch_first=True, bidirectional=True, dropout=drop)
        return 256

    def core(s, h, m):
        return s.rnn(h)[0]
