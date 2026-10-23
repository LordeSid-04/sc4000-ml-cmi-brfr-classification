"""Transformer encoder: 1x1 stem, learned positions, 4 pre-norm layers with padding mask (global attention)."""
import torch
import torch.nn as nn

from models.seqnet import SeqNet


class Transformer(SeqNet):
    n_blocks = 0

    def build_core(s, d, L, drop):
        s.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        return d

    def core(s, h, m):
        return s.enc(h + s.pos, src_key_padding_mask=m == 0)
