"""One configurable network: motion branch (+ optional proximity branch) -> fuse -> sequence core -> attention pooling.
core: cnn | gru | transformer | cnn_gru (main model). The same stem, pooling and heads are used for every core."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Residual Conv1d block with squeeze-and-excitation."""
    def __init__(s, i, o, k=5, p=0.2):
        super().__init__()
        s.c1, s.b1 = nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o)
        s.c2, s.b2 = nn.Conv1d(o, o, k, padding=k // 2), nn.BatchNorm1d(o)
        s.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(o, o // 8, 1), nn.GELU(),
                             nn.Conv1d(o // 8, o, 1), nn.Sigmoid())
        s.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()
        s.drop = nn.Dropout(p)

    def forward(s, x):
        h = s.b2(s.c2(F.gelu(s.b1(s.c1(x)))))
        return F.gelu(s.skip(x) + s.drop(h * s.se(h)))


def branch(n_in, n_blocks):
    layers, c = [nn.Conv1d(n_in, 64, 1)], 64
    for o in [64, 128, 128][:n_blocks]:
        layers.append(Block(c, o))
        c = o
    return nn.Sequential(*layers), c


class Net(nn.Module):
    def __init__(s, imu, prox, core='cnn_gru', L=128, d=192, hid=128, drop=0.2):
        super().__init__()
        s.core = core
        s.register_buffer('imu', torch.tensor(imu, dtype=torch.long))
        s.register_buffer('prox', torch.tensor(prox, dtype=torch.long))
        nb = 3 if core in ('cnn', 'cnn_gru') else 0
        s.b_imu, c = branch(len(imu), nb)
        s.b_prox, c2 = branch(len(prox), nb) if prox else (None, 0)
        s.fuse = nn.Conv1d(c + c2, d, 1)
        if core in ('gru', 'cnn_gru'):
            s.seq, do = nn.GRU(d, hid, 2, batch_first=True, bidirectional=True, dropout=drop), 2 * hid
        elif core == 'transformer':
            s.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
            s.seq = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True),
                                          4, enable_nested_tensor=False)
            do = d
        else:
            s.seq, do = None, d
        s.att = nn.Linear(do, 1)
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * do, 18))
        s.phase = nn.Linear(do, 3)         # auxiliary: transition / pause / gesture for every time step

    def forward(s, z, m):
        h = s.b_imu(z[..., s.imu].transpose(1, 2))
        if s.b_prox is not None:
            h = torch.cat([h, s.b_prox(z[..., s.prox].transpose(1, 2))], 1)
        mk = m[..., None]
        h = s.fuse(h).transpose(1, 2) * mk
        if s.core == 'transformer':
            h = s.seq(h + s.pos, src_key_padding_mask=m == 0)
        elif s.seq is not None:
            h = s.seq(h)[0]
        h = h * mk
        w = s.att(h).squeeze(-1).masked_fill(m == 0, -1e4).softmax(1)[..., None]
        pooled = torch.cat([(w * h).sum(1), h.sum(1) / mk.sum(1).clamp(min=1)], 1)
        return s.head(pooled), s.phase(h)
