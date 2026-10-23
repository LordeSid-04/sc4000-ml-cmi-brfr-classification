"""Shared building blocks and the base network. Each architecture (cnn, gru, transformer, cnn_gru) is a small
subclass in its own file that only chooses the conv depth and the sequence core.

  motion branch (+ proximity branch when thermopile/ToF are used) -> fuse -> core -> attention + mean pooling -> heads
  heads: 18-way gesture softmax (main) and a per-step transition/pause/gesture classifier (auxiliary)."""
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


class SeqNet(nn.Module):
    n_blocks = 3                                    # conv depth per branch (0 = 1x1 stem only)

    def __init__(s, imu, prox, L=128, d=192, drop=0.2):
        super().__init__()
        s.register_buffer('imu', torch.tensor(imu, dtype=torch.long))
        s.register_buffer('prox', torch.tensor(prox, dtype=torch.long))
        s.b_imu, c = branch(len(imu), s.n_blocks)
        s.b_prox, c2 = branch(len(prox), s.n_blocks) if prox else (None, 0)
        s.fuse = nn.Conv1d(c + c2, d, 1)
        out = s.build_core(d, L, drop)
        s.att = nn.Linear(out, 1)
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * out, 18))
        s.phase = nn.Linear(out, 3)

    def build_core(s, d, L, drop):                  # override: create the sequence layer, return its output size
        return d

    def core(s, h, m):                              # override: (B, L, d) -> (B, L, out)
        return h

    def forward(s, z, m):
        h = s.b_imu(z[..., s.imu].transpose(1, 2))
        if s.b_prox is not None:
            h = torch.cat([h, s.b_prox(z[..., s.prox].transpose(1, 2))], 1)
        mk = m[..., None]
        h = s.core(s.fuse(h).transpose(1, 2) * mk, m) * mk
        w = s.att(h).squeeze(-1).masked_fill(m == 0, -1e4).softmax(1)[..., None]
        pooled = torch.cat([(w * h).sum(1), h.sum(1) / mk.sum(1).clamp(min=1)], 1)
        return s.head(pooled), s.phase(h)
