"""Training loop shared by every network (cnn, gru, transformer, cnn_gru). Each model's own file just calls cli(core).

The whole dataset lives on the GPU (no data loader). Epoch count is fixed in advance and the EMA weights from the last
epoch are reported, so nothing is ever chosen on the validation fold. Finished (seed, fold) runs are skipped, so a
crashed or interrupted session resumes where it stopped.

  python -m models.train.cnn_gru --variant full            # all sensors (also saves 'blank': same model, sensors off)
  python -m models.train.cnn_gru --variant imu             # motion-only model
  options: --folds 0 1  --seeds 0 1 2  --epochs 40  --phase 0 (no aux loss)  --aug 0 (no copies)  --save (keep weights)
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.common import blank, groups, load, oof_path, prep, save_oof, scaler, score, shape
from models.registry import NETS


def parse(core, argv=None):
    ap = argparse.ArgumentParser(description=f'train the {core} network')
    ap.add_argument('--name', default=core, help='run name (runs/<name>/); change it for ablations')
    ap.add_argument('--variant', default='full', choices=['full', 'imu'])
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--phase', type=float, default=0.3, help='weight of the auxiliary phase loss (0 = off)')
    ap.add_argument('--aug', type=int, default=1, help='use the pre-computed augmented copies if present')
    ap.add_argument('--drop-sub', nargs='*', default=[], help='zero columns whose name contains any of these')
    ap.add_argument('--save', action='store_true', help='keep model weights (needed only for the deployed model)')
    return ap.parse_args(argv)


def train(core, a):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp = dev == 'cuda'
    d, meta, seqs, feats, phase = load(a.data)
    cols, L = meta['feature_columns'], meta['fixed_length']
    imu, prox, flags = groups(cols)
    used = [] if a.variant == 'imu' else prox
    kinds = ['imu'] if a.variant == 'imu' else ['full', 'blank']
    todo = [(s, k) for s in a.seeds for k in a.folds if not all(oof_path(a.name, kd, s, k).exists() for kd in kinds)]
    if not todo:
        print(f'{a.name} {a.variant}: all requested runs already done', flush=True)
        return
    keep = torch.tensor([0. if any(s in c for s in a.drop_sub) else 1. for c in cols], device=dev)

    X, n, P = shape(feats, phase, seqs, L)
    T = lambda v: torch.tensor(v, device=dev)
    X32, N0, P0, Y, fold = T(X), T(n), T(P), T(seqs['gesture_id'].to_numpy()), T(seqs['fold'].to_numpy())
    pool, plen, psrc = X32.half(), N0, torch.arange(len(seqs), device=dev)
    if a.aug and (d / 'aug_x.npy').exists():
        pool = torch.cat([pool, T(np.load(d / 'aug_x.npy'))])
        plen = torch.cat([plen, T(np.load(d / 'aug_len.npy'))])
        psrc = torch.cat([psrc, T(np.load(d / 'aug_src.npy'))])
    isorig = torch.arange(len(pool), device=dev) < len(seqs)
    print(f'{a.name} {a.variant} on {dev}: {len(seqs)} recordings, pool {len(pool)}, {len(todo)} runs', flush=True)

    @torch.no_grad()
    def predict(model, idx, mean, std, off=False):
        model.eval()
        out = []
        for i in range(0, len(idx), 512):
            j = idx[i:i + 512]
            z, m = prep(X32[j], N0[j], mean, std)
            z = z * keep
            if off:
                z = blank(z, m, prox, flags)
            with torch.autocast(dev, torch.float16, enabled=amp):
                out.append(model(z, m)[0].float().softmax(1))
        return torch.cat(out).cpu().numpy()

    for seed, k in todo:
        torch.manual_seed(seed)
        mean, std = scaler(d, k, dev, cols)
        tr = (fold[psrc] != k).nonzero().squeeze(1)
        va = (fold == k).nonzero().squeeze(1)
        n_tr = int((fold != k).sum())
        steps = n_tr // a.bs
        model = NETS[core](imu, used, L).to(dev)
        ema = copy.deepcopy(model).requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.05)
        gs = torch.amp.GradScaler(enabled=amp)
        for _ in range(a.epochs):
            model.train()
            perm = tr[torch.randperm(len(tr), device=dev)[:n_tr]]     # an epoch = as many samples as original train rows
            for s in range(steps):
                idx = perm[s * a.bs:(s + 1) * a.bs]
                z, m = prep(pool[idx], plen[idx], mean, std)
                ph = torch.where(isorig[idx, None], P0[psrc[idx]], -100)   # phase labels exist for originals only
                with torch.autocast(dev, torch.float16, enabled=amp):
                    lg, pl = model(z * keep, m)
                    loss = F.cross_entropy(lg.float(), Y[psrc[idx]], label_smoothing=0.1)
                    if a.phase:
                        loss = loss + a.phase * F.cross_entropy(pl.float().transpose(1, 2), ph, ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                gs.scale(loss).backward()
                gs.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gs.step(opt)
                gs.update()
                sched.step()
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 0.005)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)
        y = Y[va].cpu().numpy()
        res = {'imu': predict(ema, va, mean, std)} if a.variant == 'imu' else \
              {'full': predict(ema, va, mean, std), 'blank': predict(ema, va, mean, std, off=True)}
        for kind, p in res.items():
            save_oof(a.name, kind, seed, k, p)
        if a.save:
            torch.save({'cfg': dict(core=core, imu=imu, prox=used, L=L), 'sd': ema.state_dict()},
                       Path('runs') / a.name / f'{a.variant}_s{seed}_f{k}.pt')
        print(f'{a.name} {a.variant} seed {seed} fold {k}: ' +
              '  '.join(f'{kd} {score(y, p):.4f}' for kd, p in res.items()), flush=True)


def cli(core):
    train(core, parse(core))
