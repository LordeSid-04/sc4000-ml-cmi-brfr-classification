"""Train one neural network (5 folds x seeds). Whole dataset lives on the GPU, so there is no data loader.

  python train_nn.py --name cnn_gru --variant full      # all sensors (also saves 'blank' = same model, sensors off)
  python train_nn.py --name cnn_gru --variant imu       # motion-only model
  --core cnn|gru|transformer|cnn_gru   --folds 0 1 ...   --seeds 0 1 2   --phase 0 (no aux loss)   --aug 0 (no copies)
Epoch count is fixed in advance and the EMA weights at the last epoch are reported (never picked on the validation fold)."""
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from models.common import blank, groups, load, prep, save_oof, scaler, score, shape
from models.nets import Net

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--core', default='cnn_gru', choices=['cnn', 'gru', 'transformer', 'cnn_gru'])
ap.add_argument('--variant', default='full', choices=['full', 'imu'])
ap.add_argument('--data', default='data/processed')
ap.add_argument('--folds', type=int, nargs='+', default=range(5))
ap.add_argument('--seeds', type=int, nargs='+', default=[0])
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--phase', type=float, default=0.3)
ap.add_argument('--aug', type=int, default=1)
ap.add_argument('--drop-sub', nargs='*', default=[], help='zero columns whose name contains any of these (ablations)')
a = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
amp = dev == 'cuda'
d, meta, seqs, feats, phase = load(a.data)
cols, L = meta['feature_columns'], meta['fixed_length']
imu, prox, flags = groups(cols)
used = [] if a.variant == 'imu' else prox
keep = torch.tensor([0. if any(s in c for s in a.drop_sub) else 1. for c in cols], device=dev)

X, n, P = shape(feats, phase, seqs, L)
T = lambda v, dt=None: torch.tensor(v, device=dev, dtype=dt)
X32, N0, P0, Y, fold = T(X), T(n), T(P), T(seqs['gesture_id'].to_numpy()), T(seqs['fold'].to_numpy())
pool, plen, psrc = X32.half(), N0, torch.arange(len(seqs), device=dev)
if a.aug and (d / 'aug_x.npy').exists():
    pool = torch.cat([pool, T(np.load(d / 'aug_x.npy'))])
    plen = torch.cat([plen, T(np.load(d / 'aug_len.npy'))])
    psrc = torch.cat([psrc, T(np.load(d / 'aug_src.npy'))])
isorig = torch.arange(len(pool), device=dev) < len(seqs)
print(f'{dev}: {len(seqs)} recordings, pool {len(pool)}, variant {a.variant}, core {a.core}', flush=True)


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


def run(k, seed):
    torch.manual_seed(seed)
    mean, std = scaler(d, k, dev, cols)
    tr = (fold[psrc] != k).nonzero().squeeze(1)
    va = (fold == k).nonzero().squeeze(1)
    n_tr = int((fold != k).sum())
    steps = n_tr // a.bs
    model = Net(imu, used, a.core, L).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.05)
    gs = torch.amp.GradScaler(enabled=amp)
    for ep in range(a.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev)[:n_tr]]       # one epoch = as many samples as original train rows
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
    kinds = {'imu': predict(ema, va, mean, std)} if a.variant == 'imu' else \
            {'full': predict(ema, va, mean, std), 'blank': predict(ema, va, mean, std, off=True)}
    for kind, p in kinds.items():
        save_oof(a.name, kind, seed, k, p)
    print(f'seed {seed} fold {k}: ' + '  '.join(f'{kd} {score(y, p):.4f}' for kd, p in kinds.items()), flush=True)
    torch.save({'cfg': dict(imu=imu, prox=used, core=a.core, L=L), 'sd': ema.state_dict()},
               f'runs/{a.name}/{a.variant}_s{seed}_f{k}.pt')


for seed in a.seeds:
    for k in a.folds:
        run(k, seed)
