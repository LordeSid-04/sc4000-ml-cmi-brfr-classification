"""Training loop shared by every network (cnn, gru, transformer, cnn_gru). Each model's own file just calls cli(core).

The whole dataset lives on the GPU (no data loader). The epoch count is fixed in advance and the EMA weights from the
last epoch are reported, so nothing is ever chosen on a validation fold. Finished (seed, fold) runs are skipped, so a
crashed or interrupted session resumes where it stopped.

  python -m models.train.cnn_gru --variant full            # all sensors (also saves 'blank': same model, sensors off)
  python -m models.train.cnn_gru --variant imu             # motion-only model
  options: --folds 0 1  --seeds 0 1 2  --params results/02_hyperparameter_tuning/params/cnn_gru.json (tuned settings)  --save (keep weights)
           --phase 0 (no aux loss)  --aug 0 (no copies)  --drop-sub _r0 (zero matching columns)  --shuffle-labels (sanity)
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.common import blank, groups, load, oof_path, prep, save_oof, scaler, score, shape
from models.registry import NETS

HP = dict(lr=2e-3, epochs=40, bs=128, phase=0.3, drop=0.2, d=192)      # defaults; tuning overrides these


def parse(core, argv=None):
    ap = argparse.ArgumentParser(description=f'train the {core} network')
    ap.add_argument('--name', default=core, help='run name (runs/<name>/); change it for ablations')
    ap.add_argument('--variant', default='full', choices=['full', 'imu'])
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--params', help='JSON written by models.train.tune (overrides the defaults below)')
    for k, v in HP.items():
        ap.add_argument(f'--{k}', type=type(v), default=None, help=f'default {v}')
    ap.add_argument('--aug', type=int, default=1, help='use the pre-computed augmented copies if present')
    ap.add_argument('--drop-sub', nargs='*', default=[], help='zero columns whose name contains any of these')
    ap.add_argument('--shuffle-labels', action='store_true', help='sanity check: train on permuted labels')
    ap.add_argument('--save', action='store_true', help='keep model weights (needed for every deployed model)')
    a = ap.parse_args(argv)
    a.hp = {**HP, **(json.load(open(a.params))['params'] if a.params else {}),
            **{k: getattr(a, k) for k in HP if getattr(a, k) is not None}}
    return a


class Data:
    """Everything a training run needs, loaded once onto the device."""
    def __init__(s, data='data/processed', aug=True, drop_sub=()):
        s.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        s.amp = s.dev == 'cuda'
        s.d, meta, s.seqs, s.feats, phase = load(data)
        s.cols, s.L = meta['feature_columns'], meta['fixed_length']
        s.mirror = bool(meta['pipeline_config'].get('mirror_left', False))    # left-handers mirrored in this data?
        s.imu, s.prox, s.flags = groups(s.cols)
        s.keep = torch.tensor([0. if any(t in c for t in drop_sub) else 1. for c in s.cols], device=s.dev)
        X, n, P = shape(s.feats, phase, s.seqs, s.L)
        T = lambda v: torch.tensor(v, device=s.dev)
        s.X, s.n, s.P = T(X), T(n), T(P)
        s.Y, s.fold = T(s.seqs['gesture_id'].to_numpy()), T(s.seqs['fold'].to_numpy())
        s.pool, s.plen, s.src = s.X.half(), s.n, torch.arange(len(s.seqs), device=s.dev)
        if aug and (s.d / 'aug_x.npy').exists():
            s.pool = torch.cat([s.pool, T(np.load(s.d / 'aug_x.npy'))])
            s.plen = torch.cat([s.plen, T(np.load(s.d / 'aug_len.npy'))])
            s.src = torch.cat([s.src, T(np.load(s.d / 'aug_src.npy'))])
        s.orig = torch.arange(len(s.pool), device=s.dev) < len(s.seqs)

    @torch.no_grad()
    def predict(s, model, idx, mean, std, off=False):
        model.eval()
        out = []
        for i in range(0, len(idx), 512):
            j = idx[i:i + 512]
            z, m = prep(s.X[j], s.n[j], mean, std)
            z = z * s.keep
            if off:
                z = blank(z, m, s.prox, s.flags)
            with torch.autocast(s.dev, torch.float16, enabled=s.amp):
                out.append(model(z, m)[0].float().softmax(1))
        return torch.cat(out).cpu().numpy()

    def fit(s, core, train_mask, val_idx, mean, std, seed, hp, variant='full', shuffle=False):
        """Train on the recordings in train_mask (+ their augmented copies), predict val_idx.
        Returns (EMA model, {kind: probabilities})."""
        torch.manual_seed(seed)
        mean, std = (torch.as_tensor(v, dtype=torch.float32, device=s.dev) for v in (mean, std))
        Y = s.Y[torch.randperm(len(s.Y), device=s.dev)] if shuffle else s.Y
        tr = train_mask[s.src].nonzero().squeeze(1)
        n_tr, bs = int(train_mask.sum()), hp['bs']
        steps = n_tr // bs
        model = NETS[core](s.imu, [] if variant == 'imu' else s.prox, s.L, hp['d'], hp['drop']).to(s.dev)
        ema = copy.deepcopy(model).requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, hp['lr'], total_steps=hp['epochs'] * steps, pct_start=0.05)
        gs = torch.amp.GradScaler(enabled=s.amp)
        for _ in range(hp['epochs']):
            model.train()
            perm = tr[torch.randperm(len(tr), device=s.dev)[:n_tr]]   # an epoch = as many samples as original train rows
            for i in range(steps):
                idx = perm[i * bs:(i + 1) * bs]
                z, m = prep(s.pool[idx], s.plen[idx], mean, std)
                ph = torch.where(s.orig[idx, None], s.P[s.src[idx]], -100)   # phase labels exist for originals only
                with torch.autocast(s.dev, torch.float16, enabled=s.amp):
                    lg, pl = model(z * s.keep, m)
                    loss = F.cross_entropy(lg.float(), Y[s.src[idx]], label_smoothing=0.1)
                    if hp['phase']:
                        loss = loss + hp['phase'] * F.cross_entropy(pl.float().transpose(1, 2), ph, ignore_index=-100)
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
        if variant == 'imu':
            return ema, {'imu': s.predict(ema, val_idx, mean, std)}
        return ema, {'full': s.predict(ema, val_idx, mean, std), 'blank': s.predict(ema, val_idx, mean, std, off=True)}


def train(core, a):
    kinds = ['imu'] if a.variant == 'imu' else ['full', 'blank']
    done = lambda s, k: all(oof_path(a.name, kd, s, k).exists() for kd in kinds) and \
        (not a.save or (Path('runs') / a.name / f'{a.variant}_s{s}_f{k}.pt').exists())    # --save also needs the weights
    todo = [(s, k) for s in a.seeds for k in a.folds if not done(s, k)]
    if not todo:
        print(f'{a.name} {a.variant}: all requested runs already done', flush=True)
        return
    D = Data(a.data, a.aug, a.drop_sub)
    print(f'{a.name} {a.variant} on {D.dev}: {len(D.seqs)} recordings, pool {len(D.pool)}, {len(todo)} runs, hp {a.hp}',
          flush=True)
    for seed, k in todo:
        mean, std = scaler(D.d, k, D.dev, D.cols)                      # stored fold scaler = what inference uses
        va = (D.fold == k).nonzero().squeeze(1)
        model, res = D.fit(core, D.fold != k, va, mean, std, seed, a.hp, a.variant, a.shuffle_labels)
        for kind, p in res.items():
            save_oof(a.name, kind, seed, k, p)
        if a.save:
            torch.save({'cfg': dict(core=core, imu=D.imu, prox=[] if a.variant == 'imu' else D.prox, L=D.L,
                                    d=a.hp['d'], drop=a.hp['drop']), 'sd': model.state_dict(), 'mirror_left': D.mirror},
                       Path('runs') / a.name / f'{a.variant}_s{seed}_f{k}.pt')
        y = D.Y[va].cpu().numpy()
        print(f'{a.name} {a.variant} seed {seed} fold {k}: ' +
              '  '.join(f'{kd} {score(y, p):.4f}' for kd, p in res.items()), flush=True)


def cli(core):
    train(core, parse(core))
