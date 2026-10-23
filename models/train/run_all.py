"""Run the complete training pipeline for every model, in parallel, then evaluate and package the results.

  python -m models.train.run_all [--data-zip PATH] [--seeds 0 1 2] [--workers 3] [--ablations]

 1. (optional) --data-zip: build data/processed from the competition data (python run_pipeline.py)
 2. pre-compute augmented copies of the training recordings (once, all CPU cores)
 3. network jobs (cnn_gru, cnn, gru, transformer x {all-sensor, motion-only} x 5 folds) run --workers at a time on the
    GPU(s) while the summary-feature models (lgbm, xgb, rf, mlp) train on the CPU at the same time
 4. evaluation: results tables with CIs, hybrids, blend weights
 5. runs/ (+ results/) are zipped: cmi_models.zip is what the submission notebook needs
Safe to re-run after a crash or timeout: finished (model, fold, seed) runs are skipped."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from models.common import complete

PY = sys.executable
NETS = ['cnn_gru', 'cnn', 'gru', 'transformer']            # main model first, so partial results are already useful
TREES = ['lgbm', 'xgb', 'rf', 'mlp']
DEPLOY = {'cnn_gru', 'lgbm'}                                # only these keep model files (the submission needs them)
ABLATIONS = {'abl_noaug': ['--aug', '0'], 'abl_nophase': ['--phase', '0'],
             'abl_notofreg': ['--drop-sub', '_r0', '_r1', '_r2']}       # all variants of the main model


def sh(cmd, log=None, env=None):
    t = time.time()
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        r = subprocess.run(cmd, env=env)
    return r.returncode, time.time() - t


def jobs(a):
    only = lambda n: not a.only or n in a.only
    seeds = [str(s) for s in a.seeds]
    nets = [(n, n, []) for n in NETS if only(n)] + \
           [(n, 'cnn_gru', x) for n, x in ABLATIONS.items() if a.ablations]
    gpu, cpu = [], []
    for name, core, extra in nets:
        for v in ('full', 'imu'):
            for k in range(5):
                cmd = [PY, '-m', f'models.train.{core}', '--name', name, '--variant', v, '--folds', str(k),
                       '--seeds', *seeds, '--epochs', str(a.epochs), *extra]
                gpu.append((f'{name}_{v}_f{k}', cmd + (['--save'] if name in DEPLOY else [])))
    for m in TREES:
        if only(m):
            cmd = [PY, '-m', f'models.train.{m}', '--seeds', *seeds, '--trees', str(a.trees)]
            cpu.append((m, cmd + (['--save'] if m in DEPLOY else [])))
    return gpu, cpu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-zip', help='competition zip/folder: builds data/processed first (skipped if already built)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=3, help='network jobs running at once (total, shared over GPUs)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--aug-copies', type=int, default=6)
    ap.add_argument('--only', nargs='*', help='train only these models (e.g. cnn_gru lgbm)')
    ap.add_argument('--ablations', action='store_true', help='also run the main-model ablations')
    ap.add_argument('--no-eval', action='store_true')
    a = ap.parse_args()
    d = Path('data/processed')

    if not (d / 'features.npz').exists() or not (d / 'canonical' / 'offsets.npy').exists():
        if not a.data_zip:
            sys.exit('data/processed/features.npz or canonical/ missing: rerun with --data-zip <competition zip or folder>')
        print('building the dataset (about 5 minutes) ...', flush=True)
        if sh([PY, 'run_pipeline.py', '--data', a.data_zip])[0]:
            sys.exit('run_pipeline.py failed')
    if a.aug_copies and not (d / 'aug_x.npy').exists():
        print('pre-computing augmented copies ...', flush=True)
        if sh([PY, '-m', 'models.train.augment', '--copies', str(a.aug_copies)])[0]:
            sys.exit('augmentation failed')

    import torch
    n_gpu = torch.cuda.device_count()
    gpu_jobs, cpu_jobs = jobs(a)
    print(f'{len(gpu_jobs)} network jobs ({a.workers} at a time, {n_gpu} GPU(s)) + {len(cpu_jobs)} tree models '
          f'on the CPU, seeds {a.seeds}', flush=True)
    t0, failed = time.time(), []

    def run(i, name, cmd, use_gpu):
        env = dict(os.environ)
        if use_gpu and n_gpu:
            env['CUDA_VISIBLE_DEVICES'] = str(i % n_gpu)
        elif not use_gpu:
            env['CUDA_VISIBLE_DEVICES'] = ''                  # trees: keep the GPU free for the networks (xgb falls back to CPU)
        return name, *sh(cmd, f'runs/logs/{name}.log', env)

    with ThreadPoolExecutor(a.workers) as gp, ThreadPoolExecutor(1) as cp:
        futs = [gp.submit(run, i, n, c, True) for i, (n, c) in enumerate(gpu_jobs)] + \
               [cp.submit(run, 0, n, c, False) for n, c in cpu_jobs]
        for j, f in enumerate(as_completed(futs), 1):
            name, rc, secs = f.result()
            print(f'[{j}/{len(futs)}] {name} {"ok" if rc == 0 else "FAILED (see runs/logs/" + name + ".log)"} '
                  f'{secs / 60:.1f} min   (elapsed {(time.time() - t0) / 60:.0f} min)', flush=True)
            if rc:
                failed.append(name)

    if not a.no_eval:
        names = [n for n in NETS + TREES if complete(n)]
        print('\nevaluating', names, flush=True)
        ev = lambda *args: sh([PY, '-m', 'models.evaluate', *args], 'results/last_eval.log')
        outs = []
        if names:
            outs.append(('table', ['table', *names, *(['cnn_gru:blank'] if 'cnn_gru' in names else [])]))
        if {'cnn_gru', 'lgbm'} <= set(names):
            outs.append(('hybrid', ['hybrid', 'cnn_gru', 'lgbm']))
        if a.ablations and all(complete(n) for n in ['cnn_gru', *ABLATIONS]):
            outs.append(('ablations', ['table', 'cnn_gru', *ABLATIONS]))
        for title, args in outs:
            print(f'--- {title}', flush=True)
            r = subprocess.run([PY, '-m', 'models.evaluate', *args])
            if r.returncode:
                failed.append(f'evaluate {title}')

    shutil.make_archive('cmi_models', 'zip', 'runs')              # checkpoints + OOF + blend_weights.json
    if Path('results').exists():
        shutil.make_archive('cmi_results', 'zip', 'results')
    print(f'\nfinished in {(time.time() - t0) / 60:.0f} min. Saved: cmi_models.zip, cmi_results.zip, runs/, results/')
    if failed:
        print('FAILED:', ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
