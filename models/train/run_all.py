"""Run the complete training pipeline for every model, in parallel, then evaluate and package the results.

  python -m models.train.run_all [--data-zip PATH] [--tune] [--ablations] [--sanity] [--seeds 0 1 2] [--workers 3]

 1. (optional) --data-zip: build data/processed from the competition data (python run_pipeline.py)
 2. pre-compute augmented copies of the training recordings (once, all CPU cores)
 3. (--tune) nested tuning of the deployed pair: cnn_gru on the GPU and lgbm on the CPU at the same time
 4. network jobs (cnn_gru, cnn, gru, transformer [+ cnn_gru_tuned, ablations, sanity] x {all-sensor, motion-only} x
    5 folds) run --workers at a time on the GPU(s) while the summary-feature models train on the CPU
 5. evaluation: results table with CIs, Holm-corrected comparisons, hybrids + blend weights, error analysis
 6. runs/ and results/ are zipped: cmi_models.zip is what the submission notebook needs
Safe to re-run after a crash or timeout: finished (model, fold, seed) runs and finished tuning are skipped."""
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
ABLATIONS = {'abl_noaug': ['--aug', '0'], 'abl_nophase': ['--phase', '0'],
             'abl_notofreg': ['--drop-sub', '_r0', '_r1', '_r2']}         # each removes one part of the main model
PARAMS = Path('results/params')


def sh(cmd, log=None, env=None):
    t = time.time()
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    else:
        r = subprocess.run(cmd, env=env)
    return r.returncode, time.time() - t


def plan(a):
    """-> (network jobs, tree jobs, deployed (nn, gb) names). A job = (name, command)."""
    only = lambda n: not a.only or n in a.only
    s = ['--seeds', *map(str, a.seeds)]
    nn, gb = ('cnn_gru_tuned', 'lgbm_tuned') if a.tune else ('cnn_gru', 'lgbm')
    nets = [(n, n, []) for n in NETS if only(n)]
    trees = [(m, m, []) for m in TREES if only(m)]
    if a.tune:
        nets.insert(0, ('cnn_gru_tuned', 'cnn_gru', ['--params', str(PARAMS / 'cnn_gru.json')]))
        trees.insert(0, ('lgbm_tuned', 'lgbm', ['--params', str(PARAMS / 'lgbm.json')]))
    if a.ablations:
        nets += [(n, 'cnn_gru', x) for n, x in ABLATIONS.items()]
    if a.sanity:      # permuted labels: scores must collapse to chance, otherwise something leaks
        nets.append(('sanity_shuffled_cnn_gru', 'cnn_gru', ['--shuffle-labels', '--seeds', '0']))
        trees.append(('sanity_shuffled_lgbm', 'lgbm', ['--shuffle-labels', '--seeds', '0']))
    gpu = [(f'{name}_{v}_f{k}', [PY, '-m', f'models.train.{core}', '--name', name, '--variant', v, '--folds', str(k),
                                 *s, *(['--epochs', str(a.epochs)] if a.epochs else []), *extra,
                                 *(['--save'] if name == nn else [])])
           for name, core, extra in nets for v in ('full', 'imu') for k in range(5)]
    cpu = [(name, [PY, '-m', f'models.train.{core}', '--name', name, *s, '--trees', str(a.trees), *extra,
                   *(['--save'] if name == gb else [])]) for name, core, extra in trees]
    return gpu, cpu, (nn, gb)


def bundle(d):
    """Ship the exact metadata, scalers and tree-column order this run trained with, next to the models,
    so the submission can never pair the models with a differently built copy."""
    import json
    import pyarrow.parquet as pq
    out = Path('runs/processed')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(d / 'meta.json', out / 'meta.json')
    shutil.copytree(d / 'scalers', out / 'scalers', dirs_exist_ok=True)
    cols = [c for c in pq.read_schema(d / 'tree_full.parquet').names if c != 'sequence_id']
    (out / 'tree_columns.json').write_text(json.dumps(cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-zip', help='competition zip/folder: builds data/processed first (skipped if already built)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=3, help='network jobs running at once (total, shared over GPUs)')
    ap.add_argument('--epochs', type=int, default=None, help='override the epoch count of every network (testing)')
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--aug-copies', type=int, default=6)
    ap.add_argument('--only', nargs='*', help='train only these base models (e.g. cnn_gru lgbm)')
    ap.add_argument('--tune', action='store_true', help='nested tuning of cnn_gru + lgbm, then train and deploy them')
    ap.add_argument('--trials', type=int, nargs=2, default=[12, 20], metavar=('NN', 'TREES'))
    ap.add_argument('--ablations', action='store_true', help='also run the main-model ablations')
    ap.add_argument('--sanity', action='store_true', help='also run the shuffled-label leakage check')
    ap.add_argument('--no-eval', action='store_true')
    a = ap.parse_args()
    d = Path('data/processed')
    t0, failed = time.time(), []

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

    def env(i, use_gpu):
        """Spread network jobs over the GPUs; hide the GPU from tree jobs so the networks keep it."""
        e = dict(os.environ)
        if not use_gpu:
            e['CUDA_VISIBLE_DEVICES'] = ''
        elif n_gpu:
            e['CUDA_VISIBLE_DEVICES'] = str(i % n_gpu)
        return e

    if a.tune:
        todo = [(m, n, g) for m, n, g in (('cnn_gru', a.trials[0], True), ('lgbm', a.trials[1], False))
                if not (PARAMS / f'{m}.json').exists()]
        if todo:
            print('tuning ' + ', '.join(f'{m} ({n} trials)' for m, n, _ in todo) + ' in parallel ...', flush=True)
            with ThreadPoolExecutor(2) as ex:
                cmd = lambda m, n: [PY, '-m', 'models.train.tune', '--model', m, '--trials', str(n),
                                    *(['--epochs', str(a.epochs)] if a.epochs else [])]
                res = list(ex.map(lambda j: (j[0], *sh(cmd(j[0], j[1]), f'runs/logs/tune_{j[0]}.log', env(0, j[2]))),
                                  todo))
            for m, rc, secs in res:
                print(f'tune {m}: {"ok" if rc == 0 else "FAILED (runs/logs/tune_" + m + ".log)"} {secs / 60:.0f} min')
                if rc:
                    sys.exit('tuning failed')

    gpu_jobs, cpu_jobs, (nn, gb) = plan(a)
    print(f'{len(gpu_jobs)} network jobs ({a.workers} at a time, {n_gpu} GPU(s)) + {len(cpu_jobs)} tree models '
          f'on the CPU, seeds {a.seeds}; deploying {nn} + {gb}', flush=True)

    def run(i, name, cmd, use_gpu):
        return name, *sh(cmd, f'runs/logs/{name}.log', env(i, use_gpu))

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
        def ev(title, *args):
            print(f'\n--- {title}', flush=True)
            if subprocess.run([PY, '-m', 'models.evaluate', *args]).returncode:
                failed.append(title)

        base = [n for n in [nn, gb, *NETS, *TREES] if complete(n)]
        base = list(dict.fromkeys(base))
        if base:
            ev('model comparison', 'table', *base, *([f'{nn}:blank'] if complete(nn, ('full', 'blank')) else []))
        if nn in base and len(base) > 1:
            ev(f'is {nn} better than each other model? (Holm-corrected)', 'compare', nn, *[b for b in base if b != nn])
        if a.tune and complete('cnn_gru') and complete('lgbm'):
            ev('does tuning help? (fold0 = untouched by tuning)', 'table', 'cnn_gru', nn, 'lgbm', gb)
        if {nn, gb} <= set(base):
            ev('hybrids (deployed pair)', 'hybrid', nn, gb)
            ev('error analysis of the main model', 'analysis', nn)
        abl = [n for n in ABLATIONS if complete(n)]
        if abl and complete('cnn_gru'):
            ev('ablations: does each component help? (Holm-corrected)', 'compare', 'cnn_gru', *abl)
        sanity = [n for n in ('sanity_shuffled_cnn_gru', 'sanity_shuffled_lgbm') if complete(n)]
        if sanity:
            ev('sanity check: shuffled labels must score near chance', 'table', *sanity)

    bundle(d)
    shutil.make_archive('cmi_models', 'zip', 'runs')              # checkpoints + OOF + blend_weights.json + deploy.json
    if Path('results').exists():
        shutil.make_archive('cmi_results', 'zip', 'results')
    print(f'\nfinished in {(time.time() - t0) / 60:.0f} min. Saved: cmi_models.zip, cmi_results.zip, runs/, results/')
    if failed:
        print('FAILED:', ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
