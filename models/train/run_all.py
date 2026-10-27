"""Run the complete training pipeline for every model, in parallel, then evaluate and package the results.

  python -m models.train.run_all [--data-zip PATH] [--tune] [--ablations] [--sanity] [--n1] [--final] [--seeds 0 1 2]

--final trains what is submitted: the tuned CNN-BiGRU, the BiGRU and the Transformer, all on left-hand-mirrored data
and all with saved weights, then averages them and sets up per-person balancing (runs/deploy.json). It implies --tune
and --n1.

 1. (optional) --data-zip: build data/processed from the competition data (python run_pipeline.py)
 2. pre-compute augmented copies of the training recordings (once, all CPU cores)
 3. (--tune) nested tuning of the deployed pair: cnn_gru on a GPU and lgbm on the CPU at the same time
 4. (--n1) build data/processed_mirror (left-handed wearers mirrored into the right-hand frame) + its copies
 5. network jobs (+ xgboost) run --workers at a time on the GPU(s); the other summary-feature models use the CPU
 6. evaluation (old result files are cleared first): tables with CIs, Holm-corrected comparisons, hybrids,
    the final ensemble and its per-person balancing, error analysis, seed averaging, binary/macro split and
    figures, organised by topic (models/paths.py); results/06_validation/run_info.json records versions, hardware and settings
 7. runs/ and results/ are zipped: cmi_models.zip is what the submission notebook needs
Safe to re-run after a crash or timeout: finished (model, fold, seed) runs, tuning and data builds are skipped."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from models.common import complete
from models.paths import ERRORS, FOLDS, PARAMS, R, RUN_INFO, SECTIONS

PY = sys.executable
NETS = ['cnn_gru', 'cnn', 'gru', 'transformer']            # main model first, so partial results are already useful
TREES = ['lgbm', 'xgb', 'rf', 'mlp']
ABLATIONS = {'abl_noaug': ['--aug', '0'], 'abl_nophase': ['--phase', '0'],
             'abl_notofreg': ['--drop-sub', '_r0', '_r1', '_r2']}         # each removes one part of the main model
FINAL = ['n1_mirror', 'n1_gru', 'n1_transformer']        # the submitted ensemble (all on mirrored data, all saved)
D, MIRROR = Path('data/processed'), Path('data/processed_mirror')


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
    """-> (GPU jobs, CPU jobs, deployed (nn, gb) names). A job = (name, command)."""
    only = lambda n: not a.only or n in a.only
    s = ['--seeds', *map(str, a.seeds)]
    ep = ['--epochs', str(a.epochs)] if a.epochs else []
    nn, gb = ('cnn_gru_tuned', 'lgbm_tuned') if a.tune else ('cnn_gru', 'lgbm')
    tuned = ['--params', str(PARAMS / 'cnn_gru.json')] if a.tune else []
    nets = [(n, n, []) for n in NETS if only(n)]
    trees = [(m, m, []) for m in TREES if only(m)]
    if a.tune:
        nets.insert(0, ('cnn_gru_tuned', 'cnn_gru', tuned))
        trees.insert(0, ('lgbm_tuned', 'lgbm', ['--params', str(PARAMS / 'lgbm.json')]))
    if a.n1:          # same model and settings as the tuned network, mirrored data: isolates the effect of N1
        nets.append(('n1_mirror', 'cnn_gru', [*tuned, '--data', str(MIRROR)]))
    if a.final:       # the other two architectures on mirrored data, at their (default) settings
        nets += [(n, core, ['--data', str(MIRROR)]) for n, core in (('n1_gru', 'gru'), ('n1_transformer', 'transformer'))]
        nets.sort(key=lambda n: n[0] not in FINAL)     # submitted networks first: a cut-short session still has them
    keep = {nn, *(FINAL if a.final else ())}           # runs whose weights are saved (everything that can be submitted)
    if a.ablations:
        nets += [(n, 'cnn_gru', x) for n, x in ABLATIONS.items()]
    if a.sanity:      # permuted labels: scores must collapse to chance, otherwise something leaks
        nets.append(('sanity_shuffled_cnn_gru', 'cnn_gru', ['--shuffle-labels', '--seeds', '0']))
        trees.append(('sanity_shuffled_lgbm', 'lgbm', ['--shuffle-labels', '--seeds', '0']))
    gpu = [(f'{name}_{v}_f{k}', [PY, '-m', f'models.train.{core}', '--name', name, '--variant', v, '--folds', str(k),
                                 *s, *ep, *extra, *(['--save'] if name in keep else [])])
           for name, core, extra in nets for v in ('full', 'imu') for k in range(5)]
    cpu = []
    for name, core, extra in trees:
        cmd = [PY, '-m', f'models.train.{core}', '--name', name, *s, '--trees', str(a.trees), *extra,
               *(['--save'] if name == gb else [])]
        (gpu if core == 'xgb' and a.n_gpu else cpu).append((name, cmd))    # xgboost is much faster on a GPU
    return gpu, cpu, (nn, gb)


def bundle():
    """Ship the exact metadata, scalers and tree-column order this run trained with, next to the models,
    so the submission can never pair the models with a differently built copy. runs/processed = as recorded,
    runs/processed_mirror = left-handers mirrored (the submission picks the one each checkpoint was trained on)."""
    import pyarrow.parquet as pq
    for src, out in ((D, Path('runs/processed')), (MIRROR, Path('runs/processed_mirror'))):
        if not (src / 'meta.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / 'meta.json', out / 'meta.json')
        shutil.copytree(src / 'scalers', out / 'scalers', dirs_exist_ok=True)
        pd.read_csv(src / 'sequences.csv')[['sequence_id', 'subject', 'fold']].to_csv(out / 'folds_used.csv', index=False)
        cols = [c for c in pq.read_schema(src / 'tree_full.parquet').names if c != 'sequence_id']
        (out / 'tree_columns.json').write_text(json.dumps(cols))


def clear_results():
    """Remove result files from earlier runs (all regenerated below). Kept: tuned settings and trials (reused),
    the fold split, run info and data-build evidence."""
    keep = {'run_info.json', 'folds_used.csv'}
    for sec in SECTIONS:
        for p in list(sec.glob('*')) if sec.exists() else []:
            if p.is_file() and p.name not in keep:
                p.unlink()
            elif p.is_dir() and (p.name == 'figures' or sec == ERRORS):
                shutil.rmtree(p)
    shutil.rmtree(R / 'other', ignore_errors=True)


def run_info(a, t0, failed, nn, gb):
    import importlib.metadata as md
    import torch
    vers = {}
    for p in ('torch', 'lightgbm', 'xgboost', 'scikit-learn', 'optuna', 'numpy', 'pandas'):
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            vers[p] = None
    info = dict(finished=time.strftime('%Y-%m-%d %H:%M:%S %Z'), minutes=round((time.time() - t0) / 60, 1),
                args=vars(a), failed=failed, deployed=json.loads(Path('runs/deploy.json').read_text())
                if Path('runs/deploy.json').exists() else None, python=platform.python_version(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], packages=vers,
                complete_runs=sorted(p.name for p in Path('runs').iterdir()
                                     if p.is_dir() and p.name not in ('logs', 'processed') and complete(p.name)))
    RUN_INFO.parent.mkdir(parents=True, exist_ok=True)
    if (D / 'sequences.csv').exists():          # the exact person -> fold split behind every number
        pd.read_csv(D / 'sequences.csv')[['sequence_id', 'subject', 'fold']].to_csv(FOLDS, index=False)
    RUN_INFO.write_text(json.dumps(info, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-zip', help='competition zip/folder: builds data/processed first (skipped if already built)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=3, help='GPU jobs running at once (total, shared over GPUs)')
    ap.add_argument('--epochs', type=int, default=None, help='override the epoch count of every network (testing)')
    ap.add_argument('--trees', type=int, default=500)
    ap.add_argument('--aug-copies', type=int, default=6)
    ap.add_argument('--only', nargs='*', help='train only these base models (e.g. cnn_gru lgbm)')
    ap.add_argument('--tune', action='store_true', help='nested tuning of cnn_gru + lgbm, then train and deploy them')
    ap.add_argument('--trials', type=int, nargs=2, default=[12, 20], metavar=('NN', 'TREES'))
    ap.add_argument('--n1', action='store_true', help='N1 experiment: the tuned network on left-hand-mirrored data')
    ap.add_argument('--final', action='store_true', help='train, save and deploy the submitted ensemble (implies --tune --n1)')
    ap.add_argument('--ablations', action='store_true', help='also run the main-model ablations')
    ap.add_argument('--sanity', action='store_true', help='also run the shuffled-label leakage check')
    ap.add_argument('--no-eval', action='store_true')
    a = ap.parse_args()
    if a.final:
        a.tune = a.n1 = True
    t0, failed = time.time(), []
    legacy = R / 'params'                    # older layout (Pass 1-2 output): keep its tuned settings, don't re-tune
    if legacy.exists() and not PARAMS.exists():
        PARAMS.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(PARAMS))

    if not (D / 'features.npz').exists() or not (D / 'canonical' / 'offsets.npy').exists():
        if not a.data_zip:
            sys.exit('data/processed/features.npz or canonical/ missing: rerun with --data-zip <competition zip or folder>')
        print('building the dataset (about 5 minutes) ...', flush=True)
        if sh([PY, 'run_pipeline.py', '--data', a.data_zip])[0]:
            sys.exit('run_pipeline.py failed')
    if a.aug_copies and not (D / 'aug_x.npy').exists():
        print('pre-computing augmented copies ...', flush=True)
        if sh([PY, '-m', 'models.train.augment', '--copies', str(a.aug_copies)])[0]:
            sys.exit('augmentation failed')

    import torch
    a.n_gpu = n_gpu = torch.cuda.device_count()

    def env(i, use_gpu, xgb=False):
        """Spread GPU jobs over the GPUs; hide the GPU from CPU jobs so the networks keep it."""
        e = dict(os.environ)
        if not use_gpu:
            e['CUDA_VISIBLE_DEVICES'] = ''
            e['CMI_XGB_DEVICE'] = 'cpu'
        elif n_gpu:
            e['CUDA_VISIBLE_DEVICES'] = str(i % n_gpu)
            e['CMI_XGB_DEVICE'] = 'cuda' if xgb else 'cpu'
        return e

    if a.tune:
        todo = [(m, n, g) for m, n, g in (('cnn_gru', a.trials[0], True), ('lgbm', a.trials[1], False))
                if not (PARAMS / f'{m}.json').exists()]
        if todo:
            print('tuning ' + ', '.join(f'{m} ({n} trials)' for m, n, _ in todo) + ' in parallel ...', flush=True)
            cmd = lambda m, n: [PY, '-m', 'models.train.tune', '--model', m, '--trials', str(n),
                                *(['--epochs', str(a.epochs)] if a.epochs else [])]
            with ThreadPoolExecutor(2) as ex:
                res = list(ex.map(lambda j: (j[0], *sh(cmd(j[0], j[1]), f'runs/logs/tune_{j[0]}.log', env(0, j[2]))),
                                  todo))
            for m, rc, secs in res:
                print(f'tune {m}: {"ok" if rc == 0 else "FAILED (runs/logs/tune_" + m + ".log)"} {secs / 60:.0f} min')
                if rc:
                    sys.exit('tuning failed')
        else:
            print(f'tuning: reusing {PARAMS}/cnn_gru.json and lgbm.json', flush=True)

    if a.n1:
        if not (MIRROR / 'features.npz').exists() or not (MIRROR / 'canonical' / 'offsets.npy').exists():
            src = a.data_zip or '/kaggle/input'
            print('N1: building the left-hand-mirrored dataset ...', flush=True)
            if sh([PY, 'scripts/01_build_dataset.py', '--out', str(MIRROR), '--mirror-left',
                   '--folds-from', str(D / 'sequences.csv'),
                   *(['--data', a.data_zip] if a.data_zip else [])], 'runs/logs/build_mirror.log')[0]:
                sys.exit(f'mirrored build failed (runs/logs/build_mirror.log); pass --data-zip <competition folder> '
                         f'(tried {src})')
        s0, s1 = pd.read_csv(D / 'sequences.csv'), pd.read_csv(MIRROR / 'sequences.csv')
        if not (s0[['sequence_id', 'fold', 'gesture_id']].equals(s1[['sequence_id', 'fold', 'gesture_id']])):
            sys.exit('mirrored dataset has a different recording order or folds: comparison would be invalid')
        if a.aug_copies and not (MIRROR / 'aug_x.npy').exists():
            if sh([PY, '-m', 'models.train.augment', '--data', str(MIRROR), '--copies', str(a.aug_copies)],
                  'runs/logs/augment_mirror.log')[0]:
                sys.exit('mirrored augmentation failed')

    gpu_jobs, cpu_jobs, (nn, gb) = plan(a)
    print(f'{len(gpu_jobs)} GPU jobs ({a.workers} at a time, {n_gpu} GPU(s)) + {len(cpu_jobs)} CPU tree models, '
          f'seeds {a.seeds}; deploying {" + ".join(FINAL) if a.final else nn + " + " + gb}', flush=True)

    def run(i, name, cmd, use_gpu):
        return name, *sh(cmd, f'runs/logs/{name}.log', env(i, use_gpu, xgb='models.train.xgb' in cmd))

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
        clear_results()

        def ev(title, *args):
            print(f'\n--- {title}', flush=True)
            if subprocess.run([PY, '-m', 'models.evaluate', *args]).returncode:
                failed.append(title)

        base = list(dict.fromkeys(n for n in [nn, gb, *NETS, *TREES] if complete(n)))
        if base:
            ev('model comparison', 'table', *base, *([f'{nn}:blank'] if complete(nn, ('full', 'blank')) else []),
               '--tag', 'main')
        if nn in base and len(base) > 1:
            ev(f'is {nn} better than each other model? (Holm-corrected)', 'compare', nn,
               *[b for b in base if b != nn], '--tag', 'main')
        if a.tune and all(complete(n) for n in ('cnn_gru', 'lgbm', nn, gb)):
            ev('does tuning help? (fold0 = untouched by tuning)', 'table', 'cnn_gru', nn, 'lgbm', gb, '--tag', 'tuning')
            ev('tuning, tested (Holm-corrected)', 'compare', nn, 'cnn_gru', '--tag', 'tuning_nn')
            ev('tuning, tested (Holm-corrected)', 'compare', gb, 'lgbm', '--tag', 'tuning_gb')
        if {nn, gb} <= set(base):
            ev('hybrids (deployed pair)', 'hybrid', nn, gb)
            h = [f'{x}_{nn}_{gb}' for x in ('h1', 'h2', 'h3')]
            ev('do the hybrids beat the network alone? (Holm-corrected)', 'compare', h[0], nn, *h[1:], '--tag', 'hybrids')
            for m in (nn, gb, h[0]):
                ev(f'error analysis: {m}', 'analysis', m)
        if complete(nn, ('full', 'blank')):
            ev('routing: dedicated motion-only model vs the all-sensor model with sensors off', 'compare', nn,
               f'{nn}:blank', '--tag', 'routing')
        abl = [n for n in ABLATIONS if complete(n)]
        if abl and complete('cnn_gru'):
            ev('ablations (each removes one component of the main model)', 'table', 'cnn_gru', *abl, '--tag', 'ablations')
            ev('ablations, tested (Holm-corrected)', 'compare', 'cnn_gru', *abl, '--tag', 'ablations')
        n1 = [(m, b) for m, b in (('n1_mirror', nn), ('n1_gru', 'gru'), ('n1_transformer', 'transformer'))
              if complete(m) and complete(b)]                 # each architecture with and without mirroring
        if n1:
            ev('N1: mirroring left-handed wearers', 'table', *[x for m, b in n1 for x in (b, m)], '--tag', 'n1')
            ev('N1, tested per architecture (Holm-corrected)', 'pairs', *[f'{m},{b}' for m, b in n1], '--tag', 'n1')
            ev('error analysis: n1_mirror', 'analysis', 'n1_mirror')
        if all(complete(m) for m in FINAL):
            h1 = [f'h1_{nn}_{gb}'] if complete(f'h1_{nn}_{gb}') else []
            ev('final ensemble = equal-weight average of the three mirrored networks', 'ensemble', 'final', *FINAL)
            ev('final ensemble vs its members and the first submission', 'table', 'final', *FINAL, *h1, nn,
               '--tag', 'final')
            ev('final ensemble, tested (Holm-corrected)', 'compare', 'final', *FINAL, *h1, nn, '--tag', 'final')
            ev('error analysis: final', 'analysis', 'final')
            ev('per-person balancing (strength chosen on other folds) -> runs/deploy.json', 'deploy', 'final', *FINAL)
        arch = [n for n in ('gru', 'cnn_gru', 'cnn', 'transformer') if complete(n)]
        if len(arch) > 1:     # a fair architecture test: every core at its default settings
            ev('architectures at default settings (Holm-corrected)', 'compare', *arch, '--tag', 'arch_default')
        trained = [n for n in [nn, gb, *NETS, *TREES, *FINAL, *ABLATIONS] if complete(n)]
        ev('seed averaging', 'seeds', *dict.fromkeys(trained))
        ev('binary vs macro F1', 'breakdown', *dict.fromkeys(trained + [n for n in (f'h1_{nn}_{gb}', 'final') if complete(n)]))
        sanity = [n for n in ('sanity_shuffled_cnn_gru', 'sanity_shuffled_lgbm') if complete(n)]
        if sanity:
            ev('sanity check: shuffled labels must fall to chance (about 0.37-0.40 for this metric)', 'table',
               *sanity, '--tag', 'sanity')

    if not a.no_eval and subprocess.run([PY, '-m', 'models.report_figures']).returncode:
        failed.append('figures')
    if (D / 'meta.json').exists():
        bundle()
    run_info(a, t0, failed, nn, gb)
    shutil.make_archive('cmi_models', 'zip', 'runs')              # checkpoints + OOF + blend_weights.json + deploy.json
    shutil.make_archive('cmi_results', 'zip', 'results')
    print(f'\nfinished in {(time.time() - t0) / 60:.0f} min. Saved: cmi_models.zip, cmi_results.zip, runs/, results/')
    if failed:
        print('FAILED:', ', '.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
