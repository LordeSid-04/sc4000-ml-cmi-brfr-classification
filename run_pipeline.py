"""Run the whole preprocessing pipeline with one command.

    python run_pipeline.py            # tests -> build -> checks -> publish (~6 min)
    python run_pipeline.py --data PATH

Stages:
  1. unit tests           the code works on synthetic data with known answers
  2. build                clean + orient + features + labels for all 8,151 recordings,
                          written to a candidate folder (data/processed_candidate/)
  3. parity check         the Kaggle predict() path gives exactly the training features
  4. quality checks       22 checks on the candidate dataset
  5. publish              ONLY if 1-4 all pass: the candidate replaces data/processed/
                          and the large files are zipped for the GitHub Release

If any stage fails, the run stops and data/processed/ is left untouched.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / 'data' / 'processed'
CANDIDATE = ROOT / 'data' / 'processed_candidate'
RELEASE_ZIP = ROOT / 'outputs' / 'release' / 'processed_large_files.zip'
SUMMARY = ROOT / 'outputs' / 'run_summary.json'
# Too big for git (GitHub rejects files over 100 MB): shipped as a Release asset.
LARGE_FILES = ['features.npz', 'canonical']


def run_stage(name, args):
    print(f'\n=== {name} ===', flush=True)
    start = time.time()
    ok = subprocess.run([sys.executable, *args], cwd=ROOT).returncode == 0
    seconds = round(time.time() - start)
    print(f'--- {name}: {"PASS" if ok else "FAIL"} ({seconds}s)', flush=True)
    return {'stage': name, 'pass': ok, 'seconds': seconds}


def remove_old(old):
    """Delete the previous build. Windows/OneDrive can briefly lock files that another
    program has open; that only leaves a stale copy behind, so it is a warning."""
    shutil.rmtree(old, ignore_errors=True)
    if old.exists():
        print(f'warning: could not fully delete {old.relative_to(ROOT)} (a file is open elsewhere); '
              'delete it by hand, the new data is already in place')


def publish():
    """Swap the checked candidate into data/processed/ and zip the large files."""
    old = PROCESSED.with_name('processed_old')
    if old.exists():
        remove_old(old)
    if PROCESSED.exists():
        PROCESSED.rename(old)
    CANDIDATE.rename(PROCESSED)
    if old.exists():
        remove_old(old)
    package()


def package():
    """Zip the files too big for git, with paths relative to the repository root."""
    RELEASE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RELEASE_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in LARGE_FILES:
            path = PROCESSED / name
            for f in sorted([path] if path.is_file() else path.rglob('*')):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT).as_posix())
    print(f'published {PROCESSED.relative_to(ROOT)}; release zip {RELEASE_ZIP.relative_to(ROOT)} '
          f'({RELEASE_ZIP.stat().st_size / 1e6:.0f} MB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='Kaggle ZIP or extracted folder (default: data/raw/)')
    args = parser.parse_args()
    data = ['--data', args.data] if args.data else []
    candidate = str(CANDIDATE.relative_to(ROOT))

    if CANDIDATE.exists():
        shutil.rmtree(CANDIDATE)
    stages = [('unit tests', ['-m', 'pytest', '-q', 'tests']),
              ('build dataset', ['scripts/01_build_dataset.py', '--out', candidate, *data]),
              ('parity check', ['scripts/02_parity_check.py', '--processed', candidate, *data]),
              ('quality checks', ['scripts/03_quality_checks.py', '--processed', candidate])]

    results = []
    start = time.time()
    for name, stage_args in stages:
        results.append(run_stage(name, stage_args))
        if not results[-1]['pass']:
            break
    passed = len(results) == len(stages) and all(r['pass'] for r in results)
    if passed:
        publish()
    elif CANDIDATE.exists():
        print(f'\nnot published: {candidate}/ kept for inspection, data/processed/ unchanged')

    summary = {'passed': passed, 'minutes': round((time.time() - start) / 60, 1), 'stages': results}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n=== summary ===')
    for r in results:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['stage']:<28} {r['seconds']:>5}s")
    for name, _ in stages[len(results):]:
        print(f'  SKIP  {name}')
    print(f"\n{'ALL STAGES PASSED: data/processed/ is ready' if passed else 'RUN FAILED'} "
          f"({summary['minutes']} min)")
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
