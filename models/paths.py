"""Where every result file lives. One place, so trainers, evaluation, figures and the data builder agree.

results/
  01_model_comparison/        every model on the competition metric, architectures, seeds, binary vs macro F1
  02_hyperparameter_tuning/   nested Optuna settings + trials, default vs tuned
  03_ablations/               component ablations, N1 (mirroring), N3 (routing)
  04_hybrids_and_ensembles/   blends, stacker, the final ensemble, per-person balancing, what is deployed
  05_error_analysis/          per model: per-class F1, confusion matrices, per person, per posture
  06_validation/              shuffled-label check, the fold split, run info, data-build evidence
Each folder has its CSV/JSON tables and a figures/ sub-folder (PNG)."""
from pathlib import Path

R = Path('results')
COMPARISON = R / '01_model_comparison'
TUNING = R / '02_hyperparameter_tuning'
ABLATIONS = R / '03_ablations'
HYBRIDS = R / '04_hybrids_and_ensembles'
ERRORS = R / '05_error_analysis'
VALIDATION = R / '06_validation'
SECTIONS = (COMPARISON, TUNING, ABLATIONS, HYBRIDS, ERRORS, VALIDATION)

PARAMS = TUNING / 'params'                     # tuned settings, read by the trainers via --params
FOLDS = VALIDATION / 'folds_used.csv'           # the published person -> fold split, reused by every rebuild
RUN_INFO = VALIDATION / 'run_info.json'

TAG_DIR = {'main': COMPARISON, 'arch_default': COMPARISON,
           'tuning': TUNING, 'tuning_nn': TUNING, 'tuning_gb': TUNING,
           'ablations': ABLATIONS, 'n1': ABLATIONS, 'routing': ABLATIONS,
           'hybrids': HYBRIDS, 'final': HYBRIDS, 'balancing': HYBRIDS,
           'sanity': VALIDATION}


def out_dir(tag=None):
    """Folder for a table/comparison with this tag (untagged ad-hoc analyses go to results/other/)."""
    d = TAG_DIR.get(tag, R / 'other')
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures(section):
    d = section / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d
