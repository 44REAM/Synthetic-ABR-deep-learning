"""Summary of experiments/dev_size.py (synthetic development-pool size, EEGConformer).

Per augmentation, training source and development-pool size:
  held-out aug. set   AUROC on the four Southampton subjects kept for augmentation tuning:
                      median over runs, and mean over runs with its 95% participant-bootstrap CI
                      and the SD over runs
  test sets           AUROC on all waveforms and near threshold (levels at which both labels
                      occur in that set): mean over runs with its 95% participant-bootstrap CI
                      (analysis/bootstrap.py), and the SD over runs
  validation          AUROC on the run's own synthetic validation subjects and the stopping step
Only AUROC is reported; threshold metrics stay in per_run_metrics.csv.
Writes per_run_metrics.csv, bootstrap_auroc_ci.csv and summary.txt into the results folder.

Run from the repository root:
  python analysis/dev_size.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from analysis.bootstrap import bootstrap_auroc
from analysis.metrics import near_threshold_ids, per_run_metrics
from src.data import AUGDEV_SET
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'dev_size'
# Display order and names
SOURCES = {'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'gaussian': 'Control: Gaussian'}
AUGMENTATIONS = {'none': 'NO AUGMENTATION', 'tuned': 'TUNED AUGMENTATION'}
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
KEYS = ['sources', 'n_dev', 'augmentation']


def ci_cells(ci: pd.DataFrame, sd: pd.Series) -> pd.Series:
    """'mean (CI low-high) sd ..' per (KEYS..., eval_set); sd indexed the same way."""
    c = ci.set_index(KEYS + ['eval_set'])
    return (c['auroc'].map('{:.3f}'.format) + ' (' + c['ci_low'].map('{:.3f}'.format) + '-'
            + c['ci_high'].map('{:.3f}'.format) + ') sd ' + sd.reindex(c.index).map('{:.3f}'.format))


def main() -> None:
    runs, preds, waveforms = load_results(RESULTS_DIR)
    metrics = per_run_metrics(runs, preds, waveforms)
    metrics.to_csv(RESULTS_DIR / 'per_run_metrics.csv', index=False, float_format='%.6g')

    test_wf = waveforms[waveforms['eval_set'].isin(TEST_SETS)]
    ci = pd.concat([
        bootstrap_auroc(runs, preds, test_wf, KEYS, {'all': None, 'near': near_threshold_ids(test_wf)}),
        bootstrap_auroc(runs, preds, waveforms[waveforms['eval_set'] == AUGDEV_SET], KEYS, {'all': None}),
    ], ignore_index=True)
    ci.to_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')
    sd = metrics.groupby(KEYS + ['eval_set'])[['auc', 'auc_near']].std()
    rows = [(s, n) for s in SOURCES if s in set(runs['sources']) for n in sorted(runs['n_dev'].unique())]

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    def show(table: pd.DataFrame) -> None:
        out(table.reindex(rows).rename(index=SOURCES, level='sources').to_string())

    pd.set_option('display.width', 250)
    n = runs.groupby(KEYS)['run'].nunique()
    out(f'{runs["run"].nunique()} runs, {n.min()}-{n.max()} per setting; {", ".join(sorted(runs["arch"].unique()))}')

    held = metrics[metrics['eval_set'] == AUGDEV_SET]
    for aug, aug_title in AUGMENTATIONS.items():
        if aug not in set(runs['augmentation']):
            continue
        out(f'\n######## {aug_title} ########')
        h_ci = ci[(ci['eval_set'] == AUGDEV_SET) & (ci['augmentation'] == aug)]
        h_median = held[held['augmentation'] == aug].groupby(['sources', 'n_dev'])['auc'].median()
        out('\n=== HELD-OUT AUGMENTATION SET (Southampton s08-s11: 24 waveforms, 4 participants) ===')
        show(pd.DataFrame({'median AUROC': h_median.map('{:.3f}'.format),
                           'mean (95% participant-bootstrap CI), SD over runs':
                               ci_cells(h_ci, sd['auc']).droplevel(['augmentation', 'eval_set'])}))
        for subset, title, sd_col in [('all', 'AUROC, all waveforms', 'auc'),
                                      ('near', 'AUROC near threshold', 'auc_near')]:
            c = ci[(ci['subset'] == subset) & ci['eval_set'].isin(TEST_SETS) & (ci['augmentation'] == aug)]
            out(f'\n=== TEST DATASETS: {title}, mean over runs (95% participant-bootstrap CI), SD over runs ===')
            show(ci_cells(c, sd[sd_col]).droplevel('augmentation').unstack('eval_set')
                 .reindex(columns=list(TEST_SETS)).rename(columns=TEST_SETS))

    val = metrics.drop_duplicates('run').groupby(KEYS)
    table = pd.DataFrame({'AUROC': val['val_auc'].apply(lambda s: f'{s.mean():.3f} ± {s.std():.3f}'),
                          'best step': val['best_step'].apply(lambda s: f'{s.mean():.0f} ± {s.std():.0f}')})
    out('\n=== SYNTHETIC VALIDATION SUBJECTS (own label rule), mean ± SD over runs ===')
    show(table.unstack('augmentation').swaplevel(axis=1).sort_index(axis=1))

    (RESULTS_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
