"""Summary of experiments/cutpoints.py (Roenne+Verhulst, EEGConformer, five label rules).

  held-out aug. set   AUROC on the four Southampton subjects kept for augmentation tuning:
                      median over runs, and as for the test sets
  test sets           AUROC on all waveforms: mean over runs with its 95% participant-bootstrap
                      CI (analysis/bootstrap.py), and the SD over runs
  synthetic val.      AUROC on the run's own validation subjects, labelled by its own rule, and
                      the stopping step
Only AUROC is reported; threshold metrics stay in per_run_metrics.csv.
Writes per_run_metrics.csv, bootstrap_auroc_ci.csv and summary.txt into the results folder.

Run from the repository root:
  python analysis/cutpoints.py
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
from analysis.metrics import per_run_metrics
from src.data import AUGDEV_SET
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'cutpoints'
LABEL_RULES = ['paper', 'cut50', 'cut40', 'cut30', 'cut20']   # display order
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}


def mean_sd(values: pd.Series) -> str:
    return f'{values.mean():.3f} ± {values.std():.3f}'


def ci_cells(ci: pd.DataFrame, sd: pd.Series) -> pd.Series:
    """'mean (CI low-high) sd ..' per (label_rule, eval_set); sd indexed the same way."""
    c = ci.set_index(['label_rule', 'eval_set'])
    return (c['auroc'].map('{:.3f}'.format) + ' (' + c['ci_low'].map('{:.3f}'.format) + '-'
            + c['ci_high'].map('{:.3f}'.format) + ') sd ' + sd.reindex(c.index).map('{:.3f}'.format))


def sizes_line(ci: pd.DataFrame, names: dict[str, str]) -> str:
    n = ci.groupby('eval_set')[['n_waveforms', 'n_participants']].first()
    return '  ' + ', '.join(f'{names[s]}: {n.loc[s, "n_waveforms"]} waveforms, {n.loc[s, "n_participants"]} '
                            f'participants' for s in names)


def main() -> None:
    runs, preds, waveforms = load_results(RESULTS_DIR)
    metrics = per_run_metrics(runs, preds, waveforms)
    metrics.to_csv(RESULTS_DIR / 'per_run_metrics.csv', index=False, float_format='%.6g')

    test_wf = waveforms[waveforms['eval_set'].isin(TEST_SETS)]
    ci = pd.concat([
        bootstrap_auroc(runs, preds, test_wf, ['label_rule'], {'all': None}),
        bootstrap_auroc(runs, preds, waveforms[waveforms['eval_set'] == AUGDEV_SET], ['label_rule'], {'all': None}),
    ], ignore_index=True)
    ci.to_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')
    sd = metrics.groupby(['label_rule', 'eval_set'])['auc'].std()

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    out(f'{runs["run"].nunique()} runs: {", ".join(sorted(runs["sources"].unique()))}, '
        f'{", ".join(sorted(runs["arch"].unique()))}, {runs.groupby("label_rule")["run"].nunique().min()} per rule')

    held_ci = ci[ci['eval_set'] == AUGDEV_SET]
    held_median = metrics[metrics['eval_set'] == AUGDEV_SET].groupby('label_rule')['auc'].median()
    out('\n=== HELD-OUT AUGMENTATION SET (Southampton s08-s11): selection by median AUROC over runs ===')
    out(sizes_line(held_ci, {AUGDEV_SET: 'Held-out'}))
    out(pd.DataFrame({'median AUROC': held_median.map('{:.3f}'.format),
                      'mean (95% participant-bootstrap CI), SD over runs':
                          ci_cells(held_ci, sd).droplevel('eval_set')}).reindex(LABEL_RULES).to_string())

    c = ci[ci['eval_set'].isin(TEST_SETS)]
    out('\n=== TEST DATASETS: AUROC, all waveforms, mean over runs (95% participant-bootstrap CI), SD over runs ===')
    out(sizes_line(c, TEST_SETS))
    out(ci_cells(c, sd).unstack('eval_set').reindex(index=LABEL_RULES, columns=list(TEST_SETS))
        .rename(columns=TEST_SETS).to_string())

    val = metrics.drop_duplicates('run').groupby('label_rule')
    out('\n=== SYNTHETIC VALIDATION SUBJECTS (own label rule), mean ± SD over runs ===')
    out(pd.DataFrame({'AUROC': val['val_auc'].apply(mean_sd), 'best step': val['best_step'].apply(
        lambda s: f'{s.mean():.0f} ± {s.std():.0f}')}).reindex(LABEL_RULES).to_string())

    (RESULTS_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
