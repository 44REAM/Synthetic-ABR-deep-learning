"""Summary of experiments/real_vs_synthetic.py (8 development subjects, EEGConformer), for the
SETTING given below ('noaug' or 'aug', as in the experiment script).

Per training source:
  held-out aug. set   AUROC on the four Southampton subjects kept for augmentation tuning:
                      median over runs, and mean over runs with its 95% participant-bootstrap CI
                      and the SD over runs
  test sets           AUROC on all waveforms and near threshold (levels at which both labels
                      occur in that set): mean over runs with its 95% participant-bootstrap CI
                      (analysis/bootstrap.py), and the SD over runs; '-' where a real-data source
                      would be tested on its own dataset
  validation          AUROC on the run's own validation subjects (real: expert labels;
                      synthetic: its label rule) and the stopping step
  synthetic - real    every synthetic, mixing and control source minus every real-data
                      source on each test set the real-data source is tested on (AUROC, all
                      waveforms): paired difference on the same participant draws with its
                      95% CI and bootstrap p, Benjamini-Hochberg q over all these comparisons
                      as one family (* q < 0.05), computed as in analysis/multi_arch.py.
Only AUROC is reported; threshold metrics stay in per_run_metrics.csv.
Writes per_run_metrics.csv, bootstrap_auroc_ci.csv, paired_diff.csv and summary.txt into the
results folder.

Run from the repository root:
  python analysis/real_vs_synthetic.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from analysis.bootstrap import bootstrap_auroc
from analysis.metrics import near_threshold_ids, per_run_metrics
from analysis.multi_arch import ALPHA, HUMAN, synthetic_vs_human
from src.data import AUGDEV_SET
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 'noaug' or 'aug'; 3_analysis_and_outputs.py sets it through ABR_SETTING
SETTING = os.environ.get('ABR_SETTING', 'aug')
RESULTS_DIR = REPOSITORY_ROOT / 'results' / f'real_vs_synthetic_{SETTING}'
# Display order and names
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet',
           'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'mixup': 'MixUp', 'cutmix': 'CutMix',
           'gaussian': 'Control: Gaussian', 'gaussian_latency': 'Control: latency-only Gaussian',
           'simple_gaussian': 'Control: simple Gaussian'}
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}


def mean_sd(values: pd.Series) -> str:
    return f'{values.mean():.3f} ± {values.std():.3f}'


def ci_cells(ci: pd.DataFrame, sd: pd.Series) -> pd.Series:
    """'mean (CI low-high) sd ..' per (sources, eval_set); sd indexed the same way."""
    c = ci.set_index(['sources', 'eval_set'])
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
    # Grouped by (arch, sources) so the draws can be paired as in analysis/multi_arch.py (one architecture here)
    test_ci, draws = bootstrap_auroc(runs, preds, test_wf, ['arch', 'sources'],
                                     {'all': None, 'near': near_threshold_ids(test_wf)}, return_draws=True)
    ci = pd.concat([
        test_ci,
        bootstrap_auroc(runs, preds, waveforms[waveforms['eval_set'] == AUGDEV_SET], ['arch', 'sources'], {'all': None}),
    ], ignore_index=True)
    ci.to_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')
    sd = metrics.groupby(['sources', 'eval_set'])[['auc', 'auc_near']].std()
    order = [s for s in SOURCES if s in set(runs['sources'])]
    archs = sorted(runs['arch'].unique())
    diffs = synthetic_vs_human(ci, draws, archs, order)
    diffs.to_csv(RESULTS_DIR / 'paired_diff.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    n = runs.groupby('sources')['run'].nunique()
    out(f'{runs["run"].nunique()} runs, {n.min()}-{n.max()} per source; {", ".join(sorted(runs["arch"].unique()))}, '
        f'n_dev {", ".join(map(str, sorted(runs["n_dev"].unique())))}, augmentation '
        f'{", ".join(sorted(runs["augmentation"].unique()))}')

    held_ci = ci[ci['eval_set'] == AUGDEV_SET]
    held_median = metrics[metrics['eval_set'] == AUGDEV_SET].groupby('sources')['auc'].median()
    out('\n=== HELD-OUT AUGMENTATION SET (Southampton s08-s11) ===')
    out(sizes_line(held_ci, {AUGDEV_SET: 'Held-out'}))
    out(pd.DataFrame({'median AUROC': held_median.map('{:.3f}'.format),
                      'mean (95% participant-bootstrap CI), SD over runs':
                          ci_cells(held_ci, sd['auc']).droplevel('eval_set')})
        .reindex(order).rename(index=SOURCES).to_string())

    for subset, title, sd_col in [('all', 'AUROC, all waveforms', 'auc'), ('near', 'AUROC near threshold', 'auc_near')]:
        c = ci[(ci['subset'] == subset) & ci['eval_set'].isin(TEST_SETS)]
        out(f'\n=== TEST DATASETS: {title}, mean over runs (95% participant-bootstrap CI), SD over runs ===')
        out(sizes_line(c, TEST_SETS))
        out(ci_cells(c, sd[sd_col]).unstack('eval_set').reindex(index=order, columns=list(TEST_SETS))
            .rename(index=SOURCES, columns=TEST_SETS).fillna('-').to_string())

    sig = diffs['q'] < ALPHA
    out(f'\n=== SYNTHETIC / CONTROL MINUS HUMAN BASELINE, all with 8 development subjects: AUROC difference (95% '
        f'paired participant-bootstrap CI), * BH q < {ALPHA} over all {len(diffs)} comparisons ===')
    out(f'  higher than the human baseline in {(diffs["diff"] > 0).sum()} of {len(diffs)}; significantly higher in '
        f'{(sig & (diffs["diff"] > 0)).sum()}, significantly lower in {(sig & (diffs["diff"] < 0)).sum()}')
    cells = (diffs['diff'].map('{:+.3f}'.format) + ' (' + diffs['ci_low'].map('{:+.3f}'.format) + ', '
             + diffs['ci_high'].map('{:+.3f}'.format) + ')' + sig.map({True: '*', False: ' '}))
    column = 'vs ' + diffs['baseline'].map(TEST_SETS) + ' on ' + diffs['eval_set'].map(TEST_SETS)
    columns = [f'vs {TEST_SETS[b]} on {TEST_SETS[t]}' for b in HUMAN for t in TEST_SETS if t != b]
    wide = diffs.assign(cell=cells, column=column).pivot_table(index='source', columns='column', values='cell',
                                                               aggfunc='first')
    out(wide.reindex(index=[s for s in order if s not in HUMAN], columns=columns).rename(index=SOURCES).to_string())

    val = metrics.drop_duplicates('run').groupby('sources')
    out('\n=== VALIDATION SUBJECTS (own labels), mean ± SD over runs ===')
    out(pd.DataFrame({'AUROC': val['val_auc'].apply(mean_sd), 'best step': val['best_step'].apply(
        lambda s: f'{s.mean():.0f} ± {s.std():.0f}')}).reindex(order).rename(index=SOURCES).to_string())

    (RESULTS_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
