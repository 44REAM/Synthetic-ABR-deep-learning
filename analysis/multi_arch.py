"""Summary of experiments/multi_arch.py (five waveform architectures and CWT-ViT, tuned augmentation).

  overview            per architecture and source: the mean of its test-set mean AUROCs
                      (real-data sources over the two other datasets)
Per architecture and training source:
  held-out aug. set   AUROC on the four Southampton subjects kept for augmentation tuning:
                      median over runs, and mean over runs with its 95% participant-bootstrap CI
                      and the SD over runs
  test sets           AUROC on all waveforms and near threshold (levels at which both labels
                      occur in that set): mean over runs with its 95% participant-bootstrap CI
                      (analysis/bootstrap.py), and the SD over runs; '-' where a real-data source
                      would be tested on its own dataset
  validation          AUROC on the run's own validation subjects and the stopping step
  synthetic vs real   every synthetic source and control minus every real-data baseline, on
                      each test set the baseline is tested on (AUROC, all waveforms): paired
                      participant-bootstrap difference with 95% CI and two-sided p, Benjamini-
                      Hochberg q over all these comparisons as one family (* q < 0.05)
Only AUROC is reported; threshold metrics stay in per_run_metrics.csv.
Writes per_run_metrics.csv, bootstrap_auroc_ci.csv, paired_diff.csv and summary.txt into the
results folder.

Run from the repository root:
  python analysis/multi_arch.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from analysis.bootstrap import benjamini_hochberg, bootstrap_auroc, paired_difference
from analysis.metrics import near_threshold_ids, per_run_metrics
from src.data import AUGDEV_SET
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
# Display order and names
ARCHS = {'eegconformer': 'EEGConformer', 'deep4': 'DeepConvNet', 'atcnet': 'ATCNet',
         'shallow': 'ShallowConvNet', 'eegnet': 'EEGNet', 'cwt_vit': 'CWT-ViT'}
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet',
           'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'mixup': 'MixUp', 'cutmix': 'CutMix',
           'gaussian': 'Control: Gaussian'}
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
HUMAN = ['wang', 'southampton', 'physionet']
ALPHA = 0.05                         # BH q threshold for '*'
KEYS = ['arch', 'sources']


def synthetic_vs_human(ci: pd.DataFrame, draws: dict, archs: list[str], sources: list[str]) -> pd.DataFrame:
    """Paired differences of every synthetic source and control against every real-data baseline."""
    point = ci[ci['subset'] == 'all'].set_index(['arch', 'sources', 'eval_set'])['auroc']
    rows = []
    for arch in archs:
        for source in (s for s in sources if s not in HUMAN):
            for baseline in (s for s in sources if s in HUMAN):
                for test_set in (t for t in TEST_SETS if t != baseline):
                    a, b = ((arch, source), test_set, 'all'), ((arch, baseline), test_set, 'all')
                    rows.append({'arch': arch, 'source': source, 'baseline': baseline, 'eval_set': test_set,
                                 **paired_difference(point[(arch, source, test_set)], point[(arch, baseline, test_set)],
                                                     draws[a], draws[b])})
    diffs = pd.DataFrame(rows)
    diffs['q'] = benjamini_hochberg(diffs['p'].to_numpy())
    return diffs


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
    test_ci, draws = bootstrap_auroc(runs, preds, test_wf, KEYS, {'all': None, 'near': near_threshold_ids(test_wf)},
                                     return_draws=True)
    ci = pd.concat([
        test_ci,
        bootstrap_auroc(runs, preds, waveforms[waveforms['eval_set'] == AUGDEV_SET], KEYS, {'all': None}),
    ], ignore_index=True)
    ci.to_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')
    sd = metrics.groupby(KEYS + ['eval_set'])[['auc', 'auc_near']].std()
    archs = [a for a in ARCHS if a in set(runs['arch'])]
    sources = [s for s in SOURCES if s in set(runs['sources'])]
    diffs = synthetic_vs_human(ci, draws, archs, sources)
    diffs.to_csv(RESULTS_DIR / 'paired_diff.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    n = runs.groupby(KEYS)['run'].nunique()
    out(f'{runs["run"].nunique()} runs, {n.min()}-{n.max()} per architecture and source; augmentation '
        f'{", ".join(sorted(runs["augmentation"].unique()))}')

    overview = (ci[(ci['subset'] == 'all') & ci['eval_set'].isin(TEST_SETS)].groupby(KEYS)['auroc'].mean()
                .unstack('arch').reindex(index=sources, columns=archs))
    out('\n=== OVERVIEW: mean over the test sets of the mean AUROC over runs ===')
    out(overview.rename(index=SOURCES, columns=ARCHS).to_string(float_format='{:.3f}'.format))

    held = metrics[metrics['eval_set'] == AUGDEV_SET]
    for arch in archs:
        out(f'\n######## {ARCHS[arch]} ########')
        h_ci = ci[(ci['eval_set'] == AUGDEV_SET) & (ci['arch'] == arch)]
        h_median = held[held['arch'] == arch].groupby('sources')['auc'].median()
        out('\n=== HELD-OUT AUGMENTATION SET (Southampton s08-s11: 24 waveforms, 4 participants) ===')
        out(pd.DataFrame({'median AUROC': h_median.map('{:.3f}'.format),
                          'mean (95% participant-bootstrap CI), SD over runs':
                              ci_cells(h_ci, sd['auc']).droplevel(['arch', 'eval_set'])})
            .reindex(sources).rename(index=SOURCES).to_string())
        for subset, title, sd_col in [('all', 'AUROC, all waveforms', 'auc'),
                                      ('near', 'AUROC near threshold', 'auc_near')]:
            c = ci[(ci['subset'] == subset) & ci['eval_set'].isin(TEST_SETS) & (ci['arch'] == arch)]
            out(f'\n=== TEST DATASETS: {title}, mean over runs (95% participant-bootstrap CI), SD over runs ===')
            out(ci_cells(c, sd[sd_col]).droplevel('arch').unstack('eval_set')
                .reindex(index=sources, columns=list(TEST_SETS)).rename(index=SOURCES, columns=TEST_SETS)
                .fillna('-').to_string())

    sig = diffs['q'] < ALPHA
    out(f'\n=== SYNTHETIC / CONTROL MINUS HUMAN BASELINE: AUROC difference (95% paired participant-bootstrap CI), '
        f'* BH q < {ALPHA} over all {len(diffs)} comparisons ===')
    out(f'  higher than the human baseline in {(diffs["diff"] > 0).sum()} of {len(diffs)}; significantly higher in '
        f'{(sig & (diffs["diff"] > 0)).sum()}, significantly lower in {(sig & (diffs["diff"] < 0)).sum()}')
    cells = (diffs['diff'].map('{:+.3f}'.format) + ' (' + diffs['ci_low'].map('{:+.3f}'.format) + ', '
             + diffs['ci_high'].map('{:+.3f}'.format) + ')' + sig.map({True: '*', False: ' '}))
    column = 'vs ' + diffs['baseline'].map(TEST_SETS) + ' on ' + diffs['eval_set'].map(TEST_SETS)
    columns = [f'vs {TEST_SETS[b]} on {TEST_SETS[t]}' for b in HUMAN for t in TEST_SETS if t != b]
    wide = diffs.assign(cell=cells, column=column).pivot_table(index=['arch', 'source'], columns='column',
                                                               values='cell', aggfunc='first')
    for arch in archs:
        out(f'\n{ARCHS[arch]}')
        out(wide.loc[arch].reindex(index=[s for s in sources if s not in HUMAN], columns=columns)
            .rename(index=SOURCES).to_string())

    val = metrics.drop_duplicates('run').groupby(KEYS)
    table = pd.DataFrame({'AUROC': val['val_auc'].apply(lambda s: f'{s.mean():.3f} ± {s.std():.3f}'),
                          'best step': val['best_step'].apply(lambda s: f'{s.mean():.0f} ± {s.std():.0f}')})
    out('\n=== VALIDATION SUBJECTS (own labels), mean ± SD over runs ===')
    out(table.unstack('arch').swaplevel(axis=1).reindex(columns=[(a, m) for a in archs for m in ('AUROC', 'best step')])
        .reindex(sources).rename(index=SOURCES, columns=ARCHS, level=0).to_string())

    (RESULTS_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
