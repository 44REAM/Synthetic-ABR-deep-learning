"""Threshold metrics of experiments/multi_arch.py.

AUROC, accuracy, F1, sensitivity and specificity on all waveforms of each test set, and of the
PhysioNet test set split by tone-burst frequency ('1 kHz', '4 kHz'), for every architecture and
training source, with two threshold rules:
  0.5             a fixed probability threshold (no calibration)
  youden_heldout  one threshold per run, maximising Youden's J (sensitivity + specificity - 1)
                  on the run's predictions for the held-out Southampton subjects (s08-s11, also
                  used for augmentation tuning); ties: the middle of the
                  widest tied range (analysis.metrics.youden_threshold)
Positive class = wave V absent: sensitivity is the share of wave-V-absent waveforms classified
absent, specificity the share of wave-V-present waveforms classified present, F1 is that of the
absent class. Each value is the mean over runs with its 95% participant-bootstrap CI
(analysis/bootstrap.py; a run's threshold is fixed over the draws); AUROC, which needs no
threshold, is read from results/multi_arch/bootstrap_auroc_ci.csv (detectors:
results/template/ and results/adjacent/; per frequency: results/physionet_frequency/), so it
matches those analyses. The SD over runs is kept in threshold_metrics.csv only (none for the
detectors and per-frequency AUROC). The conventional detectors, one per training source and run
(arch 'template': template cross-correlation of analysis/template.py, score -RM; 'adjacent' and
'template_adjacent': analysis/adjacent.py, score -correlation), have the youden_heldout rule
only. Their thresholds in thresholds.csv are in units of -correlation. Run
analysis/template.py, analysis/adjacent.py and then analysis/physionet_frequency.py first.
Writes threshold_metrics.csv, thresholds.csv and summary.txt into results/threshold_metrics/.
Tables: tables/threshold_metrics.py.

Run from the repository root:
  python analysis/threshold_metrics.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from analysis.bootstrap import bootstrap_threshold_metrics
from analysis.metrics import physionet_by_frequency, youden_threshold
from src.data import AUGDEV_SET
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NETWORK_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
TEMPLATE_DIR = REPOSITORY_ROOT / 'results' / 'template'
ADJACENT_DIR = REPOSITORY_ROOT / 'results' / 'adjacent'
DETECTOR_DIRS = (TEMPLATE_DIR, ADJACENT_DIR)
FREQUENCY_DIR = REPOSITORY_ROOT / 'results' / 'physionet_frequency'
OUT_DIR = REPOSITORY_ROOT / 'results' / 'threshold_metrics'
THRESHOLD = 0.5
# Display order and names
DETECTORS = {'template': 'Template', 'adjacent': 'Adjacent', 'template_adjacent': 'Template + adjacent'}
ARCHS = {'eegconformer': 'EEGConformer', 'deep4': 'DeepConvNet', 'atcnet': 'ATCNet',
         'shallow': 'ShallowConvNet', 'eegnet': 'EEGNet', 'cwt_vit': 'CWT-ViT', **DETECTORS}
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet',
           'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'mixup': 'MixUp', 'cutmix': 'CutMix',
           'gaussian': 'Control: Gaussian'}
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
FREQUENCY_SETS = {'1 kHz': 'PhysioNet 1 kHz', '4 kHz': 'PhysioNet 4 kHz'}     # eval_set of physionet_by_frequency
METRICS = {'acc': 'Acc', 'f1': 'F1', 'sens': 'Sens', 'spec': 'Spec'}
RULES = {'0.5': '0.5', 'youden_heldout': 'tuned'}


def detector_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs and scores of analysis/template.py and analysis/adjacent.py as runs of the archs in
    DETECTORS (prob_absent = -correlation)."""
    scores = pd.concat([pd.read_csv(d / 'scores.csv', usecols=['run', 'waveform_id', 'prob_absent'])
                        for d in DETECTOR_DIRS], ignore_index=True)
    names = scores['run'].drop_duplicates()
    pattern = '|'.join(sorted(DETECTORS, key=len, reverse=True))      # longest name first
    parts = names.str.extract(rf'^(?P<arch>{pattern})_(?P<sources>.+)_cv(?P<cv>\d+)$')
    if parts['arch'].isna().any():
        raise ValueError('unknown detector run names')
    runs = pd.DataFrame({'run': names.to_numpy(), 'arch': parts['arch'].to_numpy(),
                         'sources': parts['sources'].to_numpy(), 'cv': parts['cv'].astype(int).to_numpy()})
    return runs, scores


def heldout_thresholds(runs: pd.DataFrame, preds: pd.DataFrame, waveforms: pd.DataFrame) -> pd.DataFrame:
    """Youden threshold of every run from its held-out predictions, with the held-out J."""
    held = preds.merge(waveforms.loc[waveforms['eval_set'] == AUGDEV_SET, ['waveform_id', 'label_absent']],
                       on='waveform_id')
    rows = [{'run': run, **dict(zip(('threshold', 'heldout_j'),
                                    youden_threshold(g['prob_absent'].to_numpy(), g['label_absent'].to_numpy())))}
            for run, g in held.groupby('run')]
    out = runs[['run', 'arch', 'sources', 'cv']].merge(pd.DataFrame(rows), on='run', how='left')
    if out['threshold'].isna().any():
        raise ValueError('some runs have no held-out predictions')
    return out


def main() -> None:
    runs, preds, waveforms = load_results(NETWORK_DIR)
    test_wf = waveforms[waveforms['eval_set'].isin(TEST_SETS)]
    # The detectors as runs next to the networks (the participant draws do not depend on the runs)
    det_runs, det_preds = detector_runs()
    all_runs = pd.concat([runs[['run', 'arch', 'sources', 'cv']], det_runs], ignore_index=True)
    all_preds = pd.concat([preds[['run', 'waveform_id', 'prob_absent']], det_preds], ignore_index=True)
    thresholds = heldout_thresholds(all_runs, all_preds, waveforms)
    run_thresholds = dict(zip(thresholds['run'], thresholds['threshold']))
    # The frequency sets in calls of their own, so the draws of the three test sets stay as they were
    by_freq = physionet_by_frequency(waveforms)
    fixed, tuned = [], []
    for wf in (test_wf, by_freq):
        fixed.append(bootstrap_threshold_metrics(runs, preds, wf, ['arch', 'sources'], {'all': None},
                                                 threshold=THRESHOLD).assign(threshold_rule='0.5'))
        tuned.append(bootstrap_threshold_metrics(all_runs, all_preds, wf, ['arch', 'sources'], {'all': None},
                                                 run_thresholds=run_thresholds).assign(threshold_rule='youden_heldout'))
    fixed, tuned = pd.concat(fixed, ignore_index=True), pd.concat(tuned, ignore_index=True)

    # Point values at 0.5 must equal the per-run metrics of analysis/multi_arch.py (same definitions)
    per_run = pd.read_csv(NETWORK_DIR / 'per_run_metrics.csv')
    per_run = per_run[per_run['eval_set'].isin(TEST_SETS)]
    check = per_run.groupby(['arch', 'sources', 'eval_set'])[list(METRICS)].mean().stack()
    mine = fixed.set_index(['arch', 'sources', 'eval_set', 'metric'])['value']
    if not np.allclose(mine.reindex(check.index).to_numpy(), check.to_numpy()):
        raise ValueError('threshold metrics at 0.5 differ from per_run_metrics.csv')

    auroc = pd.concat([pd.read_csv(d / 'bootstrap_auroc_ci.csv') for d in (NETWORK_DIR, *DETECTOR_DIRS)],
                      ignore_index=True)
    auroc = auroc[auroc['eval_set'].isin(TEST_SETS)]
    frequency = pd.read_csv(FREQUENCY_DIR / 'bootstrap_auroc_ci.csv')
    auroc = pd.concat([auroc, frequency[frequency['eval_set'].isin(FREQUENCY_SETS)]], ignore_index=True)
    auroc = auroc[auroc['subset'] == 'all'].rename(columns={'auroc': 'value'}).assign(
        metric='auroc', threshold_rule='none').merge(
        per_run.groupby(['arch', 'sources', 'eval_set'])['auc'].std().rename('sd').reset_index(),
        on=['arch', 'sources', 'eval_set'], how='left')
    table = pd.concat([auroc[fixed.columns], fixed, tuned], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / 'threshold_metrics.csv', index=False, float_format='%.6g')
    thresholds.to_csv(OUT_DIR / 'thresholds.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    def ordered(d: pd.DataFrame) -> pd.DataFrame:
        order = [(a, s) for a in ARCHS for s in SOURCES if (a, s) in d.index]
        d = d.reindex(order)
        d.index = pd.MultiIndex.from_tuples([(ARCHS[a], SOURCES[s]) for a, s in order], names=['arch', 'source'])
        return d

    pd.set_option('display.width', 300)
    q = thresholds.groupby(['arch', 'sources'])['threshold']
    summary = pd.DataFrame({'median': q.median(), 'q1': q.quantile(0.25), 'q3': q.quantile(0.75),
                            'min': q.min(), 'max': q.max(),
                            'heldout_j_median': thresholds.groupby(['arch', 'sources'])['heldout_j'].median()})
    out('=== HELD-OUT YOUDEN THRESHOLDS per run: median [q1-q3] (min-max) over runs, median held-out J '
        '(detectors: in units of -correlation) ===')
    out(ordered(summary).round(3).to_string())

    cell = (table['value'].map('{:.3f}'.format) + ' (' + table['ci_low'].map('{:.3f}'.format) + '-'
            + table['ci_high'].map('{:.3f}'.format) + ')')
    column = np.where(table['metric'] == 'auroc', 'AUROC',
                      table['metric'].map(METRICS) + ' @' + table['threshold_rule'].map(RULES).fillna(''))
    wide = table.assign(cell=cell, column=column).pivot_table(index=['eval_set', 'arch', 'sources'], columns='column',
                                                              values='cell', aggfunc='first')
    columns = ['AUROC'] + [f'{m} @{r}' for m in METRICS.values() for r in RULES.values()]
    out(f'\nPositive class = wave V absent; @0.5 = fixed threshold, @tuned = held-out Youden threshold of each run; '
        f'mean over runs (95% participant-bootstrap CI)')
    for test_set, name in {**TEST_SETS, **FREQUENCY_SETS}.items():
        out(f'\n=== {name} ===')
        out(ordered(wide.loc[test_set].reindex(columns=columns)).fillna('-').to_string())

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
