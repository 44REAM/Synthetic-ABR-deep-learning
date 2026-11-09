"""Summary of experiments/tune_augmentation.py: AUROC on the held-out Southampton subjects only.

Per training source and augmentation configuration: median AUROC over runs, and mean over runs
with its 95% participant-bootstrap CI and the SD over
runs.
Writes per_run_metrics.csv, bootstrap_auroc_ci.csv and summary.txt into the results folder.

Run from the repository root:
  python analysis/tune_augmentation.py
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
from analysis.metrics import per_run_metrics
from experiments.tune_augmentation import CONFIGS, STACKED, TUNE_SETTINGS
from src.data import AUGDEV_SET
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 'eegconformer' or 'cwt_vit', as in the experiment script; 3_analysis_and_outputs.py sets it
# through ABR_ARCH
ARCH = os.environ.get('ABR_ARCH', 'eegconformer')
RESULTS_DIR = REPOSITORY_ROOT / 'results' / TUNE_SETTINGS[ARCH]['folder']
# Display order and names
SOURCES = {'roenne': 'Roenne', 'verhulst': 'Verhulst', 'gaussian': 'Ctrl Gaussian',
           'gaussian_latency': 'Ctrl latency-only Gaussian',
           'simple_gaussian': 'Ctrl simple Gaussian', 'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
# Settings of the human sources, listed for any human source not in 'tune_human' (none at present)
PAPER_HUMAN_SETTINGS = {
    'wang': {'shift_limit_ms': 1.0, 'pink_noise_sigma': 0.05},
    'southampton': {'pink_noise_sigma': 0.2},
    'physionet': {'shift_limit_ms': 1.0, 'pink_noise_sigma': 0.05},
}


def main() -> None:
    runs, preds, waveforms = load_results(RESULTS_DIR)
    metrics = per_run_metrics(runs, preds, waveforms)
    metrics.to_csv(RESULTS_DIR / 'per_run_metrics.csv', index=False, float_format='%.6g')
    held_wf = waveforms[waveforms['eval_set'] == AUGDEV_SET]
    ci = bootstrap_auroc(runs, preds, held_wf, ['sources', 'augmentation'], {'all': None})
    ci.to_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')

    held = metrics[metrics['eval_set'] == AUGDEV_SET]
    sources = [s for s in SOURCES if s in set(runs['sources'])]
    order = [c for c in list(CONFIGS) + STACKED if c in set(runs['augmentation'])]
    sd = held.groupby(['sources', 'augmentation'])['auc'].std()
    c = ci.set_index(['sources', 'augmentation'])
    cells = (c['auroc'].map('{:.3f}'.format) + ' (' + c['ci_low'].map('{:.3f}'.format) + '-'
             + c['ci_high'].map('{:.3f}'.format) + ') sd ' + sd.reindex(c.index).map('{:.3f}'.format))

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 300)
    n = held.groupby(['sources', 'augmentation'])['run'].nunique()
    out(f'{runs["run"].nunique()} runs, {n.min()}-{n.max()} per source and configuration; held-out set: '
        f'{len(held_wf)} waveforms, {held_wf["subject"].nunique()} participants')
    out('\n=== HELD-OUT SET: median AUROC over runs ===')
    out(held.groupby(['augmentation', 'sources'])['auc'].median().map('{:.3f}'.format).unstack('sources')
        .reindex(index=order, columns=sources).rename(columns=SOURCES).fillna('-').to_string())
    out('\n=== HELD-OUT SET: mean AUROC over runs (95% participant-bootstrap CI), SD over runs ===')
    out(cells.unstack('sources').reindex(index=order, columns=sources).rename(columns=SOURCES).fillna('-').to_string())
    not_tuned = [s for s in PAPER_HUMAN_SETTINGS if s not in TUNE_SETTINGS[ARCH]['tune_human']]
    if not_tuned:
        out('\nHuman sources not tuned here, in selected.json as in the paper (Supplementary Table 6): '
            + '; '.join(f'{SOURCES[s]} {PAPER_HUMAN_SETTINGS[s]}' for s in not_tuned))

    (RESULTS_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
