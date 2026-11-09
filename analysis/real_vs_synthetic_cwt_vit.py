"""The comparison of analysis/real_vs_synthetic.py (8 development subjects per source) for CWT-ViT,
from runs of other experiments (no training of its own):
  noaug  experiments/tune_augmentation.py (CWT-ViT), configuration 'none': the three real
         datasets, Roenne, Verhulst and the matched Gaussian control, 10 runs each
  aug    synthetic and control: the tuning runs of each source's selected configuration
         (results/tune_augmentation_cwt_vit/selected.json), 10 runs each; real data: the
         CWT-ViT runs of experiments/multi_arch.py (8 development subjects, tuned augmentation),
         30 runs each
These are the sources CWT-ViT was trained on with 8 development subjects.

Per setting:
  test sets           AUROC on all waveforms and at mixed-label levels: mean over runs with its 95%
                      participant-bootstrap CI and the SD over runs; '-' where a real-data model
                      would be tested on its own dataset
  synthetic - real    every synthetic and control source minus every real-data model on each
                      test set the real-data model is tested on (AUROC, all waveforms): paired
                      participant-bootstrap difference with its 95% CI, Benjamini-Hochberg q over
                      all comparisons of the setting (* q < 0.05), as in analysis/real_vs_synthetic.py
Writes per_run_metrics.csv, bootstrap_auroc_ci.csv, paired_diff.csv and summary.txt into
results/real_vs_synthetic_cwt_vit/<setting>/.

Run from the repository root:
  python analysis/real_vs_synthetic_cwt_vit.py
"""

from __future__ import annotations

import json
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
from analysis.real_vs_synthetic import SOURCES, TEST_SETS, ci_cells, sizes_line

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS = REPOSITORY_ROOT / 'results'
TUNE_DIR = RESULTS / 'tune_augmentation_cwt_vit'
MULTI_DIR = RESULTS / 'multi_arch'
OUT_DIR = RESULTS / 'real_vs_synthetic_cwt_vit'
ARCH = 'cwt_vit'
N_DEV = 8
SYNTHETIC = ['roenne', 'verhulst', 'gaussian']
SELECTED = json.loads((TUNE_DIR / 'selected.json').read_text())


def config_name(setting: dict) -> str:
    """Tuning configuration of a one-augmentation setting, e.g. {'pink_noise_sigma': 0.2} -> 'pink0.2'."""
    (key, value), = ((k, v) for k, v in setting.items() if k != 'noisy_validation')
    return {'shift_limit_ms': 'shift', 'pink_noise_sigma': 'pink'}[key] + f'{value:g}'


# Per setting: (folder, run filter)
SETTINGS = {
    'noaug': [(TUNE_DIR, lambda r: r['augmentation'] == 'none' and r['sources'] in HUMAN + SYNTHETIC)],
    'aug': [(TUNE_DIR, lambda r: r['sources'] in SYNTHETIC
             and r['augmentation'] == config_name(SELECTED[r['sources']])),
            (MULTI_DIR, lambda r: r['arch'] == ARCH and r['n_dev'] == N_DEV and r['sources'] in HUMAN)],
}


def load(parts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, predictions) of the runs that pass each folder's filter."""
    runs, preds = [], []
    for folder, keep in parts:
        r = pd.DataFrame([x for x in (json.loads(p.read_text()) for p in sorted((folder / 'runs').glob('*.json')))
                          if keep(x)])
        runs.append(r)
        preds += [pd.read_csv(folder / 'predictions' / f'{run}.csv.gz') for run in r['run']]
    runs = pd.concat(runs, ignore_index=True)
    assert runs['run'].is_unique and (runs['n_dev'] == N_DEV).all() and (runs['arch'] == ARCH).all()
    return runs, pd.concat(preds, ignore_index=True)


def main() -> None:
    waveforms = pd.read_csv(TUNE_DIR / 'waveforms.csv')
    assert waveforms.equals(pd.read_csv(MULTI_DIR / 'waveforms.csv'))
    test_wf = waveforms[waveforms['eval_set'].isin(TEST_SETS)]
    near = near_threshold_ids(test_wf)
    pd.set_option('display.width', 250)
    for setting, parts in SETTINGS.items():
        out_dir = OUT_DIR / setting
        out_dir.mkdir(parents=True, exist_ok=True)
        runs, preds = load(parts)
        metrics = per_run_metrics(runs, preds, waveforms)
        metrics.to_csv(out_dir / 'per_run_metrics.csv', index=False, float_format='%.6g')
        ci, draws = bootstrap_auroc(runs, preds, test_wf, ['arch', 'sources'], {'all': None, 'near': near},
                                    return_draws=True)
        ci.to_csv(out_dir / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')
        order = [s for s in SOURCES if s in set(runs['sources'])]
        diffs = synthetic_vs_human(ci, draws, [ARCH], order)
        diffs.to_csv(out_dir / 'paired_diff.csv', index=False, float_format='%.6g')
        sd = metrics.groupby(['sources', 'eval_set'])[['auc', 'auc_near']].std()

        first = runs.groupby('sources')
        lines = [f'CWT-ViT, {setting}, {N_DEV} development subjects per source',
                 '  runs: ' + ', '.join(f'{SOURCES[s]} {first.size()[s]}' for s in order),
                 '  augmentation: ' + ', '.join(f'{SOURCES[s]} {first["augmentation"].first()[s]}' for s in order)]
        for subset, title, sd_col in [('all', 'all waveforms', 'auc'), ('near', 'mixed-label levels', 'auc_near')]:
            c = ci[ci['subset'] == subset]
            lines += [f'\n=== TEST DATASETS: AUROC, {title}, mean over runs (95% participant-bootstrap CI), '
                      f'SD over runs ===', sizes_line(c, TEST_SETS),
                      ci_cells(c, sd[sd_col]).unstack('eval_set').reindex(index=order, columns=list(TEST_SETS))
                      .rename(index=SOURCES, columns=TEST_SETS).fillna('-').to_string()]
        sig = diffs['q'] < ALPHA
        lines += [f'\n=== SYNTHETIC / CONTROL MINUS REAL-DATA MODEL: AUROC difference (95% paired participant-'
                  f'bootstrap CI), * BH q < {ALPHA} over all {len(diffs)} comparisons ===',
                  f'  higher in {(diffs["diff"] > 0).sum()} of {len(diffs)}; significantly higher in '
                  f'{(sig & (diffs["diff"] > 0)).sum()}, significantly lower in {(sig & (diffs["diff"] < 0)).sum()}']
        cells = (diffs['diff'].map('{:+.3f}'.format) + ' (' + diffs['ci_low'].map('{:+.3f}'.format) + ', '
                 + diffs['ci_high'].map('{:+.3f}'.format) + ')' + sig.map({True: '*', False: ' '}))
        column = 'vs ' + diffs['baseline'].map(TEST_SETS) + ' on ' + diffs['eval_set'].map(TEST_SETS)
        columns = [f'vs {TEST_SETS[b]} on {TEST_SETS[t]}' for b in HUMAN for t in TEST_SETS if t != b]
        wide = diffs.assign(cell=cells, column=column).pivot_table(index='source', columns='column', values='cell',
                                                                   aggfunc='first')
        lines.append(wide.reindex(index=[s for s in order if s not in HUMAN], columns=columns)
                     .rename(index=SOURCES).to_string())
        text = '\n'.join(lines)
        print(text + '\n', flush=True)
        (out_dir / 'summary.txt').write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
