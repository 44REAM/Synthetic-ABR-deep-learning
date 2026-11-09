"""Sensitivity of synthetic-trained models to simulated impairment.

The EEGConformer runs of experiments/multi_arch.py trained on Roenne, Verhulst,
Roenne+Verhulst, MixUp, CutMix and the matched Gaussian control (64 development subjects, tuned
augmentation, 30 runs each)
predict the impairment sweeps (src/sensitivity_roenne.py: cOHC and cIHC; and
src/sensitivity_verhulst.py: synaptopathy and flat OHC loss; one simulated subject per
setting, 0-90 dB in 10 dB steps, no noise). Each setting is normalised like a training
subject (min-max over its levels). The normal Roenne setting (cOHC = cIHC = 1) belongs to
both Roenne sweeps.
Reported per sweep setting:
  - P(wave V absent) averaged over the setting's levels: mean ± SD over the 30 runs of each
    training source
  - per level for PER_LEVEL_SOURCE (mean over runs), next to the wave V amplitude measured
    in the simulation (largest value in 4-12 ms)
Writes predictions.csv and summary.txt into results/sensitivity/.

Run from the repository root:
  python analysis/sensitivity.py
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

from src.data import DATA_ROOT, load_synthetic_folder
from src.predict import predict_runs

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RUNS_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
ARCH = 'eegconformer'
N_DEV = 64
AUGMENTATION = 'tuned'
# Training sources, in display order
SOURCES = {'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'mixup': 'MixUp', 'cutmix': 'CutMix', 'gaussian': 'Matched Gaussian'}
SWEEPS = {'roenne': DATA_ROOT / 'sensitivity_roenne', 'verhulst': DATA_ROOT / 'sensitivity_verhulst'}
PER_LEVEL_SOURCE = 'roenne+verhulst'
OUT_DIR = REPOSITORY_ROOT / 'results' / 'sensitivity'


def selected_runs() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((RUNS_DIR / 'runs').glob('*.json'))]
    runs = pd.DataFrame(rows)
    runs = runs[(runs['arch'] == ARCH) & (runs['n_dev'] == N_DEV) & (runs['augmentation'] == AUGMENTATION)]
    runs = runs.assign(sources=runs['sources'].map(lambda s: '+'.join(s) if isinstance(s, list) else s))
    return runs[runs['sources'].isin(SOURCES)][['run', 'sources', 'cv']]


def sweep_rows(model: str, labels: pd.DataFrame) -> pd.DataFrame:
    """(index, sweep, setting, order) of every record of a sweep dataset; order runs from
    normal to most impaired."""
    rows = []
    for i, r in enumerate(labels.to_dict('records')):
        if model == 'roenne':
            if r['cihc'] == 1.0:
                rows.append((i, 'OHC loss (cOHC)', f'cOHC {r["cohc"]:.1f}', -r['cohc']))
            if r['cohc'] == 1.0:
                rows.append((i, 'IHC loss (cIHC)', f'cIHC {r["cihc"]:.1f}', -r['cihc']))
        elif r['hearing_profile'] == 'Normal':
            rows.append((i, 'synaptopathy', f'numH/M/L {r["num_h"]}/{r["num_m"]}/{r["num_l"]}', r['record_id']))
        else:
            rows.append((i, 'OHC loss (flat profile)', r['hearing_profile'], r['record_id']))
    return pd.DataFrame(rows, columns=['index', 'sweep', 'setting', 'order'])


def main() -> None:
    runs = selected_runs()
    n = runs.groupby('sources')['run'].nunique()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for sweep, folder in SWEEPS.items():
        records, labels = load_synthetic_folder(folder, f'sensitivity_{sweep}')
        preds = predict_runs(RUNS_DIR, {sweep: records}, runs=runs['run'].tolist())
        info = sweep_rows(sweep, labels).merge(
            labels[['intensity_db', 'wave_v_amplitude_uv']].assign(index=range(len(labels))), on='index')
        frames.append(preds.merge(runs, on='run').merge(info.assign(sweep_model=sweep), on='index'))
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(OUT_DIR / 'predictions.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    out(f'{ARCH}, n = {N_DEV}, augmentation {AUGMENTATION}; runs per source: '
        + ', '.join(f'{SOURCES[s]} {n.get(s, 0)}' for s in SOURCES))
    for (sweep_model, sweep), d in preds.groupby(['sweep_model', 'sweep'], sort=False):
        order = list(dict.fromkeys(d.sort_values('order')['setting']))
        per_run = d.groupby(['sources', 'run', 'setting'])['prob_absent'].mean()
        stats = per_run.groupby(['setting', 'sources']).agg(['mean', 'std'])
        cells = (stats['mean'].map('{:.3f}'.format) + ' ± ' + stats['std'].map('{:.3f}'.format)).unstack('sources')
        out(f'\n=== {sweep_model.upper()} SWEEP, {sweep}: P(wave V absent) averaged over levels, '
            f'mean ± SD over runs ===')
        out(cells.reindex(index=order, columns=list(SOURCES)).rename(columns=SOURCES).to_string())

        level = d[d['sources'] == PER_LEVEL_SOURCE].groupby(['setting', 'intensity_db'])['prob_absent'].mean()
        out(f'\n{SOURCES[PER_LEVEL_SOURCE]}: P(wave V absent) per level (dB), mean over runs')
        out(level.unstack('intensity_db').reindex(order).rename(columns=lambda v: f'{v:g}')
            .to_string(float_format='{:.3f}'.format))
        amp = d.drop_duplicates(['setting', 'intensity_db']).set_index(['setting', 'intensity_db'])['wave_v_amplitude_uv']
        out('\nMeasured wave V amplitude (µV) per level (dB)')
        out(amp.unstack('intensity_db').reindex(order).rename(columns=lambda v: f'{v:g}')
            .to_string(float_format='{:.3f}'.format))

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
