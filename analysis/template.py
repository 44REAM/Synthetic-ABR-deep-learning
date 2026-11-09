"""Conventional detector: cross-correlation with a template.

Template: for each training source and run, the mean of the training subjects' waveforms at
their highest level labelled wave V present (PhysioNet: per tone-burst frequency; subjects
without such a level contribute nothing), as preprocessed for the networks, without time
alignment. The training subjects are those of the experiments/multi_arch.py run with the same
cv, so every template matches a network run; real-data sources give normative templates from
the recordings of that dataset.
Score (Aoyagi et al. 1991, Audiology 30:266-274; Elberling 1979, Scand Audiol 8:187-190):
whole 0-15 ms waveforms, mean removed and Hanning-tapered, cross-correlated by FFT and
normalised to a correlation coefficient at every lag (-7.5 to +7.5 ms); RM, its maximum, is
the score (higher = wave V present; AUROC uses -RM, positive class = wave V absent), and DL,
the lag of the maximum (> 0: test waveform later than the template), is kept in scores.csv.
The test waveforms and preprocessing are those of the networks (src/data.py).

Reported: AUROC of the templates (mean over runs) with its 95% participant-bootstrap CI
(analysis/bootstrap.py), on the test sets (all waveforms and near threshold) and the held-out
set. Writes templates.npz, scores.csv, bootstrap_auroc_ci.csv and
summary.txt into results/template/.

Run from the repository root:
  python analysis/template.py
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
from torch.utils.data import ConcatDataset

from analysis.bootstrap import bootstrap_auroc
from analysis.metrics import near_threshold_ids
from experiments.multi_arch import N_DEV_HUMAN, N_DEV_SYNTHETIC, N_RUNS
from src.data import (AUGDEV_SET, HELD_OUT_SOTON_SUBJECTS, HUMAN_SOURCES, TARGET_RATE, build_training_data,
                      load_human_datasets, normalise)
from src.run import LABEL_RULE, load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NETWORK_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
OUT_DIR = REPOSITORY_ROOT / 'results' / 'template'
# Training sources with a template (as in experiments/multi_arch.py), display names
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet',
           'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'gaussian': 'Control: Gaussian'}
ARCHS = {'eegconformer': 'EEGConformer', 'deep4': 'DeepConvNet', 'atcnet': 'ATCNet',
         'shallow': 'ShallowConvNet', 'eegnet': 'EEGNet', 'cwt_vit': 'CWT-ViT'}
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
ALPHA = 0.05


def evaluation_waveforms() -> dict[str, dict[str, list]]:
    """Raw and normalised records of every evaluation set, in the order of src.data.load_eval_sets."""
    by_source = load_human_datasets()
    raw = {
        'wang': by_source['wang'],
        'southampton': [r for r in by_source['southampton'] if r['subject'] not in HELD_OUT_SOTON_SUBJECTS],
        'physionet': by_source['physionet'],
        AUGDEV_SET: [r for r in by_source['southampton'] if r['subject'] in HELD_OUT_SOTON_SUBJECTS],
    }
    return {'raw': raw, 'normalised': {name: normalise(recs) for name, recs in raw.items()}}


def taper(x: np.ndarray) -> np.ndarray:
    """Mean removed and Hanning-tapered, along the last axis."""
    x = x - x.mean(axis=-1, keepdims=True)
    return x * np.hanning(x.shape[-1])


def cross_correlation(tests: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RM (largest correlation coefficient over all circular lags) and DL (its lag, ms) per test row."""
    a, b = taper(tests), taper(template[None, :])
    n = a.shape[1]
    cc = np.fft.irfft(np.fft.rfft(a, axis=1) * np.conj(np.fft.rfft(b, axis=1)), n=n, axis=1)
    cc /= np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b) + 1e-12
    lags = np.arange(n)
    lags = np.where(lags < n / 2, lags, lags - n)            # cc[:, k]: test later than template by lag k
    best = cc.argmax(axis=1)
    return cc[np.arange(len(cc)), best], lags[best] * 1000.0 / TARGET_RATE


def template_of(source: str, cv: int) -> tuple[np.ndarray, int]:
    """Template of one run and the number of waveforms averaged."""
    sources = tuple(source.split('+'))
    n_dev = N_DEV_HUMAN if source in HUMAN_SOURCES else N_DEV_SYNTHETIC
    train_ds, _, _ = build_training_data(sources, n_dev, cv, LABEL_RULE, {})
    parts = train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds]
    records = pd.DataFrame([{'subject': r['subject'], 'freq': r.get('header', {}).get('freq_khz'),
                             'intensity': r['intensity'], 'label': r['label'], 'wave': r['waveform_proc']}
                            for p in parts for r in p.records])
    present = records[records['label'] == 0]
    top = present.loc[present.groupby(['subject', 'freq'], dropna=False)['intensity'].idxmax()]
    return np.mean(np.stack(top['wave'].to_list()), axis=0), len(top)


def main() -> None:
    _, _, waveforms = load_results(NETWORK_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tests = evaluation_waveforms()['normalised']
    test_x = {}
    for eval_set, wf in waveforms.groupby('eval_set', sort=False):
        wf = wf.sort_values('index_in_set')
        recs = tests[eval_set]
        if len(recs) != len(wf) or [int(r['label']) for r in recs] != wf['label_absent'].tolist():
            raise ValueError(f'{eval_set}: records do not match waveforms.csv')
        test_x[eval_set] = (wf['waveform_id'].to_numpy(), np.stack([r['waveform_proc'] for r in recs]))

    templates, rows, score_frames = {}, [], []
    for source in SOURCES:
        sets = [s for s in HUMAN_SOURCES if s != source] + [AUGDEV_SET]
        for cv in range(N_RUNS):
            template, n_waves = template_of(source, cv)
            run = f'template_{source}_cv{cv}'
            templates[run] = template
            rows.append({'run': run, 'arch': 'template', 'sources': source, 'cv': cv, 'n_waveforms': n_waves})
            for s in sets:
                ids, x = test_x[s]
                rm, dl = cross_correlation(x, template)
                score_frames.append(pd.DataFrame({'run': run, 'waveform_id': ids, 'rm': rm, 'dl_ms': dl,
                                                  'prob_absent': -rm}))
    det_runs = pd.DataFrame(rows)
    scores = pd.concat(score_frames, ignore_index=True)
    np.savez_compressed(OUT_DIR / 'templates.npz', **templates)
    scores.to_csv(OUT_DIR / 'scores.csv', index=False, float_format='%.6g')

    # The participant draws do not depend on the runs, so these CIs equal those computed next to the networks
    test_wf = waveforms[waveforms['eval_set'].isin(TEST_SETS)]
    test_ci = bootstrap_auroc(det_runs, scores, test_wf, ['arch', 'sources'],
                              {'all': None, 'near': near_threshold_ids(test_wf)})
    held_ci = bootstrap_auroc(det_runs, scores, waveforms[waveforms['eval_set'] == AUGDEV_SET], ['arch', 'sources'],
                              {'all': None})
    ci = pd.concat([test_ci, held_ci], ignore_index=True)
    ci.to_csv(OUT_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    n = det_runs.groupby('sources')['n_waveforms'].agg(['min', 'max'])
    out('Template waveforms per run (min-max over runs): '
        + '; '.join(f'{SOURCES[s]} {n.loc[s, "min"]}-{n.loc[s, "max"]}' for s in SOURCES))
    cell = (ci['auroc'].map('{:.3f}'.format) + ' (' + ci['ci_low'].map('{:.3f}'.format) + '-'
            + ci['ci_high'].map('{:.3f}'.format) + ')')
    table = ci.assign(cell=cell, column=ci['eval_set'].map({**TEST_SETS, AUGDEV_SET: 'Held-out'})
                      + np.where(ci['subset'] == 'near', ' (near threshold)', ''))
    columns = list(TEST_SETS.values()) + [f'{v} (near threshold)' for v in TEST_SETS.values()] + ['Held-out']
    out(f'\n=== TEMPLATE CROSS-CORRELATION (RM, no training): AUROC, mean over {N_RUNS} runs '
        f'(95% participant-bootstrap CI) ===')
    out(table.pivot_table(index='sources', columns='column', values='cell', aggfunc='first')
        .reindex(index=list(SOURCES), columns=columns).rename(index=SOURCES).fillna('-').to_string())

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
