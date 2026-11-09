"""Conventional detectors that compare each waveform with the next higher level of its series.

Series: the waveforms of one participant (PhysioNet: one participant and tone-burst frequency)
at their stimulus levels. Adjacent-level correlation (adapted from Suthakar & Liberman 2019,
Hear Res 381:107782): each waveform against the waveform one level higher in
its series, both mean removed and Hanning-tapered as for the template (analysis/template.py),
normalised cross-correlation by FFT at lags of -1 to +2 ms (> 0: the lower level later),
maximum over these lags. It needs no training and no labels. The highest level of a series has
no higher neighbour and takes the template RM of the training source and run instead
(results/template/scores.csv), so both detectors exist per training source and run like the
template:
  adjacent           adjacent-level correlation; at the highest level the template RM
  template_adjacent  (template RM + adjacent-level correlation) / 2; at the highest level the RM
Score: higher = wave V present; AUROC uses -score (positive class = wave V absent).

Reported, as analysis/template.py: AUROC of both detectors (mean over runs) with its 95%
participant-bootstrap CI (analysis/bootstrap.py), on the test sets (all waveforms and near
threshold) and the held-out set. For all three conventional detectors (template, adjacent,
template_adjacent): every synthetic source and the matched Gaussian control minus each
real-data source on each test set both are tested on (paired participant-bootstrap difference,
as the networks in analysis/multi_arch.py), Benjamini-Hochberg q over these
comparisons as their own family (* q < 0.05).
Writes scores.csv, bootstrap_auroc_ci.csv, paired_diff_vs_real.csv and summary.txt into
results/adjacent/.

Run from the repository root (after analysis/template.py):
  python analysis/adjacent.py
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

from analysis.bootstrap import benjamini_hochberg, bootstrap_auroc, paired_difference
from analysis.metrics import near_threshold_ids
from analysis.template import ALPHA, SOURCES, TEST_SETS, evaluation_waveforms, taper
from experiments.multi_arch import N_RUNS
from src.data import AUGDEV_SET, HUMAN_SOURCES, TARGET_RATE
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NETWORK_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
TEMPLATE_DIR = REPOSITORY_ROOT / 'results' / 'template'
OUT_DIR = REPOSITORY_ROOT / 'results' / 'adjacent'
LAGS_MS = (-1.0, 2.0)                # lag window of the adjacent-level correlation
DETECTORS = {'adjacent': 'Adjacent-level correlation', 'template_adjacent': 'Template + adjacent'}
ALL_DETECTORS = {'template': 'Template', **DETECTORS}       # compared across training sources


def versus_real(point: pd.Series, draws: dict, sets: list[str]) -> pd.DataFrame:
    """Every synthetic source and control minus each real-data source, per conventional detector and
    evaluation set both are tested on (paired participant-bootstrap differences), BH over all of them."""
    rows = []
    for detector in ALL_DETECTORS:
        for source in (s for s in SOURCES if s not in HUMAN_SOURCES):
            for baseline in (s for s in SOURCES if s in HUMAN_SOURCES):
                for eval_set in sets:
                    if (detector, baseline, eval_set) not in point.index:
                        continue                          # real source on its own dataset
                    rows.append({'arch': detector, 'source': source, 'baseline': baseline, 'eval_set': eval_set,
                                 **paired_difference(point[(detector, source, eval_set)],
                                                     point[(detector, baseline, eval_set)],
                                                     draws[((detector, source), eval_set, 'all')],
                                                     draws[((detector, baseline), eval_set, 'all')])})
    diffs = pd.DataFrame(rows)
    diffs['q'] = benjamini_hochberg(diffs['p'].to_numpy())
    return diffs


def adjacent_correlation(x: np.ndarray, ref: np.ndarray) -> float:
    """Largest normalised cross-correlation of x with ref over LAGS_MS (lag > 0: x later than ref)."""
    a, b = taper(x[None, :])[0], taper(ref[None, :])[0]
    n = len(a)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=n)
    cc /= np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    lags = np.arange(n)
    lags = np.where(lags < n / 2, lags, lags - n) * 1000.0 / TARGET_RATE
    return float(cc[(lags >= LAGS_MS[0]) & (lags <= LAGS_MS[1])].max())


def adjacent_scores(waveforms: pd.DataFrame) -> pd.Series:
    """Adjacent-level correlation of every evaluation waveform (NaN at the highest level of its series)."""
    tests = evaluation_waveforms()['normalised']
    wave = {}
    for eval_set, wf in waveforms.groupby('eval_set', sort=False):
        wf = wf.sort_values('index_in_set')
        recs = tests[eval_set]
        if len(recs) != len(wf) or [int(r['label']) for r in recs] != wf['label_absent'].tolist():
            raise ValueError(f'{eval_set}: records do not match waveforms.csv')
        wave.update(zip(wf['waveform_id'], (np.asarray(r['waveform_proc'], dtype=np.float64) for r in recs)))
    series = waveforms['eval_set'] + '|' + waveforms['subject'].astype(str) + '|' + waveforms['freq_khz'].fillna(0).astype(str)
    out = {}
    for _, g in waveforms.assign(series=series).groupby('series'):
        g = g.sort_values('intensity_db')
        if g['intensity_db'].duplicated().any():
            raise ValueError('a series has two waveforms at one level')
        ids = g['waveform_id'].to_list()
        for k, i in enumerate(ids):
            out[i] = adjacent_correlation(wave[i], wave[ids[k + 1]]) if k + 1 < len(ids) else np.nan
    return pd.Series(out, name='r_adj')


def main() -> None:
    _, _, waveforms = load_results(NETWORK_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r_adj = adjacent_scores(waveforms)
    template = pd.read_csv(TEMPLATE_DIR / 'scores.csv')[['run', 'waveform_id', 'rm']]
    parts = template['run'].str.extract(r'^template_(?P<sources>.+)_cv(?P<cv>\d+)$')
    template = template.assign(sources=parts['sources'], cv=parts['cv'].astype(int))
    if set(template['sources']) != set(SOURCES) or template.groupby('sources')['cv'].nunique().ne(N_RUNS).any():
        raise ValueError('results/template/scores.csv does not cover every source and run; run analysis/template.py')

    t = template.merge(r_adj, left_on='waveform_id', right_index=True, how='left')
    top = t['r_adj'].isna()
    frames = [t.assign(arch='adjacent', score=np.where(top, t['rm'], t['r_adj'])),
              t.assign(arch='template_adjacent', score=np.where(top, t['rm'], (t['rm'] + t['r_adj']) / 2))]
    scores = pd.concat(frames, ignore_index=True)
    scores['run'] = scores['arch'] + '_' + scores['sources'] + '_cv' + scores['cv'].astype(str)
    scores['prob_absent'] = -scores['score']
    scores[['run', 'waveform_id', 'r_adj', 'rm', 'score', 'prob_absent']].to_csv(
        OUT_DIR / 'scores.csv', index=False, float_format='%.6g')
    det_runs = scores[['run', 'arch', 'sources', 'cv']].drop_duplicates('run')

    # The template too, for the comparisons across sources (the participant draws do not depend on the runs)
    template_runs = template[['run', 'sources']].drop_duplicates('run').assign(arch='template')
    all_runs = pd.concat([template_runs[['run', 'arch', 'sources']], det_runs[['run', 'arch', 'sources']]],
                         ignore_index=True)
    all_preds = pd.concat([template[['run', 'waveform_id']].assign(prob_absent=-template['rm']),
                           scores[['run', 'waveform_id', 'prob_absent']]], ignore_index=True)
    test_wf = waveforms[waveforms['eval_set'].isin(TEST_SETS)]
    test_ci, draws = bootstrap_auroc(all_runs, all_preds, test_wf, ['arch', 'sources'],
                                     {'all': None, 'near': near_threshold_ids(test_wf)}, return_draws=True)
    held_ci = bootstrap_auroc(det_runs, scores, waveforms[waveforms['eval_set'] == AUGDEV_SET], ['arch', 'sources'],
                              {'all': None})
    ci = pd.concat([test_ci[test_ci['arch'].isin(DETECTORS)], held_ci], ignore_index=True)
    ci.to_csv(OUT_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')

    point = test_ci[test_ci['subset'] == 'all'].set_index(['arch', 'sources', 'eval_set'])['auroc']
    vs_real = versus_real(point, draws, list(TEST_SETS))
    vs_real.to_csv(OUT_DIR / 'paired_diff_vs_real.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    n_top = int(r_adj.isna().sum())
    out(f'Adjacent-level correlation, lags {LAGS_MS[0]:+g} to {LAGS_MS[1]:+g} ms; {n_top} series (their highest '
        f'level takes the template RM)')
    reference = pd.read_csv(TEMPLATE_DIR / 'bootstrap_auroc_ci.csv')
    shown = pd.concat([reference, ci], ignore_index=True)
    cell = (shown['auroc'].map('{:.3f}'.format) + ' (' + shown['ci_low'].map('{:.3f}'.format) + '-'
            + shown['ci_high'].map('{:.3f}'.format) + ')')
    table = shown.assign(cell=cell, column=shown['eval_set'].map({**TEST_SETS, AUGDEV_SET: 'Held-out'})
                         + np.where(shown['subset'] == 'near', ' (near threshold)', ''))
    columns = list(TEST_SETS.values()) + [f'{v} (near threshold)' for v in TEST_SETS.values()] + ['Held-out']
    for arch, name in {'template': 'Template (analysis/template.py, for reference)', **DETECTORS}.items():
        out(f'\n=== {name.upper()}: AUROC, mean over {N_RUNS} runs (95% participant-bootstrap CI) ===')
        out(table[table['arch'] == arch].pivot_table(index='sources', columns='column', values='cell', aggfunc='first')
            .reindex(index=list(SOURCES), columns=columns).rename(index=SOURCES).fillna('-').to_string())

    sig = vs_real['q'] < ALPHA
    out(f'\n=== SYNTHETIC / CONTROL MINUS REAL-DATA SOURCE, per detector: AUROC difference (95% paired '
        f'participant-bootstrap CI), * BH q < {ALPHA} over all {len(vs_real)} comparisons ===')
    out(f'  higher than the real-data source in {(vs_real["diff"] > 0).sum()} of {len(vs_real)}; significantly '
        f'higher in {(sig & (vs_real["diff"] > 0)).sum()}, significantly lower in {(sig & (vs_real["diff"] < 0)).sum()}')
    cells = (vs_real['diff'].map('{:+.3f}'.format) + ' (' + vs_real['ci_low'].map('{:+.3f}'.format) + ', '
             + vs_real['ci_high'].map('{:+.3f}'.format) + ')' + sig.map({True: '*', False: ' '}))
    wide = vs_real.assign(cell=cells, column='vs ' + vs_real['baseline'].map(SOURCES) + ' on '
                          + vs_real['eval_set'].map(TEST_SETS)).pivot_table(
        index=['arch', 'source'], columns='column', values='cell', aggfunc='first')
    for detector, name in ALL_DETECTORS.items():
        out(f'\n{name}')
        out(wide.loc[detector].reindex(index=[s for s in SOURCES if s not in HUMAN_SOURCES])
            .rename(index=SOURCES).to_string())

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
