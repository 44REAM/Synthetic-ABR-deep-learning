"""PhysioNet results separately for the 1 kHz and 4 kHz tone bursts.

The runs of experiments/multi_arch.py (every architecture, tuned augmentation, 30 runs per
training source) on the PhysioNet test set, split by stimulus frequency. Per frequency: AUROC on all
waveforms of that frequency and near threshold (levels where both labels occur at that
frequency), mean over runs with its 95% participant-bootstrap CI (analysis/bootstrap.py). The
conventional detectors are included as archs 'template' (analysis/template.py), 'adjacent' and
'template_adjacent' (analysis/adjacent.py; run both first). The pooled PhysioNet AUROC is read
from results/multi_arch/bootstrap_auroc_ci.csv (detectors: results/template/, results/adjacent/)
for comparison.
Also the label counts per frequency, and every synthetic source and control minus the Mendeley-
and Southampton-trained models of the same network at each frequency (AUROC, all waveforms):
paired participant-bootstrap difference with 95% CI and two-sided p, Benjamini-Hochberg q over
these comparisons as one family (* q < 0.05), as in analysis/multi_arch.py; the same
for the conventional detectors (analysis.adjacent.versus_real) as a family of their own.
Writes bootstrap_auroc_ci.csv, paired_diff.csv, paired_diff_detectors.csv and summary.txt into
results/physionet_frequency/.

Run from the repository root:
  python analysis/physionet_frequency.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from analysis.adjacent import versus_real
from analysis.bootstrap import benjamini_hochberg, bootstrap_auroc, paired_difference
from analysis.metrics import near_threshold_ids, physionet_by_frequency
from analysis.multi_arch import ALPHA, HUMAN
from analysis.threshold_metrics import DETECTOR_DIRS, DETECTORS, detector_runs
from src.run import load_results

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NETWORK_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
OUT_DIR = REPOSITORY_ROOT / 'results' / 'physionet_frequency'
# Display order and names
ARCHS = {'eegconformer': 'EEGConformer', 'deep4': 'DeepConvNet', 'atcnet': 'ATCNet',
         'shallow': 'ShallowConvNet', 'eegnet': 'EEGNet', 'cwt_vit': 'CWT-ViT', **DETECTORS}
# Training sources tested on PhysioNet, in display order
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'roenne': 'Roenne', 'verhulst': 'Verhulst',
           'roenne+verhulst': 'Roenne+Verhulst', 'mixup': 'MixUp', 'cutmix': 'CutMix',
           'gaussian': 'Control: Gaussian'}
FREQUENCIES_KHZ = [1, 4]
FREQ_SETS = [f'{f} kHz' for f in FREQUENCIES_KHZ]      # eval_set of physionet_by_frequency


def synthetic_vs_real(ci: pd.DataFrame, draws: dict) -> pd.DataFrame:
    """Paired differences of every synthetic source and control against the two real-data
    baselines tested on PhysioNet, per network and frequency, BH over all of them."""
    point = ci[ci['subset'] == 'all'].set_index(['arch', 'sources', 'eval_set'])['auroc']
    rows = []
    for arch in (a for a in ARCHS if a not in DETECTORS):
        for source in (s for s in SOURCES if s not in HUMAN):
            for baseline in (s for s in SOURCES if s in HUMAN):
                for eval_set in FREQ_SETS:
                    a, b = ((arch, source), eval_set, 'all'), ((arch, baseline), eval_set, 'all')
                    rows.append({'arch': arch, 'source': source, 'baseline': baseline, 'eval_set': eval_set,
                                 **paired_difference(point[(arch, source, eval_set)],
                                                     point[(arch, baseline, eval_set)], draws[a], draws[b])})
    diffs = pd.DataFrame(rows)
    diffs['q'] = benjamini_hochberg(diffs['p'].to_numpy())
    return diffs


def main() -> None:
    runs, preds, waveforms = load_results(NETWORK_DIR)
    # The detectors as runs next to the networks (the participant draws do not depend on the runs)
    det_runs, det_preds = detector_runs()
    runs = pd.concat([runs[['run', 'arch', 'sources']], det_runs[['run', 'arch', 'sources']]], ignore_index=True)
    preds = pd.concat([preds[['run', 'waveform_id', 'prob_absent']], det_preds], ignore_index=True)
    runs = runs[runs['arch'].isin(ARCHS) & runs['sources'].isin(SOURCES)]
    preds = preds[preds['run'].isin(runs['run'])]
    # One evaluation set per frequency, so near threshold is defined within each frequency
    by_freq = physionet_by_frequency(waveforms)
    ci, draws = bootstrap_auroc(runs, preds, by_freq, ['arch', 'sources'],
                                {'all': None, 'near': near_threshold_ids(by_freq)}, return_draws=True)
    diffs = synthetic_vs_real(ci, draws)
    detector_diffs = versus_real(ci[ci['subset'] == 'all'].set_index(['arch', 'sources', 'eval_set'])['auroc'],
                                 draws, FREQ_SETS)
    pooled = pd.concat([pd.read_csv(d / 'bootstrap_auroc_ci.csv') for d in (NETWORK_DIR, *DETECTOR_DIRS)],
                       ignore_index=True)
    pooled = pooled[pooled['arch'].isin(ARCHS) & (pooled['eval_set'] == 'physionet') & pooled['sources'].isin(SOURCES)]
    ci = pd.concat([pooled.assign(eval_set='pooled'), ci], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ci.to_csv(OUT_DIR / 'bootstrap_auroc_ci.csv', index=False, float_format='%.6g')
    diffs.to_csv(OUT_DIR / 'paired_diff.csv', index=False, float_format='%.6g')
    detector_diffs.to_csv(OUT_DIR / 'paired_diff_detectors.csv', index=False, float_format='%.6g')

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    counts = by_freq.groupby('eval_set').agg(waveforms=('label_absent', 'size'),
                                             present=('label_absent', lambda a: int((a == 0).sum())),
                                             participants=('subject', 'nunique'))
    near = by_freq[by_freq['waveform_id'].isin(near_threshold_ids(by_freq))]
    counts['near_threshold_levels'] = near.groupby('eval_set')['intensity_db'].agg(
        lambda v: f'{v.min():g}-{v.max():g} dB peSPL')
    counts['near_threshold_waveforms'] = near.groupby('eval_set').size()
    out('PhysioNet labels per stimulus frequency')
    out(counts.to_string())

    cell = (ci['auroc'].map('{:.3f}'.format) + ' (' + ci['ci_low'].map('{:.3f}'.format) + '-'
            + ci['ci_high'].map('{:.3f}'.format) + ')')
    table = ci.assign(cell=cell, column=ci['eval_set'] + ', ' + ci['subset'])
    columns = [f'{s}, {sub}' for s in ['pooled'] + [f'{f} kHz' for f in FREQUENCIES_KHZ] for sub in ('all', 'near')]
    out('\n=== PhysioNet AUROC by stimulus frequency, mean over runs (95% participant-bootstrap CI); '
        'near = levels where both labels occur (within the frequency) ===')
    for arch, arch_name in ARCHS.items():
        out(f'\n{arch_name}')
        out(table[table['arch'] == arch].pivot_table(index='sources', columns='column', values='cell', aggfunc='first')
            .reindex(index=list(SOURCES), columns=columns).rename(index=SOURCES).fillna('-').to_string())

    for d, kind in ((diffs, 'NETWORKS'), (detector_diffs, 'CONVENTIONAL DETECTORS')):
        sig = d['q'] < ALPHA
        out(f'\n=== {kind}: SYNTHETIC / CONTROL MINUS REAL-DATA BASELINE per frequency: AUROC difference (95% '
            f'paired participant-bootstrap CI), * BH q < {ALPHA} over all {len(d)} comparisons ===')
        out(f'  higher than the baseline in {(d["diff"] > 0).sum()} of {len(d)}; significantly higher in '
            f'{(sig & (d["diff"] > 0)).sum()}, significantly lower in {(sig & (d["diff"] < 0)).sum()}')
        cells = (d['diff'].map('{:+.3f}'.format) + ' (' + d['ci_low'].map('{:+.3f}'.format) + ', '
                 + d['ci_high'].map('{:+.3f}'.format) + ')' + sig.map({True: '*', False: ' '}))
        wide = d.assign(cell=cells, column='vs ' + d['baseline'].map(SOURCES) + ' on ' + d['eval_set']
                        ).pivot_table(index=['arch', 'source'], columns='column', values='cell', aggfunc='first')
        columns = [f'vs {SOURCES[b]} on {f}' for b in SOURCES if b in HUMAN for f in FREQ_SETS]
        for arch in d['arch'].unique():
            out(f'\n{ARCHS[arch]}')
            out(wide.loc[arch].reindex(index=[s for s in SOURCES if s not in HUMAN and s in wide.loc[arch].index],
                                       columns=columns).rename(index=SOURCES).to_string())

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
