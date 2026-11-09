"""Figure of analysis/multi_arch.py, analysis/physionet_frequency.py, analysis/template.py and
analysis/adjacent.py: AUROC of every training source on every architecture, one heatmap
per test set.

Panel a: the three test sets; panel b: the PhysioNet test set split by tone-burst frequency
(1 and 4 kHz), without the PhysioNet-trained row, which is not tested on PhysioNet.
Rows: training sources (real datasets 8 development subjects; synthetic, mixing and control 64);
columns: the six architectures, followed by the conventional detectors built from the same
source (no MixUp/CutMix template): template cross-correlation, adjacent-level correlation and
their average. Cell colour and number: mean
AUROC over the 30 runs (the 95% participant-bootstrap CIs are in the analysis summaries).
Letters under a number name the real-data baselines (M Mendeley, S Southampton,
P PhysioNet) that the source exceeds significantly on that test set, with the same network or
detector (paired participant-bootstrap difference, Benjamini-Hochberg q < 0.05 over all 216
network comparisons of panel a, all 144 of panel b, and separately all 72 and 48 detector
comparisons, as in the analyses); a minus sign marks a significantly lower
AUROC. Real-data sources are not tested on their own dataset (blank cells).
Writes multi_arch.png (all waveforms) and multi_arch_near.png (levels at which both labels
occur, within each frequency in panel b; no significance letters), each also as .pdf, into
results/multi_arch/figures/.

Run from the repository root:
  python figures/multi_arch.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'pdf.fonttype': 42})   # Times New Roman, embedded as TrueType
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.multi_arch import ALPHA, ARCHS, HUMAN, SOURCES, TEST_SETS

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MULTI_ARCH_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
TEMPLATE_DIR = REPOSITORY_ROOT / 'results' / 'template'
ADJACENT_DIR = REPOSITORY_ROOT / 'results' / 'adjacent'
FREQUENCY_DIR = REPOSITORY_ROOT / 'results' / 'physionet_frequency'
OUT_DIR = MULTI_ARCH_DIR / 'figures'
DPI = 500
FIG_WIDTH = 7.2                      # inches, double column
SUBSETS = {'all': 'multi_arch', 'near': 'multi_arch_near'}      # subset -> output stem
DETECTORS = {'template': 'Template\n(no training)', 'adjacent': 'Adjacent\n(no training)',
             'template_adjacent': 'Template +\nadjacent'}       # column keys of the conventional detectors
COLUMNS = {**ARCHS, **DETECTORS}
SHORT = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet', 'roenne': 'Rønne',
         'verhulst': 'Verhulst', 'roenne+verhulst': 'Rønne + Verhulst', 'mixup': 'MixUp', 'cutmix': 'CutMix',
         'gaussian': 'Matched Gaussian'}       # the control name of Table 1
INITIAL = {'wang': 'M', 'southampton': 'S', 'physionet': 'P'}
ROWS = list(SOURCES)
# Row indices after which a separator is drawn (human | synthetic, MixUp and CutMix included | control)
GROUP_ENDS = [3, 8]
# Panel b: the eval_sets of analysis/physionet_frequency.py, without the PhysioNet-trained row
FREQ_SETS = {'1 kHz': 'PhysioNet 1 kHz', '4 kHz': 'PhysioNet 4 kHz'}
ROWS_B = [s for s in ROWS if s != 'physionet']
GROUP_ENDS_B = [2, 7]
# Layout in inches: heatmap row height, room for the titles and for the rotated column labels
LEFT, RIGHT = 0.16 * FIG_WIDTH, 0.9 * FIG_WIDTH
GAP = 0.08                           # between heatmaps, in heatmap widths
ROW_HEIGHT = 0.325
TITLE_HEIGHT = 0.5
LABEL_HEIGHT = 0.8
# Single-hue sequential scale, white to green (#1a9641);
# black text stays readable on its darkest cells
CMAP = LinearSegmentedColormap.from_list('auroc', ['#FFFFFF', '#1A9641'])
V_RANGE = (0.5, 1.0)                 # AUROC range of the colour scale; chance and below share the lightest colour
TEXT_COLOUR = 'black'
MISSING_COLOUR = '0.94'


def load(subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUROC and paired differences of both panels (their eval_set names do not overlap)."""
    ci = pd.concat([pd.read_csv(MULTI_ARCH_DIR / 'bootstrap_auroc_ci.csv'),
                    pd.read_csv(TEMPLATE_DIR / 'bootstrap_auroc_ci.csv'),
                    pd.read_csv(ADJACENT_DIR / 'bootstrap_auroc_ci.csv'),
                    pd.read_csv(FREQUENCY_DIR / 'bootstrap_auroc_ci.csv')], ignore_index=True)
    ci = ci[(ci['subset'] == subset) & ci['eval_set'].isin({**TEST_SETS, **FREQ_SETS})]
    diffs = (pd.concat([pd.read_csv(MULTI_ARCH_DIR / 'paired_diff.csv'),
                        pd.read_csv(FREQUENCY_DIR / 'paired_diff.csv'),
                        pd.read_csv(ADJACENT_DIR / 'paired_diff_vs_real.csv'),
                        pd.read_csv(FREQUENCY_DIR / 'paired_diff_detectors.csv')], ignore_index=True)
             if subset == 'all' else pd.DataFrame())
    return ci, diffs


def significance(diffs: pd.DataFrame, arch: str, source: str, eval_set: str) -> str:
    if diffs.empty:
        return ''
    d = diffs[(diffs['arch'] == arch) & (diffs['source'] == source) & (diffs['eval_set'] == eval_set)
              & (diffs['q'] < ALPHA)]
    d = d.set_index('baseline').reindex([h for h in HUMAN if h in set(d['baseline'])])
    return ' '.join(('' if r.diff > 0 else '−') + INITIAL[r.Index] for r in d.itertuples())


def panel(ax: plt.Axes, ci: pd.DataFrame, diffs: pd.DataFrame, eval_set: str, rows: list[str],
          group_ends: list[int], first: bool) -> plt.cm.ScalarMappable:
    table = (ci[ci['eval_set'] == eval_set].pivot(index='sources', columns='arch', values='auroc')
             .reindex(index=rows, columns=list(COLUMNS)))
    values = np.ma.masked_invalid(table.to_numpy())
    cmap = CMAP.copy()
    cmap.set_bad(MISSING_COLOUR)
    image = ax.imshow(values, cmap=cmap, vmin=V_RANGE[0], vmax=V_RANGE[1], aspect='auto')
    for i, source in enumerate(rows):
        for j, col in enumerate(COLUMNS):
            v = table.iat[i, j]
            if np.isnan(v):
                continue
            marks = significance(diffs, col, source, eval_set)
            ax.text(j, i - (0.16 if marks else 0), f'{v:.2f}', ha='center', va='center', fontsize=6, color=TEXT_COLOUR)
            if marks:
                ax.text(j, i + 0.22, marks, ha='center', va='center', fontsize=4.8, color=TEXT_COLOUR)
    n_arch = len(ARCHS)
    for k in group_ends:
        ax.axhline(k - 0.5, color='white', linewidth=1.8)
    ax.axvline(n_arch - 0.5, color='white', linewidth=2.5)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(list(COLUMNS.values()), fontsize=6, rotation=90)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SHORT[s] for s in rows] if first else [''] * len(rows), fontsize=6.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def figure(subset: str, stem: str) -> Path:
    ci, diffs = load(subset)
    panels = {'a': (TEST_SETS, ROWS, GROUP_ENDS), 'b': (FREQ_SETS, ROWS_B, GROUP_ENDS_B)}
    heights = {k: len(rows) * ROW_HEIGHT for k, (_, rows, _) in panels.items()}       # heatmap heights
    height = sum(TITLE_HEIGHT + h + LABEL_HEIGHT for h in heights.values())
    width = (RIGHT - LEFT) / (len(TEST_SETS) + (len(TEST_SETS) - 1) * GAP)          # one heatmap
    fig = plt.figure(figsize=(FIG_WIDTH, height))
    top = 0.0                            # inches from the top of the figure
    for letter, (sets, rows, group_ends) in panels.items():
        fig.text(0.012, 1 - (top + 0.05) / height, letter, fontsize=11, fontweight='bold', va='top')
        top += TITLE_HEIGHT
        h = heights[letter]
        for j, (eval_set, set_name) in enumerate(sets.items()):     # left-aligned, panel b under the first two
            ax = fig.add_axes(((LEFT + j * width * (1 + GAP)) / FIG_WIDTH, 1 - (top + h) / height,
                               width / FIG_WIDTH, h / height))
            image = panel(ax, ci, diffs, eval_set, rows, group_ends, first=j == 0)
            n = ci[ci['eval_set'] == eval_set].iloc[0]
            ax.set_title(f'Test: {set_name}\n({n.n_participants} participants, {n.n_waveforms} waveforms)',
                         fontsize=8)
        fig.text(0.015, 1 - (top + h / 2) / height, 'Training source', rotation=90, va='center', fontsize=8)
        top += h + LABEL_HEIGHT
    # Colour bar in the free third place of panel b
    h = heights['b']
    cax = fig.add_axes(((LEFT + 2 * width * (1 + GAP) + 0.35 * width) / FIG_WIDTH,
                        (LABEL_HEIGHT + 0.2 * h) / height, 0.09 / FIG_WIDTH, 0.6 * h / height))
    cbar = fig.colorbar(image, cax=cax, extend='min')
    cbar.set_label('Mean AUROC over 30 runs', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{stem}.{ext}', dpi=DPI)
    plt.close(fig)
    return OUT_DIR / f'{stem}.png'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for subset, stem in SUBSETS.items():
        print(figure(subset, stem))


if __name__ == '__main__':
    main()
