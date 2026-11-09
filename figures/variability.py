"""Figure of analysis/variability.py: how similar each source's waveforms are to the real datasets.
Into results/variability/figures/, as .png and .pdf.

  similarity    a, heatmap without augmentation; rows: the sources (real datasets, auditory models,
      Gaussian controls), columns: the real datasets. Each cell is the mean RM (largest normalised
      cross-correlation over all lags, mean removed, so the same before and after the subject-level
      min-max normalisation) over pairs of a source subject and a real subject at the highest level
      labelled present, from results/variability/correlation.csv. Outlined cells compare a real
      dataset with itself (pairs of a subject with itself left out).
      b, median power spectra (results/variability/spectra.csv) of every waveform labelled present
      (top) and absent (bottom), one panel per synthetic source or control except the
      latency-matched control, without (dotted) and with one draw of the tuned augmentation (solid), the four real datasets clean in grey;
      in dB relative to the peak of each source's clean present spectrum, so present and absent share a
      scale. Flat waveforms (no power) are not drawn.

Run from the repository root:
  python figures/variability.py
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
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IN_DIR = REPOSITORY_ROOT / 'results' / 'variability'
OUT_DIR = IN_DIR / 'figures'
DPI = 500
FIG_WIDTH, FIG_HEIGHT = 7.2, 7.25    # inches, double column
CMAP = plt.get_cmap('Blues')         # sequential, one hue: light = less similar
VMIN, VMAX = 0.5, 1.0
AUGMENTATION = 'none'                # the waveforms as generated or recorded
# Rows in groups (real datasets, auditory models, Gaussian controls), as Supplementary Table 2
ROW_GROUPS = [
    [('wang', 'Mendeley'), ('southampton', 'Southampton'), ('physionet_1', 'PhysioNet 1 kHz'),
     ('physionet_4', 'PhysioNet 4 kHz')],
    [('roenne', 'Rønne'), ('verhulst', 'Verhulst'), ('roenne+verhulst', 'Rønne+Verhulst')],
    [('gaussian', 'Matched Gaussian'), ('gaussian_latency', 'Latency-matched Gaussian'),
     ('simple_gaussian', 'Unmatched Gaussian')],
]
COLUMNS = [('wang', 'Mendeley'), ('southampton', 'Southampton'), ('physionet_1', 'PhysioNet\n1 kHz'),
           ('physionet_4', 'PhysioNet\n4 kHz')]
GROUP_GAP = 0.4                      # extra space between row groups, in rows
REAL = ['wang', 'southampton', 'physionet_1', 'physionet_4']
SPECTRUM_SOURCES = {'roenne': 'Rønne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Rønne+Verhulst',
                    'gaussian': 'Matched Gaussian', 'simple_gaussian': 'Unmatched Gaussian'}
SOURCE_COLOUR, REAL_COLOUR = '#2c5fa8', '0.7'
FREQ_MAX_HZ = 1500
DB_LIM = (-90, 3)


def inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    """Axes placed in inches from the lower left corner."""
    w, h = fig.get_size_inches()
    return fig.add_axes((left / w, bottom / h, width / w, height / h))


def heatmap(fig: plt.Figure, ax: plt.Axes, cax: plt.Axes) -> None:
    table = pd.read_csv(IN_DIR / 'correlation.csv').set_index(['source', 'augmentation'])
    rows, y = [], 0.0
    for group in ROW_GROUPS:
        for key, name in group:
            rows.append((key, name, y))
            y += 1.0
        y += GROUP_GAP
    norm = plt.Normalize(VMIN, VMAX)
    for key, _, y0 in rows:
        for j, (column, _) in enumerate(COLUMNS):
            v = float(table.loc[(key, AUGMENTATION), f'vs_{column}_rm'])
            ax.add_patch(Rectangle((j, y0), 1, 1, facecolor=CMAP(norm(v)), edgecolor='white', linewidth=1.5))
            ax.text(j + 0.5, y0 + 0.5, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if norm(v) > 0.55 else '0.15')
            if key == column:                         # a real dataset with itself
                ax.add_patch(Rectangle((j + 0.06, y0 + 0.06), 0.88, 0.88, fill=False, edgecolor='0.15',
                                       linewidth=0.8))
    ax.set_xlim(0, len(COLUMNS))
    ax.set_ylim(y - GROUP_GAP, 0)
    ax.set_xticks([j + 0.5 for j in range(len(COLUMNS))])
    ax.set_xticklabels([name for _, name in COLUMNS], fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks([y0 + 0.5 for _, _, y0 in rows])
    ax.set_yticklabels([name for _, name, _ in rows], fontsize=7.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cbar.set_label('Between-subject correlation', fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.outline.set_visible(False)


def spectra(axes: dict[str, list[plt.Axes]]) -> None:
    """axes: label ('present', 'absent') -> one panel per SPECTRUM_SOURCES."""
    s = pd.read_csv(IN_DIR / 'spectra.csv')
    s = s[s['freq_hz'] <= FREQ_MAX_HZ]
    curves = {key: (d['freq_hz'].to_numpy() / 1000.0, d['db'].to_numpy())
              for key, d in s.groupby(['source', 'label', 'augmentation'])}
    for label, row in axes.items():
        for ax, (source, name) in zip(row, SPECTRUM_SOURCES.items()):
            for real in REAL:                         # real datasets as tested, clean
                ax.plot(*curves[(real, label, 'none')], color=REAL_COLOUR, linewidth=0.8, zorder=2)
            x, y = curves[(source, label, 'none')]
            if np.isfinite(y).any():                  # flat waveforms (no power) are not drawn
                ax.plot(x, y, color=SOURCE_COLOUR, linewidth=0.9, linestyle=':', zorder=3)
            ax.plot(*curves[(source, label, 'tuned')], color=SOURCE_COLOUR, linewidth=1.4, zorder=4)
            ax.set_xlim(0, FREQ_MAX_HZ / 1000.0)
            ax.set_ylim(*DB_LIM)
            ax.set_yticks([0, -30, -60, -90])
            ax.set_xticks([0, 0.5, 1, 1.5])
            ax.set_xticklabels(['0', '0.5', '1', '1.5'] if label == 'absent' else [])
            ax.tick_params(labelsize=6.5, length=2.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if label == 'present':
                ax.set_title(name, fontsize=7.5)
            if ax is row[0]:
                ax.set_ylabel(f'Wave V {label}\nPower (dB)', fontsize=7)
            else:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], color=REAL_COLOUR, linewidth=0.8, label='Real datasets'),
               plt.Line2D([], [], color=SOURCE_COLOUR, linewidth=0.9, linestyle=':', label='Source, no augmentation'),
               plt.Line2D([], [], color=SOURCE_COLOUR, linewidth=1.4, label='Source, augmented')]
    axes['absent'][-1].legend(handles=handles, loc='lower right', fontsize=5.8, frameon=False, borderaxespad=0.2)


def similarity_figure() -> Path:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    # b: one spectrum panel per synthetic source or control across the width, present above absent
    n, left, right, gap, bottom, height, row_gap = len(SPECTRUM_SOURCES), 0.6, 7.05, 0.15, 0.5, 1.25, 0.3
    width = (right - left - (n - 1) * gap) / n
    rows = {label: [inches(fig, left + k * (width + gap), bottom + r * (height + row_gap), width, height)
                    for k in range(n)] for r, label in ((1, 'present'), (0, 'absent'))}
    spectra(rows)
    top_b = bottom + 2 * height + row_gap
    # a: heatmap above, colour bar on its right
    heatmap(fig, inches(fig, 1.75, top_b + 0.75, 2.7, 2.6), inches(fig, 4.65, top_b + 1.25, 0.1, 1.6))
    fig.text(0.5 * (left + right) / FIG_WIDTH, 0.08 / FIG_HEIGHT, 'Frequency (kHz)', ha='center', va='bottom', fontsize=7)
    for label, y_in in (('a', FIG_HEIGHT - 0.15), ('b', top_b + 0.45)):
        fig.text(0.05 / FIG_WIDTH, y_in / FIG_HEIGHT, label, fontsize=9, fontweight='bold', va='top')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'similarity.{ext}', dpi=DPI)
    plt.close(fig)
    return OUT_DIR / 'similarity.png'


if __name__ == '__main__':
    print(similarity_figure())
