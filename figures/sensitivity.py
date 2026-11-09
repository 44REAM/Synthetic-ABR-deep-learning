"""Figure of analysis/sensitivity.py: synthetic-trained EEGConformer on the simulated impairment sweeps.
Into results/sensitivity/figures/, as .png and .pdf.

  sensitivity   columns: the four sweeps (Rønne cOHC, Rønne cIHC, Verhulst synaptopathy, Verhulst flat
      OHC-loss profile); top row: the wave V amplitude of each simulated setting (the largest value in
      4-12 ms, stored with the sweep); rows below: P(wave V absent) of the models trained on Rønne, Verhulst, Rønne+Verhulst and the
      matched Gaussian control (64 development subjects, tuned augmentation), mean over the 30 runs,
      against the stimulus level. One line per setting, one colour map per sweep (the normal setting
      at its dark or blue end and drawn thicker), with a discrete colour bar under each column.

Run from the repository root:
  python figures/sensitivity.py
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
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix',
                            'pdf.fonttype': 42})   # Times New Roman, embedded as TrueType
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IN_DIR = REPOSITORY_ROOT / 'results' / 'sensitivity'
OUT_DIR = IN_DIR / 'figures'
DPI = 500
FIG_WIDTH = 7.2                      # inches, double column
# Columns: sweep, title, colour map and the part of it used (normal setting first), colour-bar label,
# tick labels shown on the colour bar (setting -> label)
SWEEPS = [
    ('OHC loss (cOHC)', 'Rønne, OHC loss', 'viridis', (0.0, 0.95), r'$c_{\mathrm{OHC}}$',
     {'cOHC 1.0': '1.0', 'cOHC 0.5': '0.5', 'cOHC 0.0': '0'}),
    ('IHC loss (cIHC)', 'Rønne, IHC loss', 'plasma', (0.0, 0.92), r'$c_{\mathrm{IHC}}$',
     {'cIHC 1.0': '1.0', 'cIHC 0.5': '0.5', 'cIHC 0.0': '0'}),
    ('synaptopathy', 'Verhulst, synaptopathy', 'magma', (0.0, 0.9), r'$N_{\mathrm{H}}/N_{\mathrm{M}}/N_{\mathrm{L}}$',
     {'numH/M/L 13/3/3': '13/3/3', 'numH/M/L 1/3/3': '1/3/3', 'numH/M/L 1/0/0': '1/0/0'}),
    ('OHC loss (flat profile)', 'Verhulst, OHC loss', 'coolwarm', (0.0, 1.0), 'Hearing profile',
     {'Flat00': 'Flat00', 'Flat35': 'Flat35'}),
]
SOURCES = {'roenne': 'Rønne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Rønne+Verhulst',
           'gaussian': 'Matched Gaussian'}
LEVEL_TICKS = [0, 30, 60, 90]
AMPLITUDE_MAX_UV = 0.6               # above the largest simulated wave V of every sweep (0.57 uV)


def settings_of(p: pd.DataFrame, sweep: str) -> list[str]:
    """Settings of a sweep in sweep order, the normal setting first."""
    return p[p['sweep'] == sweep].drop_duplicates('setting').sort_values('order')['setting'].tolist()


def draw_lines(ax: plt.Axes, series: dict[int, pd.Series], colours: list) -> None:
    """One line per setting, the normal setting (index 0) thicker and on top."""
    for k in list(range(1, len(colours))) + [0]:
        s = series[k]
        ax.plot(s.index, s.values, color=colours[k], linewidth=1.4 if k == 0 else 0.9, zorder=3 if k == 0 else 2)


def sensitivity_figure() -> Path:
    p = pd.read_csv(IN_DIR / 'predictions.csv')
    prob = p.groupby(['sources', 'sweep', 'setting', 'intensity_db'])['prob_absent'].mean()
    amp = p.groupby(['sweep', 'setting', 'intensity_db'])['wave_v_amplitude_uv'].first()
    n_rows = 1 + len(SOURCES)
    fig, axes = plt.subplots(n_rows, len(SWEEPS), figsize=(FIG_WIDTH, 7.4), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.135, hspace=0.25, wspace=0.12)   # room for the colour bars
    for j, (sweep, title, cmap_name, cmap_range, bar_label, bar_ticks) in enumerate(SWEEPS):
        settings = settings_of(p, sweep)
        colours = [plt.get_cmap(cmap_name)(v) for v in np.linspace(*cmap_range, len(settings))]
        # Top row: the simulated wave V amplitude
        ax = axes[0, j]
        draw_lines(ax, {k: amp.loc[(sweep, s)] for k, s in enumerate(settings)}, colours)
        ax.set_ylim(0, AMPLITUDE_MAX_UV)                           # one amplitude scale for the row
        ax.set_yticks(np.arange(0, AMPLITUDE_MAX_UV + 0.01, 0.2))
        ax.set_title(title, fontsize=8)
        if j == 0:
            ax.set_ylabel('Simulated wave V\namplitude (µV)', fontsize=7)
        else:
            ax.set_yticklabels([])
        # Rows below: P(wave V absent) of each training source
        for i, (source, source_name) in enumerate(SOURCES.items(), start=1):
            ax = axes[i, j]
            ax.axhline(0.5, color='0.85', linewidth=0.6, zorder=1)
            draw_lines(ax, {k: prob.loc[(source, sweep, s)] for k, s in enumerate(settings)}, colours)
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1])
            if j == 0:
                ax.set_ylabel(f'Trained on {source_name}\nP(wave V absent)', fontsize=7)
            else:
                ax.set_yticklabels([])
        axes[-1, j].set_xlabel('Stimulus level (dB peSPL)', fontsize=7)
        axes[-1, j].set_xticks(LEVEL_TICKS)
        for ax in axes[:, j]:
            ax.set_xlim(0, 90)
            ax.tick_params(labelsize=6.5, length=2.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        # Discrete colour bar under the column, normal setting first
        norm = BoundaryNorm(np.arange(len(settings) + 1) - 0.5, len(settings))
        pos = axes[-1, j].get_position()
        cax = fig.add_axes((pos.x0, 0.05, pos.width, 0.009))
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ListedColormap(colours)), cax=cax,
                            orientation='horizontal')
        cbar.set_ticks([settings.index(s) for s in bar_ticks])
        cbar.set_ticklabels(list(bar_ticks.values()))
        cbar.ax.tick_params(labelsize=6, length=2)
        cbar.ax.minorticks_off()
        cbar.outline.set_visible(False)
        cbar.set_label(bar_label, fontsize=7, labelpad=2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'sensitivity.{ext}', dpi=DPI)
    plt.close(fig)
    return OUT_DIR / 'sensitivity.png'


if __name__ == '__main__':
    print(sensitivity_figure())
