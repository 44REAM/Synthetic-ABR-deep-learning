"""Figure of analysis/tune_augmentation.py: augmentation tuning on the held-out Southampton
subjects, one figure per architecture.

Rows: the training sources (two synthetic, the Gaussian controls, three real); columns: the
augmentation types of the stage-1 grid (time shift, pink, Gaussian and uniform noise) and, for
EEGConformer, the stage-2 DC shift added to the selected stage-1 setting. Each panel is a box
plot of the AUROC of the runs at each value of the augmentation parameter, with no augmentation
('none', or the stage-1 selection 'tuned' in the DC column) as the first box. The box of the
value in selected.json is filled and its tick label bold. Panels of a source not tuned on that
grid are left empty.
Writes tune_augmentation_<arch>.png and .pdf into results/<tuning folder>/figures/.

Run from the repository root:
  python figures/tune_augmentation.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.family': 'Times New Roman', 'pdf.fonttype': 42})   # Times New Roman, embedded as TrueType
import matplotlib.pyplot as plt
import pandas as pd

from experiments.tune_augmentation import DC_SHIFTS, NOISE_LEVELS, SHIFT_MS, TUNE_SETTINGS
from src.data import AUGDEV_SET

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DPI = 500
FIG_WIDTH = 7.2                      # inches, double column
ROW_HEIGHT = 1.0                     # inches per source row
ARCHS = {'eegconformer': 'EEGConformer', 'cwt_vit': 'CWT-ViT'}
SOURCES = {'roenne': 'Rønne', 'verhulst': 'Verhulst', 'gaussian': 'Matched Gaussian\n(control)',
           'gaussian_latency': 'Latency-matched\nGaussian (control)', 'simple_gaussian': 'Unmatched Gaussian\n(control)',
           'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
# Okabe-Ito hues by source family, as in the other figures
COLOUR = {'roenne': '#0072B2', 'verhulst': '#0072B2', 'gaussian': '#D55E00',
          'simple_gaussian': '#D55E00', 'gaussian_latency': '#D55E00',
          'wang': '#555555', 'southampton': '#555555', 'physionet': '#555555'}
# Column: title, x label, run-name prefix, parameter values, key in selected.json, reference configuration
COLUMNS = {
    'shift': ('Time shift', 'Maximum shift (ms)', 'shift', SHIFT_MS, 'shift_limit_ms', 'none'),
    'pink': ('Pink noise', 'Maximum σ', 'pink', NOISE_LEVELS, 'pink_noise_sigma', 'none'),
    'randn': ('Gaussian noise', 'Maximum σ', 'randn', NOISE_LEVELS, 'randn_noise_sigma', 'none'),
    'uniform': ('Uniform noise', 'Maximum amplitude', 'uniform', NOISE_LEVELS, 'uniform_noise_amplitude', 'none'),
    'dc': ('DC shift (on the\nselected setting)', 'Maximum offset', 'tuned+dc', DC_SHIFTS, 'dc_shift_max', 'tuned'),
}
STAGE1_KEYS = {c[4] for name, c in COLUMNS.items() if name != 'dc'}


def load(arch: str) -> tuple[pd.DataFrame, dict]:
    folder = REPOSITORY_ROOT / 'results' / TUNE_SETTINGS[arch]['folder']
    m = pd.read_csv(folder / 'per_run_metrics.csv')
    m = m[m['eval_set'] == AUGDEV_SET]
    return m, json.loads((folder / 'selected.json').read_text())


def panel(ax: plt.Axes, m: pd.DataFrame, source: str, column: str, selected: dict) -> None:
    _, _, prefix, values, key, reference = COLUMNS[column]
    names = [reference] + [f'{prefix}{v:g}' for v in values]
    labels = [reference] + [f'{v:g}' for v in values]
    d = m[m['sources'] == source]
    data = [d.loc[d['augmentation'] == n, 'auc'].to_numpy() for n in names]
    if all(len(x) == 0 for x in data[1:]):
        ax.text(0.5, 0.5, 'not tuned', transform=ax.transAxes, ha='center', va='center', fontsize=6.5, color='0.5')
        ax.set_xticks([])
        return
    colour = COLOUR[source]
    chosen = selected[source].get(key)
    # The reference box is the choice when the source got nothing from that stage: no stage-1
    # augmentation at all (none), or no DC shift on top of the stage-1 selection (tuned)
    reference_chosen = not (set(selected[source]) & STAGE1_KEYS) if reference == 'none' else chosen is None
    bp = ax.boxplot(data, positions=range(len(names)), widths=0.6, showfliers=True, patch_artist=True,
                    whis=1.5, zorder=3,
                    boxprops={'linewidth': 0.7, 'edgecolor': colour, 'facecolor': 'white'},
                    medianprops={'linewidth': 1.2, 'color': colour},
                    whiskerprops={'linewidth': 0.6, 'color': colour}, capprops={'linewidth': 0.6, 'color': colour},
                    flierprops={'marker': '.', 'markersize': 2.5, 'markerfacecolor': colour,
                                'markeredgecolor': 'none', 'alpha': 0.6})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=6)
    for k, (box, tick) in enumerate(zip(bp['boxes'], ax.get_xticklabels())):
        is_chosen = reference_chosen if k == 0 else (chosen is not None and float(values[k - 1]) == float(chosen))
        if is_chosen:       # filled box, and a bold label for a box too flat to show the fill
            box.set_facecolor(colour)
            box.set_alpha(0.35)
            tick.set_fontweight('bold')
    ax.set_xlim(-0.6, len(names) - 0.4)


def figure(arch: str) -> Path:
    m, selected = load(arch)
    columns = [c for c in COLUMNS if c != 'dc' or TUNE_SETTINGS[arch]['dc_stage']]
    out_dir = REPOSITORY_ROOT / 'results' / TUNE_SETTINGS[arch]['folder'] / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {s: n for s, n in SOURCES.items() if s in set(m['sources'])}   # the simple Gaussian: EEGConformer only
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(FIG_WIDTH, ROW_HEIGHT * len(rows) + 0.6),
                             sharey=True, constrained_layout=True)
    y_low = min(0.5, (m['auc'].min() * 10) // 1 / 10)
    for i, (source, source_name) in enumerate(rows.items()):
        for j, column in enumerate(columns):
            ax = axes[i, j]
            panel(ax, m, source, column, selected)
            ax.set_ylim(y_low, 1.0)
            ax.grid(axis='y', color='0.92', linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', labelsize=6)
            if i == 0:
                ax.set_title(COLUMNS[column][0], fontsize=7.5)
            if i == len(rows) - 1:
                ax.set_xlabel(COLUMNS[column][1], fontsize=6.5)
            if j == 0:
                ax.set_ylabel(source_name, fontsize=7)
    fig.suptitle(f'{ARCHS[arch]}, tested on the held-out Southampton participants', fontsize=7.5)
    fig.supylabel('AUROC', fontsize=7.5)
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, hspace=0.03, wspace=0.02)
    stem = f'tune_augmentation_{arch}'
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=DPI)
    plt.close(fig)
    return out_dir / f'{stem}.png'


def main() -> None:
    for arch in ARCHS:
        print(figure(arch))


if __name__ == '__main__':
    main()
