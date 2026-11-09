"""Figure of analysis/real_vs_synthetic.py: real-data models against synthetic- (MixUp and CutMix
included) and control-trained EEGConformer models, all with 8 development subjects.

Rows: training without augmentation ('noaug') and with the tuned augmentation ('aug');
columns: the three test sets. Per training source: the mean AUROC over the 30 runs (marker)
with its 95% participant-bootstrap CI (bar) and the single runs behind it (small dots); the
two real-data baselines tested on that set are repeated as horizontal lines, each with its
initial in italics at its right end, beyond the last column. Letters above
a marker name the real-data baselines (M Mendeley, S Southampton, P PhysioNet) that the source
exceeds significantly (paired participant-bootstrap difference, Benjamini-Hochberg q < 0.05
over all comparisons of that setting, as in the analysis); letters below a marker, with a minus
sign, name the baselines the source falls short of significantly. Real-data sources are not tested
on their own dataset.
Reads bootstrap_auroc_ci.csv, per_run_metrics.csv and paired_diff.csv of both settings and
writes real_vs_synthetic.png (and .pdf) into results/real_vs_synthetic_aug/figures/.
figures/real_vs_synthetic_cwt_vit.py draws the same figure for CWT-ViT with draw().

Run from the repository root:
  python figures/real_vs_synthetic.py
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.multi_arch import ALPHA, HUMAN
from analysis.real_vs_synthetic import SOURCES, TEST_SETS

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SETTINGS = {'noaug': 'No augmentation', 'aug': 'Tuned augmentation'}
RESULTS = {s: REPOSITORY_ROOT / 'results' / f'real_vs_synthetic_{s}' for s in SETTINGS}
OUT_DIR = RESULTS['aug'] / 'figures'
DPI = 500
FIG_WIDTH = 7.2                      # inches, double column
SHORT = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet', 'roenne': 'Rønne',
         'verhulst': 'Verhulst', 'roenne+verhulst': 'Rønne +\nVerhulst', 'mixup': 'MixUp', 'cutmix': 'CutMix',
         # The control names of Table 1, on one line: two-line labels of neighbouring ticks touch
         'gaussian': 'Matched Gaussian', 'gaussian_latency': 'Latency-matched Gaussian',
         'simple_gaussian': 'Unmatched Gaussian'}
INITIAL = {'wang': 'M', 'southampton': 'S', 'physionet': 'P'}
# MixUp and CutMix mix the two auditory models' waveforms, so they share the physiological synthetic group
GROUP = {'wang': 'human', 'southampton': 'human', 'physionet': 'human', 'roenne': 'synthetic',
         'verhulst': 'synthetic', 'roenne+verhulst': 'synthetic', 'mixup': 'synthetic', 'cutmix': 'synthetic',
         'gaussian': 'control', 'gaussian_latency': 'control', 'simple_gaussian': 'control'}
# Okabe-Ito hues in fixed order (colour-blind safe)
COLOUR = {'human': '#555555', 'synthetic': '#0072B2', 'control': '#D55E00'}
BASELINE_COLOUR = '#009E73'          # baseline lines and their initials, apart from the black significance letters
# "Real" as in Fig. 1 of the paper
GROUP_LABEL = {'human': 'Real data', 'synthetic': 'Physiological synthetic', 'control': 'Non-physiological control'}
Y_LIM = (0.3, 1.0)
RUN_JITTER = 0.18
LABEL_GAP = 0.03                     # AUROC distance below which the two baseline labels are nudged apart
SEED = 0


def load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ci = pd.read_csv(d / 'bootstrap_auroc_ci.csv')
    ci = ci[(ci['subset'] == 'all') & ci['eval_set'].isin(TEST_SETS)]
    runs = pd.read_csv(d / 'per_run_metrics.csv')
    runs = runs[runs['eval_set'].isin(TEST_SETS)]
    diffs = pd.read_csv(d / 'paired_diff.csv')
    return ci, runs, diffs


def significance(diffs: pd.DataFrame, source: str, eval_set: str) -> tuple[str, str]:
    """(baselines the source exceeds significantly, baselines it falls short of significantly)."""
    d = diffs[(diffs['source'] == source) & (diffs['eval_set'] == eval_set) & (diffs['q'] < ALPHA)].sort_values('baseline')
    higher = ' '.join(INITIAL[r.baseline] for r in d.itertuples() if r.diff > 0)
    lower = ' '.join('−' + INITIAL[r.baseline] for r in d.itertuples() if r.diff < 0)
    return higher, lower


def panel(ax: plt.Axes, ci: pd.DataFrame, runs: pd.DataFrame, diffs: pd.DataFrame, eval_set: str,
          order: list[str], rng: np.random.Generator, y_lim: tuple[float, float] = Y_LIM) -> None:
    c = ci[ci['eval_set'] == eval_set].set_index('sources')
    r = runs[runs['eval_set'] == eval_set]
    baselines = sorted((c.loc[h, 'auroc'], h) for h in HUMAN if h in c.index)
    for k, (value, h) in enumerate(baselines):
        ax.axhline(value, color=BASELINE_COLOUR, linewidth=0.6, linestyle=':', zorder=1)
        # Label below the lower and above the upper baseline when the two are close
        close = len(baselines) == 2 and baselines[1][0] - baselines[0][0] < LABEL_GAP
        va = ('top' if k == 0 else 'bottom') if close else 'center'
        # Initial at the right end of the line, beyond the last column, so it is not read as that column's marker
        ax.text(len(order) + 0.1, value, INITIAL[h], fontsize=6, color=BASELINE_COLOUR, ha='left', va=va, style='italic')
    for x, source in enumerate(order):
        if source not in c.index:
            continue
        colour = COLOUR[GROUP[source]]
        auc = r.loc[r['sources'] == source, 'auc'].to_numpy()
        ax.scatter(x + rng.uniform(-RUN_JITTER, RUN_JITTER, len(auc)), auc, s=4, color=colour, alpha=0.25,
                   linewidths=0, zorder=2)
        row = c.loc[source]
        ax.errorbar(x, row['auroc'], yerr=[[row['auroc'] - row['ci_low']], [row['ci_high'] - row['auroc']]],
                    fmt='o', color=colour, markersize=4.5, markeredgecolor='white', markeredgewidth=0.6,
                    capsize=2, elinewidth=1.2, zorder=3)
        higher, lower = significance(diffs, source, eval_set)
        if higher:       # above the bar: significantly higher than these baselines
            ax.text(x, row['ci_high'] + 0.012, higher, fontsize=5.5, ha='center', va='bottom', color='0.2')
        if lower:        # below the bar, with a minus sign: significantly lower
            ax.text(x, row['ci_low'] - 0.012, lower, fontsize=5.5, ha='center', va='top', color='0.2')
    ax.set_xlim(-0.6, len(order) + 0.5)      # room at the right for the baseline initials
    ax.set_ylim(*y_lim)
    ax.set_yticks(np.round(np.arange(y_lim[0], y_lim[1] + 0.001, 0.1), 1))   # fixed 0.1 steps
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[s] for s in order], fontsize=6, rotation=90)
    ax.axhline(0.5, color='0.85', linewidth=0.6, zorder=0)
    ax.grid(axis='y', color='0.92', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', labelsize=7)


def draw(results: dict[str, Path], out_dir: Path, name: str, y_lim: tuple[float, float] = Y_LIM) -> None:
    """The figure from the analysis folders of the two settings, as <name>.png and .pdf in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(SETTINGS), len(TEST_SETS), figsize=(FIG_WIDTH, 5.0), sharey=True)
    data = {setting: load(results[setting]) for setting in SETTINGS}
    present = set().union(*(set(ci['sources']) for ci, _, _ in data.values()))
    order = [s for s in SOURCES if s in present]       # the same positions in both rows
    for i, (setting, setting_name) in enumerate(SETTINGS.items()):
        ci, runs, diffs = data[setting]
        for j, (eval_set, set_name) in enumerate(TEST_SETS.items()):
            ax = axes[i, j]
            panel(ax, ci, runs, diffs, eval_set, order, rng, y_lim)
            n = ci[ci['eval_set'] == eval_set].iloc[0]
            if i == 0:
                ax.set_title(f'Test: {set_name}\n({n.n_participants} participants, {n.n_waveforms} waveforms)',
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(f'{setting_name}\nAUROC', fontsize=8)
            if i < len(SETTINGS) - 1:
                ax.tick_params(axis='x', labelbottom=False)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=COLOUR[g], markersize=4.5, label=GROUP_LABEL[g])
               for g in COLOUR]
    handles.append(plt.Line2D([], [], color=BASELINE_COLOUR, linestyle=':', linewidth=0.8,
                              label='Real-data baseline on this test set'))
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.075, 'Training source (8 development subjects each)', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'{name}.{ext}', dpi=DPI)
    plt.close(fig)
    print(out_dir / f'{name}.png')


def main() -> None:
    draw(RESULTS, OUT_DIR, 'real_vs_synthetic')


if __name__ == '__main__':
    main()
