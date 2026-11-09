"""Figure of analysis/dev_size.py: EEGConformer AUROC against the number of synthetic
development subjects.

Rows: training without augmentation and with the tuned augmentation; columns: the three test
sets. One line per synthetic or control source: mean AUROC over the 30 runs at each
development-pool size with its 95% participant-bootstrap CI as an error bar (the series are
slightly offset in x so the bars do not overlap); controls dashed. The pool sizes are placed at
equal spacing on the x axis although they are not evenly spaced in value. The
real-data baselines (8 development subjects, same architecture and augmentation setting,
from analysis/real_vs_synthetic.py) are repeated as dotted lines labelled M (Mendeley),
S (Southampton) and P (PhysioNet).
Writes dev_size.png (all waveforms), also as .pdf, into results/dev_size/figures/.

Run from the repository root:
  python figures/dev_size.py
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
import pandas as pd

from analysis.dev_size import SOURCES, TEST_SETS
from analysis.multi_arch import HUMAN

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'dev_size'
OUT_DIR = RESULTS_DIR / 'figures'
DPI = 500
FIG_WIDTH = 7.2                      # inches, double column
SUBSETS = {'all': 'dev_size'}                                    # subset -> output stem
# dev_size augmentation setting -> (row title, real_vs_synthetic results folder with the human baselines)
AUGMENTATIONS = {'none': ('No augmentation', REPOSITORY_ROOT / 'results' / 'real_vs_synthetic_noaug'),
                 'tuned': ('Tuned augmentation', REPOSITORY_ROOT / 'results' / 'real_vs_synthetic_aug')}
SHORT = {'roenne': 'Rønne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Rønne + Verhulst',
         'gaussian': 'Matched Gaussian'}       # the control name of Table 1
# Okabe-Ito hues in fixed order; markers as secondary encoding.
STYLE = {'roenne': ('#0072B2', 'o', '-'), 'verhulst': ('#56B4E9', 's', '-'), 'roenne+verhulst': ('#CC79A7', '^', '-'),
         'gaussian': ('#D55E00', 'D', '--')}
BASELINE_COLOUR = '#009E73'          # human-trained baselines and their initials, as in Fig. 2
INITIAL = {'wang': 'M', 'southampton': 'S', 'physionet': 'P'}
Y_LIM = (0.4, 1.0)
LABEL_GAP = 0.03                     # AUROC distance below which the two baseline labels are nudged apart
DODGE = 0.07                         # x offset between series, in tick spacings
X_MARGIN = 0.32                      # free x range beyond the first and last pool size, in tick spacings


def load(subset: str) -> pd.DataFrame:
    ci = pd.read_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv')
    return ci[(ci['subset'] == subset) & ci['eval_set'].isin(TEST_SETS)]


def human_baselines(subset: str) -> dict[str, pd.DataFrame]:
    """Per augmentation setting: the 8-subject real-data AUROC (sources, eval_set, auroc)."""
    out = {}
    for aug, (_, folder) in AUGMENTATIONS.items():
        ci = pd.read_csv(folder / 'bootstrap_auroc_ci.csv')
        out[aug] = ci[(ci['subset'] == subset) & ci['sources'].isin(HUMAN) & ci['eval_set'].isin(TEST_SETS)]
    return out


def panel(ax: plt.Axes, ci: pd.DataFrame, human: pd.DataFrame, eval_set: str, n_dev: list[int]) -> None:
    position = {n: k for k, n in enumerate(n_dev)}       # pool size -> its equally spaced x position
    x_last = len(n_dev) - 1
    h = human[human['eval_set'] == eval_set]
    baselines = sorted((r.auroc, r.sources) for r in h.itertuples())
    close = len(baselines) == 2 and baselines[1][0] - baselines[0][0] < LABEL_GAP
    for k, (value, source) in enumerate(baselines):
        ax.axhline(value, color=BASELINE_COLOUR, linewidth=0.6, linestyle=':', zorder=1)
        va = ('top' if k == 0 else 'bottom') if close else 'center'
        ax.text(x_last + X_MARGIN / 2, value, INITIAL[source], fontsize=6, color=BASELINE_COLOUR, ha='left', va=va)
    for k, (source, (colour, marker, style)) in enumerate(STYLE.items()):
        d = ci[(ci['eval_set'] == eval_set) & (ci['sources'] == source)].sort_values('n_dev')
        if d.empty:
            continue
        x = d['n_dev'].map(position) + (k - (len(STYLE) - 1) / 2) * DODGE
        ax.errorbar(x, d['auroc'], yerr=[d['auroc'] - d['ci_low'], d['ci_high'] - d['auroc']], color=colour,
                    marker=marker, linestyle=style, markersize=3.2, linewidth=1.1, elinewidth=0.6, capsize=1.2,
                    markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    ax.set_xticks(range(len(n_dev)))
    ax.set_xticklabels([str(n) for n in n_dev], fontsize=6.5)
    ax.set_xlim(-X_MARGIN, x_last + X_MARGIN)
    ax.set_ylim(*Y_LIM)
    ax.axhline(0.5, color='0.85', linewidth=0.6, zorder=0)
    ax.grid(axis='y', color='0.92', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', labelsize=7)


def figure(subset: str, stem: str) -> Path:
    ci = load(subset)
    human = human_baselines(subset)
    n_dev = sorted(ci['n_dev'].unique())         # the same positions in every panel
    fig, axes = plt.subplots(len(AUGMENTATIONS), len(TEST_SETS), figsize=(FIG_WIDTH, 4.6), sharey=True, sharex=True)
    for i, (aug, (aug_name, _)) in enumerate(AUGMENTATIONS.items()):
        for j, (eval_set, set_name) in enumerate(TEST_SETS.items()):
            ax = axes[i, j]
            panel(ax, ci[ci['augmentation'] == aug], human[aug], eval_set, n_dev)
            n = ci[ci['eval_set'] == eval_set].iloc[0]
            if i == 0:
                ax.set_title(f'Test: {set_name}\n({n.n_participants} participants, {n.n_waveforms} waveforms)',
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(f'{aug_name}\nAUROC', fontsize=8)
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle=s, markersize=3.5, linewidth=1.1, label=SHORT[src])
               for src, (c, m, s) in STYLE.items()]
    handles.append(plt.Line2D([], [], color=BASELINE_COLOUR, linestyle=':', linewidth=0.8,
                              label='Real-data baseline, 8 subjects (M/S/P)'))
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.1, 'Synthetic development subjects', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
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
