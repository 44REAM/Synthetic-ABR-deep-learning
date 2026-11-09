"""Figure of analysis/cutpoints.py: the label rule of the synthetic training data.
Rønne+Verhulst, EEGConformer, no augmentation, 30 runs per rule.

One row of four panels, the label rules on the x axis in the order of the analysis (the
extreme-level rule, absent at <=20 and present at >=70 dB, then the cutpoints: present above
50, 40, 30 and 20 dB). First panel: the held-out Southampton subjects, a box plot of the AUROC
of the runs; the box of the selected rule
(src.run.LABEL_RULE) is filled and its label bold. Other panels: the three test sets, mean
AUROC over the runs on all waveforms with its 95% participant-bootstrap CI.
Writes cutpoints.png and .pdf into results/cutpoints/figures/.

Run from the repository root:
  python figures/cutpoints.py
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

from analysis.cutpoints import LABEL_RULES, TEST_SETS
from src.data import AUGDEV_SET
from src.run import LABEL_RULE

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'cutpoints'
OUT_DIR = RESULTS_DIR / 'figures'
DPI = 500
FIG_SIZE = (6, 2.25)                # inches, double column
# 'paper': the extreme-level rule (absent <=20, present >=70 dB); cutN: present above N dB
RULE_LABEL = {'paper': '≤20/≥70', 'cut50': '>50', 'cut40': '>40', 'cut30': '>30', 'cut20': '>20'}
COLOUR = '#0072B2'                   # synthetic sources, as in the other figures
Y_LIM = (0.0, 1.0)


def style(ax: plt.Axes) -> None:
    ax.set_ylim(*Y_LIM)
    ax.grid(axis='y', color='0.92', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xticks(range(len(LABEL_RULES)))
    ax.set_xticklabels([RULE_LABEL[r] for r in LABEL_RULES], fontsize=7, rotation=45, ha='right',
                       rotation_mode='anchor')   # slanted: '≤20/≥70' is wider than the tick spacing
    ax.set_xlim(-0.6, len(LABEL_RULES) - 0.4)


def held_out_panel(ax: plt.Axes, metrics: pd.DataFrame, ci: pd.DataFrame) -> None:
    m = metrics[metrics['eval_set'] == AUGDEV_SET]
    data = [m.loc[m['label_rule'] == r, 'auc'].to_numpy() for r in LABEL_RULES]
    bp = ax.boxplot(data, positions=range(len(LABEL_RULES)), widths=0.6, patch_artist=True, whis=1.5, zorder=3,
                    boxprops={'linewidth': 0.7, 'edgecolor': COLOUR, 'facecolor': 'white'},
                    medianprops={'linewidth': 1.2, 'color': COLOUR},
                    whiskerprops={'linewidth': 0.6, 'color': COLOUR}, capprops={'linewidth': 0.6, 'color': COLOUR},
                    flierprops={'marker': '.', 'markersize': 2.5, 'markerfacecolor': COLOUR,
                                'markeredgecolor': 'none', 'alpha': 0.6})
    style(ax)
    for rule, box, tick in zip(LABEL_RULES, bp['boxes'], ax.get_xticklabels()):
        if rule == LABEL_RULE:
            box.set_facecolor(COLOUR)
            box.set_alpha(0.35)
            tick.set_fontweight('bold')
    n = ci[ci['eval_set'] == AUGDEV_SET].iloc[0]
    ax.set_title(f'Held-out Southampton\n{n.n_participants} participants, {n.n_waveforms} waveforms', fontsize=7.5)
    ax.set_ylabel('AUROC', fontsize=8)


def test_panel(ax: plt.Axes, ci: pd.DataFrame, eval_set: str, set_name: str) -> None:
    d = ci[ci['eval_set'] == eval_set].set_index('label_rule').loc[LABEL_RULES]
    ax.errorbar(range(len(LABEL_RULES)), d['auroc'], yerr=[d['auroc'] - d['ci_low'], d['ci_high'] - d['auroc']],
                color=COLOUR, marker='o', linestyle='none', markersize=3.8, elinewidth=0.7, capsize=1.5, zorder=3)
    style(ax)
    n = d.iloc[0]
    ax.set_title(f'Test: {set_name}\n{n.n_participants} participants, {n.n_waveforms} waveforms', fontsize=7.5)


def figure() -> Path:
    metrics = pd.read_csv(RESULTS_DIR / 'per_run_metrics.csv')
    ci = pd.read_csv(RESULTS_DIR / 'bootstrap_auroc_ci.csv')
    fig, axes = plt.subplots(1, 1 + len(TEST_SETS), figsize=FIG_SIZE, sharey=True, constrained_layout=True)
    held_out_panel(axes[0], metrics, ci)
    for ax, (eval_set, set_name) in zip(axes[1:], TEST_SETS.items()):
        test_panel(ax, ci, eval_set, set_name)
    fig.supxlabel('Label rule (dB peSPL)', fontsize=7.5)
    fig.set_constrained_layout_pads(h_pad=0.02, w_pad=0.02, wspace=0.03)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'cutpoints.{ext}', dpi=DPI)
    plt.close(fig)
    return OUT_DIR / 'cutpoints.png'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(figure())


if __name__ == '__main__':
    main()
