"""Figures of synthetic waveforms, into results/synthetic_waveforms/figures/, each as .png and .pdf.

  synthetic_examples    the first N_EXAMPLES subjects of the training data, Rønne (datasets/
      synthetic_roenne) in the left column and Verhulst (datasets/synthetic_verhulst) in the right,
      each panel titled with the subject's drawn parameters, the waveforms at LEVELS overlaid.
  verhulst_brainstem    the 20 subjects of src/synthetic_verhulst_brainstem.py (datasets/
      verhulst_brainstem: CN and IC parameters drawn within +-20% of their defaults), one panel per
      subject, the waveforms at LEVELS overlaid.
Waveforms are preprocessed as the training data (src.data.load_synthetic_folder: 30-3000 Hz
band-pass at the native rate, 20 kHz, 0-15 ms after the stimulus), not normalised, in µV; all
panels of a figure share the y axis.

Run from the repository root (after src/synthetic_verhulst_brainstem.py):
  python figures/synthetic_waveforms.py
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import DATA_ROOT, POST_LEN, TARGET_RATE
from src.data_loader import pad_or_truncate, preprocess_synthetic_waveform

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUT_DIR = REPOSITORY_ROOT / 'results' / 'synthetic_waveforms' / 'figures'
DPI = 500
FIG_WIDTH = 7.2                       # inches, double column
LEVELS = [20.0, 50.0, 70.0]           # dB peSPL
# Sequential, one hue: darker = louder
COLOURS = dict(zip(LEVELS, ['#9ECAE1', '#4292C6', '#08306B']))
LINE_WIDTH = 0.9
N_EXAMPLES = 5
BRAINSTEM_DIR = DATA_ROOT / 'verhulst_brainstem'
TRAINING_DIRS = {'roenne': DATA_ROOT / 'synthetic_roenne', 'verhulst': DATA_ROOT / 'synthetic_verhulst'}


def subject_waveforms(folder: Path, subject_id: str) -> tuple[pd.Series, dict[float, np.ndarray]]:
    """First labels row of one generated subject and its preprocessed waveforms at LEVELS."""
    labels = pd.read_csv(folder / subject_id / 'labels.csv')
    waves = np.load(folder / subject_id / 'waveforms.npy')
    rate = float(labels['native_sample_rate_hz'].iloc[0])
    out = {}
    for i, row in labels.iterrows():
        level = float(row['intensity_db'])
        if level in LEVELS:
            wf = preprocess_synthetic_waveform(waves[i], rate, float(TARGET_RATE))
            offset = int(round(int(row['pre_samp']) * TARGET_RATE / rate))
            out[level] = pad_or_truncate(wf[offset:], POST_LEN)
    return labels.iloc[0], out


def draw(ax: plt.Axes, waves: dict[float, np.ndarray]) -> None:
    t_ms = np.arange(POST_LEN) / TARGET_RATE * 1000.0
    ax.axhline(0.0, color='0.75', linewidth=0.5)
    for level in LEVELS:
        ax.plot(t_ms, waves[level], color=COLOURS[level], linewidth=LINE_WIDTH, label=f'{level:g} dB peSPL')
    ax.set_xlim(0.0, POST_LEN / TARGET_RATE * 1000.0)
    ax.set_xticks([0, 5, 10, 15])
    ax.tick_params(labelsize=6.5, length=2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def finish(fig: plt.Figure, axes: np.ndarray, stem: str) -> Path:
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (ms)', fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('Amplitude (µV)', fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(LEVELS), fontsize=7.5, frameon=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{stem}.{ext}', dpi=DPI)
    plt.close(fig)
    return OUT_DIR / f'{stem}.png'


def brainstem_figure() -> Path:
    subjects = sorted(p.name for p in BRAINSTEM_DIR.iterdir() if p.is_dir())
    n_cols = 5
    n_rows = int(np.ceil(len(subjects) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, 1.25 * n_rows + 0.5), sharex=True, sharey=True,
                             squeeze=False)
    for ax, subject in zip(axes.flat, subjects):
        draw(ax, subject_waveforms(BRAINSTEM_DIR, subject)[1])
    for ax in axes.flat[len(subjects):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.93, wspace=0.12, hspace=0.18)
    return finish(fig, axes, 'verhulst_brainstem')


def title_of(model: str, row: pd.Series) -> str:
    if model == 'roenne':
        return (f'Rønne: $c_{{\\mathrm{{OHC}}}}$ = {row["cohc"]:.2f}, $c_{{\\mathrm{{IHC}}}}$ = {row["cihc"]:.2f}, '
                f'spont = {row["spont"]:.0f} spikes/s')
    return (f'Verhulst: $N_{{\\mathrm{{H}}}}$ = {row["num_h"]}, $N_{{\\mathrm{{M}}}}$ = {row["num_m"]}, '
            f'$N_{{\\mathrm{{L}}}}$ = {row["num_l"]}, seed {row["subject_seed"]}')


def examples_figure() -> Path:
    fig, axes = plt.subplots(N_EXAMPLES, 2, figsize=(FIG_WIDTH, 1.45 * N_EXAMPLES + 0.5), sharex=True,
                             sharey=True, squeeze=False)
    for col, (model, folder) in enumerate(TRAINING_DIRS.items()):
        for row in range(N_EXAMPLES):
            labels, waves = subject_waveforms(folder, f'{model}_{row:05d}')
            draw(axes[row, col], waves)
            axes[row, col].set_title(title_of(model, labels), fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.06, top=0.93, wspace=0.08, hspace=0.42)
    return finish(fig, axes, 'synthetic_examples')


def main() -> None:
    print(examples_figure())
    print(brainstem_figure())


if __name__ == '__main__':
    main()
