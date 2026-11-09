"""Figures of analysis/attribution.py and analysis/attribution_image.py: Integrated Gradients on
the real test waveforms. Into results/attribution/figures/, each as .png and .pdf.

  attribution_levels           EEGConformer, baseline subject_mean, wave V present (top half) and
      absent (bottom half); rows: the test sets (PhysioNet both frequencies together), columns:
      Rønne, Verhulst, Rønne+Verhulst and the matched Gaussian control.
      Each panel stacks the mean waveform of each stimulus level (10 dB bins for Mendeley and
      PhysioNet, formed over the whole test set so both halves share them), highest first,
      coloured by the attribution averaged over those waveforms and
      over the runs. Each panel is scaled to its own largest |attribution| (not printed).
      The 5-11 ms wave V window (Berninger et al. 2014) is shaded.
  attribution_cwt_vit_image_halves    CWT-ViT counterpart of attribution_levels: the attribution
      on the CWT image averaged over all waveforms of each label (no level split), wave V present
      (top half) and absent (bottom half), rows (test sets) and columns (four training sources)
      as attribution_levels, each panel scaled to its own largest |attribution| (not printed),
      the mean waveform of the label drawn over it.

Run from the repository root (loads the evaluation sets, about 15 s):
  python figures/attribution.py
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
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

from analysis.attribution import TEST_SETS
from src.data import load_eval_sets

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IN_DIR = REPOSITORY_ROOT / 'results' / 'attribution'
OUT_DIR = IN_DIR / 'figures'
DPI = 500
FIG_WIDTH = 7.2                      # inches, double column
CMAP = 'coolwarm'                    # attribution: red towards 'absent', blue towards 'present'
WAVE_V_WINDOW_MS = (5.0, 11.0)       # wave V window (Berninger et al. 2014), as in analysis/attribution.py
WINDOW_COLOUR = '#f2f2f2'
MAIN_ARCH, MAIN_BASELINE = 'eegconformer', 'subject_mean'
# 10 dB level bins where the levels step by 5 dB
LEVEL_BIN_DB = {'wang': 10.0, 'southampton': None, 'physionet': 10.0}
# Test sets (rows, PhysioNet both frequencies together), training sources (columns), level units
LEVEL_SOURCES = {'roenne': 'Rønne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Rønne+Verhulst',
                 'gaussian': 'Matched Gaussian'}       # the control name of Table 1
LEVEL_UNITS = {'wang': 'dB nHL', 'southampton': 'dB SL', 'physionet': 'dB peSPL'}
HALVES = {0: 'Wave V present', 1: 'Wave V absent'}       # expert label (label 1 = wave V absent)
IMAGE_FREQ_TICKS = [0, 4, 9, 19, 29]         # CWT rows (scales 1, 5, 10, 20, 30) labelled with their frequency


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def coloured_line(ax: plt.Axes, t: np.ndarray, y: np.ndarray, c: np.ndarray, vmax: float, linewidth: float = 1.6) -> LineCollection:
    points = np.column_stack([t, y])[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=CMAP, norm=plt.Normalize(-vmax, vmax), linewidth=linewidth, capstyle='round')
    lc.set_array((c[:-1] + c[1:]) / 2)
    ax.add_collection(lc)
    return lc


def shade_window(ax: plt.Axes) -> None:
    ax.axvspan(*WAVE_V_WINDOW_MS, color=WINDOW_COLOUR, zorder=0, linewidth=0)


def save(fig: plt.Figure, stem: str) -> Path:
    for ext in ('png', 'pdf'):
        fig.savefig(OUT_DIR / f'{stem}.{ext}', dpi=DPI)
    plt.close(fig)
    return OUT_DIR / f'{stem}.png'


def level_groups(records: list[dict], freq: int | None, bin_db: float | None) -> list[tuple[str, np.ndarray]]:
    """(label, indices) per stimulus level or 10 dB bin, highest first."""
    idx = np.array([i for i, r in enumerate(records) if freq is None or r.get('freq_khz') == freq])
    levels = np.array([records[i]['intensity'] for i in idx])
    if bin_db is None:
        keys = levels
    else:
        keys = np.floor((levels - levels.min()) / bin_db)
    groups = []
    for k in sorted(set(keys), reverse=True):
        members = idx[keys == k]
        lv = sorted({records[i]['intensity'] for i in members})
        label = f'{lv[0]:g}' if len(lv) == 1 else f'{lv[0]:g}–{lv[-1]:g}'
        groups.append((label, members))
    return groups


def image_axes(ax: plt.Axes, t_ms: np.ndarray, freq_hz: np.ndarray) -> None:
    ax.set_xlim(0, t_ms[-1])
    ax.set_yticks(IMAGE_FREQ_TICKS)
    ax.set_yticklabels([f'{freq_hz[k]:.0f}' for k in IMAGE_FREQ_TICKS])
    ax.tick_params(labelsize=6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def draw_image(ax: plt.Axes, img: np.ndarray, t_ms: np.ndarray, cmap: str, vmin: float, vmax: float):
    return ax.imshow(img, aspect='auto', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
                     extent=(t_ms[0], t_ms[-1], img.shape[0] - 0.5, -0.5))


def wave_v_lines(ax: plt.Axes) -> None:
    for edge in WAVE_V_WINDOW_MS:
        ax.axvline(edge, color='0.3', linewidth=0.5, linestyle=':', zorder=2)


# ------------------------------------------------------------------------------
# EEGConformer: per-level stacked waveforms, present and absent apart, four sources
# ------------------------------------------------------------------------------
def levels_figure(eval_sets: dict[str, list[dict]]) -> Path:
    attributions = np.load(IN_DIR / f'{MAIN_ARCH}_attributions.npz')
    t_ms = np.load(IN_DIR / f'{MAIN_ARCH}_curves.npz')['t_ms']
    # Level bins over the whole test set, so that both halves share them; then the waveforms of each label
    groups, waves = {}, {}
    for eval_set in TEST_SETS:
        records = eval_sets[eval_set]
        x = np.stack([r['waveform_proc'] for r in records])
        y = np.array([int(r['label']) for r in records])
        for label in HALVES:
            groups[eval_set, label] = [(name, m[y[m] == label]) for name, m in
                                       level_groups(records, None, LEVEL_BIN_DB[eval_set]) if (y[m] == label).any()]
            waves[eval_set, label] = [x[m].mean(axis=0) - x[m].mean() for _, m in groups[eval_set, label]]
    # One trace spacing per test set (0.75 of its largest mean waveform, so neighbouring traces may
    # touch), the traces of a column on one amplitude scale
    spacing = {es: 0.75 * max(w.max() - w.min() for label in HALVES for w in waves[es, label]) for es in TEST_SETS}
    # Layout in inches: each row (test set) as tall as its number of levels, margins for the titles and labels
    row_h = {(es, label): 0.1 * (len(groups[es, label]) + 1.0) for es in TEST_SETS for label in HALVES}
    gap, top_in, bottom_in = 0.2, 0.48, {0: 0.25, 1: 0.45}
    panels_in = {label: sum(row_h[es, label] for es in TEST_SETS) for label in HALVES}
    heights = {label: top_in + bottom_in[label] + panels_in[label] + 2 * gap for label in HALVES}
    fig = plt.figure(figsize=(FIG_WIDTH, sum(heights.values())))
    subfigs = fig.subfigures(2, 1, height_ratios=[heights[0], heights[1]], hspace=0.0)
    for subfig, label in zip(subfigs, HALVES):
        axes = subfig.subplots(len(TEST_SETS), len(LEVEL_SOURCES), sharex=True, sharey='row',
                               gridspec_kw={'height_ratios': [row_h[es, label] for es in TEST_SETS]})
        subfig.suptitle(HALVES[label], fontsize=8.5, fontweight='bold', y=1 - 0.04 / heights[label], va='top')
        for i, (eval_set, set_name) in enumerate(TEST_SETS.items()):
            s = spacing[eval_set]
            for j, (source, source_name) in enumerate(LEVEL_SOURCES.items()):
                ax = axes[i, j]
                a = attributions[f'{source}|{eval_set}|{MAIN_BASELINE}']
                traces = [(name, len(m), w, a[m].mean(axis=0))
                          for (name, m), w in zip(groups[eval_set, label], waves[eval_set, label])]
                vmax = max(float(np.abs(c).max()) for _, _, _, c in traces)       # one scale per panel
                shade_window(ax)
                for k, (_, _, w, c) in enumerate(traces):
                    lc = coloured_line(ax, t_ms, w - k * s, c / vmax, 1.0, linewidth=1.0)
                ax.set_xlim(0, t_ms[-1])
                ax.set_xticks([0, 5, 10, 15])
                for side in ('top', 'right'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(labelsize=5.5, length=2)
                if j == 0:                # the levels are the same along a row
                    ax.set_yticks([-k * s for k in range(len(traces))])
                    ax.set_yticklabels([name for name, _, _, _ in traces])
                    ws = [w - k * s for k, (_, _, w, _) in enumerate(traces)]    # the whole traces stay inside
                    ax.set_ylim(min(w.min() for w in ws) - 0.1 * s, max(w.max() for w in ws) + 0.1 * s)
                    ax.set_ylabel(f'{set_name}\n({LEVEL_UNITS[eval_set]})', fontsize=7)
                if i == 0:
                    ax.set_title(source_name, fontsize=7.5)
                if i == len(TEST_SETS) - 1 and label == 1:
                    ax.set_xlabel('ms after the stimulus', fontsize=7)
        mean_row = panels_in[label] / len(TEST_SETS)
        subfig.subplots_adjust(left=0.15, right=0.87, top=1 - top_in / heights[label],
                               bottom=bottom_in[label] / heights[label], hspace=gap / mean_row, wspace=0.12)
    cax = fig.add_axes((0.9, 0.3, 0.012, 0.4))
    cbar = fig.colorbar(lc, cax=cax)
    cbar.set_label('Attribution\n← present   absent →', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    return save(fig, 'attribution_levels')


# ------------------------------------------------------------------------------
# CWT-ViT: attribution on the CWT image, present and absent apart, four sources
# ------------------------------------------------------------------------------
def image_halves_figure() -> Path | None:
    """CWT-ViT counterpart of levels_figure: the attribution image averaged over all waveforms of each
    expert label (no level split), present and absent halves, test-set rows and LEVEL_SOURCES columns."""
    path = IN_DIR / 'cwt_vit_image_curves.npz'
    if not path.exists():
        print(f'{path.name} missing: CWT-ViT image figure skipped')
        return None
    z = np.load(path)
    zc = np.load(IN_DIR / 'cwt_vit_curves.npz')
    t_ms, freq_hz = z['t_ms'], z['freq_hz']
    n_time, n_rows = len(t_ms), len(freq_hz)
    waveforms = {es: zc[f'{es}|waveform_mean'][:, :n_time] for es in TEST_SETS}      # (label, time)
    # One amplitude scale per test set over both labels, as the trace spacing of levels_figure
    scales = {es: max(w.max() - w.min() for w in waveforms[es]) for es in TEST_SETS}
    panel_h, gap, top_in, bottom_in = 0.62, 0.12, 0.48, {0: 0.25, 1: 0.45}
    left_in, label_in = 0.6, 0.06          # left margin of the panels and of the frequency label, inches
    heights = {label: top_in + bottom_in[label] + len(TEST_SETS) * panel_h + (len(TEST_SETS) - 1) * gap
               for label in HALVES}
    fig = plt.figure(figsize=(FIG_WIDTH, sum(heights.values())))
    subfigs = fig.subfigures(2, 1, height_ratios=[heights[0], heights[1]], hspace=0.0)
    for subfig, label in zip(subfigs, HALVES):
        axes = subfig.subplots(len(TEST_SETS), len(LEVEL_SOURCES), sharex=True, sharey=True)
        subfig.suptitle(HALVES[label], fontsize=8.5, fontweight='bold', y=1 - 0.04 / heights[label], va='top')
        for i, (eval_set, set_name) in enumerate(TEST_SETS.items()):
            for j, (source, source_name) in enumerate(LEVEL_SOURCES.items()):
                ax = axes[i, j]
                img = z[f'{source}|{eval_set}|mean'][label]
                im = draw_image(ax, img / float(np.abs(img).max()), t_ms, CMAP, -1.0, 1.0)   # one scale per panel
                w = waveforms[eval_set][label]
                y = (n_rows - 1) / 2 - (w - (w.max() + w.min()) / 2) / scales[eval_set] * (n_rows - 1)
                ax.plot(t_ms, y, color='k', linewidth=0.7, alpha=0.8, zorder=3)
                ax.set_ylim(n_rows - 0.5, -0.5)
                wave_v_lines(ax)
                image_axes(ax, t_ms, freq_hz)
                ax.tick_params(labelsize=5.5, length=2)
                if i == 0:
                    ax.set_title(source_name, fontsize=7.5)
                if j == 0:
                    ax.set_ylabel(set_name, fontsize=7)
                if i == len(TEST_SETS) - 1 and label == 1:
                    ax.set_xlabel('ms after the stimulus', fontsize=7)
        subfig.subplots_adjust(left=left_in / FIG_WIDTH, right=0.87, top=1 - top_in / heights[label],
                               bottom=bottom_in[label] / heights[label], hspace=gap / panel_h, wspace=0.08)
    cax = fig.add_axes((0.9, 0.3, 0.012, 0.4))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Attribution\n← present   absent →', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.supylabel('CWT frequency (Hz)', fontsize=7, x=label_in / FIG_WIDTH)
    return save(fig, 'attribution_cwt_vit_image_halves')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(levels_figure(load_eval_sets()))
    print(image_halves_figure())


if __name__ == '__main__':
    main()
