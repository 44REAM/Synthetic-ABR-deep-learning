"""Variability of the synthetic and real waveforms.

Sources: the three real datasets, every subject (test and tuning subjects alike; PhysioNet
separately per tone-burst frequency), the synthetic sources (the two auditory models and the three Gaussian controls of
src/synthetic_controls.py), all 500 subjects, and Roenne+Verhulst, the two
pooled (each part with its own augmentation). Waveforms as preprocessed for
the networks (src/data.py: 30-3000 Hz band-pass, 20 kHz, 0-15 ms after the stimulus): before
normalisation for the amplitudes in uV, after the subject-level min-max normalisation (what
the networks see) for the similarity and the residual noise.

Wave V, per subject (and frequency) at the highest level labelled wave V present (real:
expert labels; synthetic: the stored labels, so 90 dB):
  latency      the local maximum in the search window (PEAK_WINDOW_MS: 5-11 ms, the wave V window
               of Berninger et al. 2014, for every source but the model-free control) with the
               largest drop to the minimum within TROUGH_MS after it, in ms after the stimulus
  latency after the tuned time shift
               the latency plus the delay of one draw of the source's tuned time shift
               (results/tune_augmentation/selected.json; src/augment.py delays the whole waveform
               by a whole number of samples, so wave V moves by exactly that delay); noise and the
               DC offset do not move the waveform, so sources without a tuned shift keep their latency
  amplitude    that drop, peak minus trough, in uV
  ambiguous    another local maximum at least AMBIGUOUS_MS away has a drop of at least
               AMBIGUOUS_RATIO times the largest, so the automatic pick is uncertain
Summarised per source: median, 5th-95th percentile, SD and CV over subjects.
Similarity of the normalised waveforms at that level, over pairs of subjects within each
source and between each source and each real dataset: the mean RM, the template detector's
statistic (analysis/template.py: mean removed, Hanning-tapered, largest normalised
cross-correlation over all lags, so independent of the latency difference). Both on the clean
waveforms and
after one draw of the source's tuned augmentation (results/tune_augmentation/selected.json,
applied as in training, after the normalisation); the real waveforms a source is compared
with stay clean, as test waveforms are.
Residual noise: RMS (mean removed) of the normalised waveforms labelled wave V absent (real:
expert labels; synthetic: the training label rule src.run.LABEL_RULE), per source, as they are
and after the pink noise of the source's tuned augmentation (results/tune_augmentation/
selected.json, applied as in training: after the normalisation, with a standard deviation
drawn uniformly between 0 and the setting for each waveform).
Power spectra (mean removed, Hann window, zero-padded FFT, 0-3000 Hz; the 15 ms
window resolves about 67 Hz, so only broad features are compared): every waveform labelled present
and every waveform labelled absent (real: expert labels; synthetic: the training label rule), as the
networks are trained and tested on them, clean and after one draw of the tuned augmentation,
normalised as for the networks so every source is in units of its subjects' range; per source the
median power over waveforms in dB relative to the peak of its clean present spectrum (spectra.csv).
Writes wave_v.csv, correlation.csv, residual_noise.csv, spectra.csv and summary.txt into
results/variability/.

Run from the repository root:
  python analysis/variability.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from analysis.template import cross_correlation
from src.data import TARGET_RATE, apply_label_rule, load_human_datasets, load_synthetic, noisy_copy, normalise
from src.run import LABEL_RULE, TUNED_AUGMENTATION_FILE

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUT_DIR = REPOSITORY_ROOT / 'results' / 'variability'
SYNTHETIC_SOURCES = ['roenne', 'verhulst', 'gaussian', 'gaussian_latency', 'simple_gaussian']
# Display order and names; PhysioNet is split by tone-burst frequency
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet_1': 'PhysioNet 1 kHz',
           'physionet_4': 'PhysioNet 4 kHz', 'roenne': 'Roenne', 'verhulst': 'Verhulst',
           'roenne+verhulst': 'Roenne+Verhulst', 'gaussian': 'Matched Gaussian (control)',
           'gaussian_latency': 'Latency-matched Gaussian (control)', 'simple_gaussian': 'Unmatched Gaussian (control)'}
HUMAN = ['wang', 'southampton', 'physionet_1', 'physionet_4']
POOLED = {'roenne+verhulst': ['roenne', 'verhulst']}   # pooled sources: the records of their parts
UV_SOURCES = {'wang', 'physionet_1', 'physionet_4', *SYNTHETIC_SOURCES, *POOLED}   # recordings in uV
AUGMENTATION_SOURCE = {'physionet_1': 'physionet', 'physionet_4': 'physionet'}  # key in selected.json
# Wave V search window, ms after the stimulus: 5-11 ms, which contains wave V for clicks from
# -18.5 to 71.5 dB nHL in normal-hearing adults (Berninger et al. 2014, Ear Hear 35:318-329; the
# window of analysis/attribution.py), for clicks and tone bursts alike; the model-free control's
# peak centre is drawn from 2-13 ms (src/synthetic_controls.py), so it gets a window around that
PEAK_WINDOW_MS = {'default': (5.0, 11.0), 'simple_gaussian': (1.5, 13.5)}
TROUGH_MS = 2.5                      # the trough is the minimum within this time after the peak
AMBIGUOUS_RATIO = 0.8
AMBIGUOUS_MS = 0.5
NOISE_SEED = 0
PERCENTILES = (5, 95)
# Power spectra: zero-padded FFT length (interpolation only; the 15 ms window
# resolves about 67 Hz) and the frequencies kept
SPECTRUM_NFFT = 2048
SPECTRUM_MAX_HZ = 3000.0             # the upper edge of the preprocessing band-pass


def load_sources() -> dict[str, list[dict]]:
    """Records per display source, preprocessed, not normalised."""
    human = load_human_datasets()
    recs = {'wang': human['wang'], 'southampton': human['southampton']}
    for f in (1, 4):
        recs[f'physionet_{f}'] = [r for r in human['physionet'] if int(r['freq_khz']) == f]
    for s in SYNTHETIC_SOURCES:
        recs[s] = load_synthetic(s)
    return recs


def wave_v(wf: np.ndarray, window_ms: tuple[float, float]) -> tuple[float, float, bool]:
    """(latency ms, peak-to-trough amplitude, ambiguous) of the largest peak-trough in the window."""
    t = np.arange(len(wf)) / TARGET_RATE * 1000.0
    n_trough = int(round(TROUGH_MS / 1000.0 * TARGET_RATE))
    cands = []
    for i in range(1, len(wf) - 1):
        if window_ms[0] <= t[i] <= window_ms[1] and wf[i] >= wf[i - 1] and wf[i] > wf[i + 1]:
            trough = wf[i + 1:i + 1 + n_trough].min()
            cands.append((float(wf[i] - trough), float(t[i])))
    if not cands:
        return np.nan, np.nan, True
    cands.sort(reverse=True)
    amp, lat = cands[0]
    ambiguous = any(a >= AMBIGUOUS_RATIO * amp and abs(l - lat) >= AMBIGUOUS_MS for a, l in cands[1:])
    return lat, amp, ambiguous


def highest_present(records: list[dict]) -> dict[str, int]:
    """subject -> index of its record at the highest level labelled wave V present."""
    best: dict[str, int] = {}
    for k, r in enumerate(records):
        if r['label'] == 0 and (r['subject'] not in best or r['intensity'] > records[best[r['subject']]]['intensity']):
            best[r['subject']] = k
    return best


def shift_delay_ms(records: list[dict], shift_limit_ms: float) -> np.ndarray:
    """Delay (ms) of each record in one draw of the training time shift (noisy_copy, NOISE_SEED),
    read off a shifted sample index."""
    index = [dict(r, waveform_proc=np.arange(len(r['waveform_proc']), dtype=float)) for r in records]
    shifted = noisy_copy(index, {'shift_limit_ms': shift_limit_ms}, NOISE_SEED)
    return np.array([len(r['waveform_proc']) - 1 - r['waveform_proc'][-1] for r in shifted]) / TARGET_RATE * 1000.0


def pct(values: pd.Series) -> str:
    return f'{values.median():.3g} ({np.percentile(values, PERCENTILES[0]):.3g}-{np.percentile(values, PERCENTILES[1]):.3g})'


def mean_pairwise_rm(a: np.ndarray, b: np.ndarray | None = None, same: bool = False) -> float:
    """Mean RM (analysis/template.py) over pairs of rows: within a (b None) or between a and b
    (same: a and b hold the same subjects in the same order, so pairs of a subject with itself are left out)."""
    if b is None:
        vals = [cross_correlation(a[i + 1:], a[i])[0] for i in range(len(a) - 1)]
    else:
        vals = [np.delete(cross_correlation(a, b[j])[0], j) if same else cross_correlation(a, b[j])[0]
                for j in range(len(b))]
    return float(np.concatenate(vals).mean())


def absolute_power(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, power) of each row: mean removed, Hann window, zero-padded FFT, kept up to
    SPECTRUM_MAX_HZ, in the units of the rows (here the subject's normalised range)."""
    x = (x - x.mean(axis=1, keepdims=True)) * np.hanning(x.shape[1])
    f = np.fft.rfftfreq(SPECTRUM_NFFT, 1.0 / TARGET_RATE)
    keep = f <= SPECTRUM_MAX_HZ
    return f[keep], np.abs(np.fft.rfft(x, SPECTRUM_NFFT, axis=1))[:, keep] ** 2


def main() -> None:
    raw = load_sources()
    norm = {s: normalise(recs) for s, recs in raw.items()}
    selected = json.loads(TUNED_AUGMENTATION_FILE.read_text())

    def config_of(part: str) -> dict:
        return selected.get(AUGMENTATION_SOURCE.get(part, part), {})

    # --- wave V at the highest present level, and its latency after the tuned time shift --------
    rows = []
    for s in SOURCES:
        for part in POOLED.get(s, [s]):          # each part of a pooled source has its own time shift
            window = PEAK_WINDOW_MS.get(part, PEAK_WINDOW_MS['default'])
            recs = [raw[part][k] for k in highest_present(raw[part]).values()]
            shift_limit = config_of(part).get('shift_limit_ms')
            delays = shift_delay_ms(recs, shift_limit) if shift_limit else np.zeros(len(recs))
            for r, delay in zip(recs, delays):
                lat, amp, amb = wave_v(r['waveform_proc'], window)
                rows.append({'source': s, 'subject': r['subject'], 'level_db': r['intensity'], 'latency_ms': lat,
                             'latency_shifted_ms': lat + delay, 'amplitude_uv': amp if s in UV_SOURCES else np.nan,
                             'amplitude_recorded_units': amp, 'ambiguous': amb})
    wv = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wv.to_csv(OUT_DIR / 'wave_v.csv', index=False, float_format='%.6g')

    # --- similarity of the normalised waveforms, clean and after the tuned augmentation --------
    top: dict[str, dict[str, np.ndarray]] = {'none': {}, 'tuned': {}}   # augmentation -> source -> (subjects, samples)
    for s in SOURCES:
        clean, augmented = [], []
        for part in POOLED.get(s, [s]):
            recs = [norm[part][k] for k in highest_present(raw[part]).values()]
            clean += recs
            augmented += noisy_copy(recs, config_of(part), NOISE_SEED)
        top['none'][s] = np.stack([r['waveform_proc'] for r in clean])
        top['tuned'][s] = np.stack([r['waveform_proc'] for r in augmented])
    corr_rows = []
    for augmentation, tops in top.items():
        for s in SOURCES:
            row = {'source': s, 'augmentation': augmentation, 'n_subjects': len(tops[s]),
                   'within_rm': mean_pairwise_rm(tops[s])}
            for h in HUMAN:
                row[f'vs_{h}_rm'] = (row['within_rm'] if s == h and augmentation == 'none'
                                     else mean_pairwise_rm(tops[s], top['none'][h], same=s == h))
            corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_DIR / 'correlation.csv', index=False, float_format='%.4f')

    # --- power spectra on one scale per source: every waveform labelled present and every waveform
    # labelled absent (real: expert labels; synthetic: the training label rule), as the networks are
    # trained and tested on them, clean and after one draw of the tuned augmentation; median power over
    # waveforms in dB relative to the peak of the source's clean present spectrum (flat waveforms have
    # no power, -inf dB where they are the majority) -----------------------------------------------
    waves: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for s in SOURCES:
        for label, code in (('present', 0), ('absent', 1)):
            clean, augmented = [], []
            for part in POOLED.get(s, [s]):
                recs = [r for r in (norm[part] if part in HUMAN else apply_label_rule(norm[part], LABEL_RULE))
                        if r['label'] == code]
                clean += recs
                augmented += noisy_copy(recs, config_of(part), NOISE_SEED)
            waves.setdefault((label, 'none'), {})[s] = np.stack([r['waveform_proc'] for r in clean])
            waves.setdefault((label, 'tuned'), {})[s] = np.stack([r['waveform_proc'] for r in augmented])
    spec_rows = []
    for s in SOURCES:
        medians = {key: np.median(absolute_power(w[s].astype(float))[1], axis=0) for key, w in waves.items()}
        f = absolute_power(waves['present', 'none'][s][:1].astype(float))[0]
        reference = medians['present', 'none'].max()
        for (label, augmentation), median in medians.items():
            with np.errstate(divide='ignore'):
                db = 10 * np.log10(median / reference)
            spec_rows += [{'source': s, 'label': label, 'augmentation': augmentation,
                           'n_waveforms': len(waves[label, augmentation][s]), 'freq_hz': fi, 'db': di}
                          for fi, di in zip(f, db)]
    spectra = pd.DataFrame(spec_rows)
    spectra.to_csv(OUT_DIR / 'spectra.csv', index=False, float_format='%.6g')

    # --- residual noise of the absent waveforms ---------------------------------------------
    def rms(records: list[dict]) -> np.ndarray:
        return np.array([np.sqrt(np.mean((r['waveform_proc'] - r['waveform_proc'].mean()) ** 2)) for r in records])
    noise_rows = []
    for s in SOURCES:
        absent, noisy, sigmas = [], [], []
        for part in POOLED.get(s, [s]):          # each part of a pooled source has its own augmentation
            recs = norm[part] if part in HUMAN else apply_label_rule(norm[part], LABEL_RULE)
            part_absent = [r for r in recs if r['label'] == 1]
            sigma = selected.get(AUGMENTATION_SOURCE.get(part, part), {}).get('pink_noise_sigma')
            absent += part_absent
            noisy += noisy_copy(part_absent, {'pink_noise_sigma': sigma}, NOISE_SEED) if sigma else part_absent
            sigmas.append('-' if sigma is None else f'{sigma:g}')
        a, b = rms(absent), rms(noisy)
        noise_rows.append({'source': s, 'n_waveforms': len(absent), 'rms_mean': a.mean(), 'rms_sd': a.std(ddof=1),
                           'rms_median': np.median(a), 'rms_p5': np.percentile(a, 5), 'rms_p95': np.percentile(a, 95),
                           'pink_noise_sigma': ' / '.join(sigmas),
                           'rms_augmented_mean': b.mean(), 'rms_augmented_sd': b.std(ddof=1),
                           'rms_augmented_median': np.median(b), 'rms_augmented_p5': np.percentile(b, 5),
                           'rms_augmented_p95': np.percentile(b, 95)})
    noise = pd.DataFrame(noise_rows)
    noise.to_csv(OUT_DIR / 'residual_noise.csv', index=False, float_format='%.4g')

    # --- summary ---------------------------------------------------------------------------
    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    g = wv.groupby('source')
    table = pd.DataFrame({
        'subjects': g.size(),
        'level dB, median (min-max), own scale': g['level_db'].agg(lambda v: f'{v.median():g} ({v.min():g}-{v.max():g})'),
        'latency ms, median (5-95%)': g['latency_ms'].agg(pct),
        'latency ms, mean +- SD': g['latency_ms'].agg(lambda v: f'{v.mean():.2f} +- {v.std():.2f}'),
        'latency ms after the tuned time shift, mean +- SD':
            g['latency_shifted_ms'].agg(lambda v: f'{v.mean():.2f} +- {v.std():.2f}'),
        'amplitude uV, median (5-95%)': g['amplitude_uv'].agg(lambda v: pct(v.dropna()) if v.notna().any() else '-'),
        'amplitude uV, SD': g['amplitude_uv'].std().map(lambda v: '-' if np.isnan(v) else f'{v:.3f}'),
        'amplitude uV, CV': (g['amplitude_uv'].std() / g['amplitude_uv'].mean()).map(lambda v: '-' if np.isnan(v) else f'{v:.2f}'),
        'amplitude recorded units, CV': (g['amplitude_recorded_units'].std()
                                         / g['amplitude_recorded_units'].mean()).map('{:.2f}'.format),
        'ambiguous picks': g['ambiguous'].mean().map('{:.0%}'.format),
    }).T.reindex(columns=list(SOURCES)).rename(columns=SOURCES)
    out(f'=== WAVE V AT THE HIGHEST LEVEL LABELLED PRESENT, one value per subject (PhysioNet: per subject and '
        f'frequency); search window {PEAK_WINDOW_MS}, trough within {TROUGH_MS} ms ===')
    out(table.to_string())
    names = {'n_subjects': 'n', 'within_rm': 'within RM', **{f'vs_{h}_rm': f'vs {SOURCES[h]} RM' for h in HUMAN}}
    for augmentation, title in [('none', 'clean waveforms'),
                                ('tuned', "after one draw of the source's tuned augmentation (the human waveforms compared "
                                          'with stay clean)')]:
        out(f'\n=== SIMILARITY OF THE NORMALISED WAVEFORMS AT THAT LEVEL, {title}: mean over pairs of subjects of RM '
            f'(largest normalised cross-correlation over all lags, as the template detector) ===')
        c = corr[corr['augmentation'] == augmentation].set_index('source').drop(columns='augmentation')
        out(c.rename(index=SOURCES, columns=names)[list(names.values())].to_string(float_format='{:.3f}'.format))
    out(f'\n=== RESIDUAL NOISE OF THE WAVEFORMS LABELLED ABSENT, normalised waveforms (fraction of the subject\'s '
        f'min-max range): RMS over waveforms, mean +- SD and median (5-95%); synthetic absent = {LABEL_RULE} rule ===')
    n2 = noise.set_index(noise['source'].map(SOURCES).rename('source'))
    nt = pd.DataFrame({
        'waveforms': n2['n_waveforms'],
        'RMS as is, mean +- SD': n2.apply(lambda r: f'{r.rms_mean:.3f} +- {r.rms_sd:.3f}', axis=1),
        'RMS as is, median (5-95%)': n2.apply(lambda r: f'{r.rms_median:.3f} ({r.rms_p5:.3f}-{r.rms_p95:.3f})', axis=1),
        'pink noise sigma (tuned)': n2['pink_noise_sigma'],
        'RMS after pink noise, mean +- SD': n2.apply(
            lambda r: f'{r.rms_augmented_mean:.3f} +- {r.rms_augmented_sd:.3f}', axis=1),
        'RMS after pink noise, median (5-95%)': n2.apply(
            lambda r: f'{r.rms_augmented_median:.3f} ({r.rms_augmented_p5:.3f}-{r.rms_augmented_p95:.3f})', axis=1),
    })
    out(nt.to_string())
    out('\n=== POWER SPECTRA ON ONE SCALE PER SOURCE: peak of the median spectrum over waveforms, dB relative to '
        "the peak of the source's clean present spectrum (every waveform labelled present or absent, synthetic by "
        f'the {LABEL_RULE} rule; tuned: one draw of the tuned augmentation; -inf: mostly flat waveforms) ===')
    peaks = spectra.groupby(['source', 'label', 'augmentation'])['db'].max().unstack(['label', 'augmentation'])
    peaks = peaks.reindex(index=list(SOURCES), columns=[(l, a) for l in ('present', 'absent') for a in ('none', 'tuned')])
    peaks.columns = [f'{l}, {"clean" if a == "none" else "augmented"}' for l, a in peaks.columns]
    out(peaks.rename(index=SOURCES).to_string(float_format='{:.1f}'.format))

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
