"""Non-physiological controls: synthetic ABRs made of a single Gaussian peak. Two matched
controls copy the wave V latency of the physiological synthetic subjects, one of them
(gaussian) the wave V amplitude too and one (gaussian_latency) with the full peak or nothing;
a model-free control (simple_gaussian) copies nothing from them.

Matched controls. Each control subject copies one physiological subject: 250 drawn from datasets/synthetic_roenne
and 250 from datasets/synthetic_verhulst (without replacement; the same subjects for every
control). At every level the peak is placed at that subject's measured wave V latency with its
measured wave V amplitude (labels.csv: the largest value in the 4-12 ms window). Where that
maximum lies at a window edge there is no wave V peak (Verhulst 0-15 dB) and the amplitude is 0.
No auditory model, no noise.

Shapes, with the shape parameters drawn uniformly once per subject:
  gaussian      Gaussian peak, sigma 0.3-1.0 ms
  gaussian_latency  the Gaussian control with its amplitude structure removed:
                the same copied subjects and the same sigma draws, the peak at the same measured
                latency at every level, but with the amplitude SIMPLE_AMPLITUDE_UV at every level
                above SIMPLE_ABSENT_MAX_DB (the present levels of the cut20 rule) and a flat line
                at the levels up to it, so each waveform is the full peak or nothing. Where the
                copied subject has no wave V peak the waveform is flat, as in the Gaussian control
                (a few Roenne subjects at 25-35 dB). Its labels.csv follows the matched controls.

Model-free control (simple_gaussian). One Gaussian peak per subject, its
centre drawn uniformly from 2-13 ms after the stimulus (the 15 ms analysis window minus a 2 ms
margin, two standard deviations of the widest peak, so the whole peak lies inside the window)
and its width from the same 0.3-1.0 ms, with the same amplitude at every level. The levels up to
20 dB are a flat line (the absent levels of the cut20 label rule, src.run.LABEL_RULE), every
level above it carries the identical peak: no level-dependent amplitude or latency, no
threshold, no auditory model, no noise. Its
labels.csv stores the peak itself (wave_v_latency_ms = the centre and wave_v_amplitude_uv = the
amplitude on the levels with the peak, NaN and 0 on the flat levels) and wave_v_present = 1 on
the levels with the peak, not the 4-12 ms measurement and the 50 dB criterion of the matched
controls, which copy those from the models; it has no source_* columns.
Output per control: datasets/synthetic_<name>, 500 subjects x 19 levels with the structure of the
Roenne and Verhulst datasets (waveforms.npy + labels.csv per subject, 20 kHz, 15 ms
pre-stimulus reference).
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd

from src.synthetic_shared import (
    compute_label,
    load_generated_dataset,
    save_subject_dataset,
    subject_dataset_exists,
)

DATASETS = REPOSITORY_ROOT / 'datasets'
NUM_SUBJECTS = 500
SUBJECTS_PER_SOURCE = {'roenne': 250, 'verhulst': 250}   # physiological subjects copied
MATCH_SEED = 20261024                # which physiological subjects are copied, and their order
SHAPE_SEEDS = {'gaussian': 20261022,
               'gaussian_latency': 20261022}   # the Gaussian control's seed: the same sigma per subject

GAUSSIAN_SIGMA_MS = (0.3, 1.0)       # Gaussian width (standard deviation)

WAVE_V_WINDOW_MS = (4.0, 12.0)       # window of the wave V measurement (compute_label)
EDGE_TOLERANCE_MS = 0.05             # a maximum this close to a window edge is no peak

# Label criterion, as the Roenne and Verhulst generators
DB_THRESHOLD = 50.0                  # wave_v_present = level > DB_THRESHOLD

SAMPLE_RATE = 20_000.0
PRE_SAMP = 300                       # 15 ms pre-stimulus reference
N_SAMPLES = 1000                     # 50 ms

T_MS = (np.arange(N_SAMPLES) - PRE_SAMP) / SAMPLE_RATE * 1000.0  # time re. stimulus onset

_LABEL_COLUMNS = [
    'record_id',
    'subject_id',
    'intensity_db',
    'wave_v_present',
    'wave_v_latency_ms',
    'wave_v_amplitude_uv',
    'model',
    # Dataset-level constants (repeated per row for self-contained records)
    'native_sample_rate_hz',
    'pre_samp',
    'n_avg',
    # The copied physiological subject and its wave V at this level
    'source_model',
    'source_subject_id',
    'source_latency_ms',
    'source_amplitude_uv',
    # Per-subject shape parameters
    'shape_params',
]


def gaussian_peak(latency_ms: float, amp_uv: float, sigma_ms: float) -> np.ndarray:
    return amp_uv * np.exp(-0.5 * ((T_MS - latency_ms) / sigma_ms) ** 2)


SHAPES = {
    'gaussian': (gaussian_peak, {'sigma_ms': GAUSSIAN_SIGMA_MS}),
    'gaussian_latency': (gaussian_peak, {'sigma_ms': GAUSSIAN_SIGMA_MS}),
}

# Model-free control (module docstring)
SIMPLE_NAME = 'simple_gaussian'
SIMPLE_SEED = 20261031
SIMPLE_LATENCY_MS = (2.0, 13.0)      # peak centre, uniform per subject; width from GAUSSIAN_SIGMA_MS
SIMPLE_AMPLITUDE_UV = 0.3            # the same at every level (removed by the per-subject normalisation)
SIMPLE_ABSENT_MAX_DB = 20.0          # levels up to this are flat: the absent levels of the cut20 label rule
LEVELS_DB = [float(v) for v in range(0, 95, 5)]   # the 19 levels of the Roenne and Verhulst datasets
# Matched controls with the full peak (SIMPLE_AMPLITUDE_UV) above SIMPLE_ABSENT_MAX_DB and nothing below
CONSTANT_AMPLITUDE = {'gaussian_latency'}


def matched_subjects() -> list[tuple[str, str, pd.DataFrame]]:
    """(source model, source subject id, its labels.csv rows) for every control subject, in order."""
    rng = np.random.default_rng(MATCH_SEED)
    chosen = []
    for model, n in SUBJECTS_PER_SOURCE.items():
        folders = sorted(p for p in (DATASETS / f'synthetic_{model}').iterdir() if p.is_dir())
        for k in sorted(rng.choice(len(folders), n, replace=False)):
            chosen.append((model, folders[k].name, pd.read_csv(folders[k] / 'labels.csv')))
    order = rng.permutation(len(chosen))
    return [chosen[i] for i in order]


def peak_amplitude(latency_ms: float, amplitude_uv: float) -> float:
    """The measured wave V amplitude, or 0 where the measurement found no peak inside the window."""
    lo, hi = WAVE_V_WINDOW_MS
    if not np.isfinite(latency_ms) or latency_ms <= lo + EDGE_TOLERANCE_MS or latency_ms >= hi - EDGE_TOLERANCE_MS:
        return 0.0
    return max(float(amplitude_uv), 0.0)


def simulate_subject(name: str, subject_index: int, source: tuple[str, str, pd.DataFrame],
                     params: dict[str, float]) -> tuple[str, list[np.ndarray], list[dict]]:
    shape, _ = SHAPES[name]
    source_model, source_id, source_labels = source
    subject_id = f'{name}_{subject_index:05d}'
    waveforms, records = [], []
    for level_index, row in enumerate(source_labels.sort_values('intensity_db').itertuples()):
        amp = peak_amplitude(row.wave_v_latency_ms, row.wave_v_amplitude_uv)
        if name in CONSTANT_AMPLITUDE:       # the full peak or nothing
            amp = SIMPLE_AMPLITUDE_UV if amp > 0.0 and float(row.intensity_db) > SIMPLE_ABSENT_MAX_DB else 0.0
        if amp > 0.0:
            wf = shape(row.wave_v_latency_ms, amp, **params).astype(np.float32)
        else:
            wf = np.zeros(N_SAMPLES, dtype=np.float32)

        # Measured wave V (largest value in 4-12 ms), as the other generators
        _, latency_ms, amplitude_uv = compute_label(wf, T_MS, threshold_uv=-np.inf)
        waveforms.append(wf)
        records.append({
            'record_id': subject_index * len(source_labels) + level_index,
            'subject_id': subject_id,
            'intensity_db': float(row.intensity_db),
            'wave_v_present': int(float(row.intensity_db) > DB_THRESHOLD),
            'wave_v_latency_ms': round(latency_ms, 3),
            'wave_v_amplitude_uv': round(amplitude_uv, 4),
            'model': name,
            'native_sample_rate_hz': SAMPLE_RATE,
            'pre_samp': PRE_SAMP,
            'n_avg': 1,
            'source_model': source_model,
            'source_subject_id': source_id,
            'source_latency_ms': row.wave_v_latency_ms,
            'source_amplitude_uv': row.wave_v_amplitude_uv,
            'shape_params': ';'.join(f'{k}={v:.4f}' for k, v in params.items()),
        })
    return subject_id, waveforms, records


def generate_dataset(name: str, n_subjects: int = NUM_SUBJECTS) -> None:
    out_dir = str(DATASETS / f'synthetic_{name}')
    sources = matched_subjects()[:n_subjects]
    _, ranges = SHAPES[name]
    rng = np.random.default_rng(SHAPE_SEEDS[name])
    print(f'[{name.upper()}] Generating {len(sources)} subjects…')

    for subject_index, source in enumerate(sources):
        # Always draw, in this order, so every subject keeps its parameters when resuming
        params = {k: float(rng.uniform(*r)) for k, r in ranges.items()}
        if subject_dataset_exists(out_dir, f'{name}_{subject_index:05d}'):
            continue
        subject_id, waveforms, records = simulate_subject(name, subject_index, source, params)
        save_subject_dataset(out_dir, subject_id, waveforms,
                             [{col: rec[col] for col in _LABEL_COLUMNS} for rec in records])

    waveforms_np, labels_df = load_generated_dataset(out_dir)
    print(f'[{name.upper()}] Saved {len(labels_df)} records -> {out_dir}')
    print(f'           Shape: {waveforms_np.shape}  dtype: {waveforms_np.dtype}')
    print(f'           Sources: {labels_df.drop_duplicates("subject_id")["source_model"].value_counts().to_dict()}')


def simulate_simple_subject(subject_index: int, latency_ms: float, sigma_ms: float) -> tuple[str, list[np.ndarray], list[dict]]:
    subject_id = f'{SIMPLE_NAME}_{subject_index:05d}'
    peak = gaussian_peak(latency_ms, SIMPLE_AMPLITUDE_UV, sigma_ms).astype(np.float32)
    flat = np.zeros(N_SAMPLES, dtype=np.float32)
    waveforms, records = [], []
    for level_index, level in enumerate(LEVELS_DB):
        present = level > SIMPLE_ABSENT_MAX_DB
        waveforms.append(peak if present else flat)
        records.append({
            'record_id': subject_index * len(LEVELS_DB) + level_index,
            'subject_id': subject_id,
            'intensity_db': level,
            'wave_v_present': int(present),
            'wave_v_latency_ms': round(latency_ms, 3) if present else float('nan'),
            'wave_v_amplitude_uv': SIMPLE_AMPLITUDE_UV if present else 0.0,
            'model': SIMPLE_NAME,
            'native_sample_rate_hz': SAMPLE_RATE,
            'pre_samp': PRE_SAMP,
            'n_avg': 1,
            'shape_params': f'latency_ms={latency_ms:.4f};sigma_ms={sigma_ms:.4f}',
        })
    return subject_id, waveforms, records


def generate_simple_dataset(n_subjects: int = NUM_SUBJECTS) -> None:
    out_dir = str(DATASETS / f'synthetic_{SIMPLE_NAME}')
    rng = np.random.default_rng(SIMPLE_SEED)
    print(f'[{SIMPLE_NAME.upper()}] Generating {n_subjects} subjects...')
    for subject_index in range(n_subjects):
        # Always draw, in this order, so every subject keeps its parameters when resuming
        latency_ms = float(rng.uniform(*SIMPLE_LATENCY_MS))
        sigma_ms = float(rng.uniform(*GAUSSIAN_SIGMA_MS))
        if subject_dataset_exists(out_dir, f'{SIMPLE_NAME}_{subject_index:05d}'):
            continue
        subject_id, waveforms, records = simulate_simple_subject(subject_index, latency_ms, sigma_ms)
        save_subject_dataset(out_dir, subject_id, waveforms, records)

    waveforms_np, labels_df = load_generated_dataset(out_dir)
    print(f'[{SIMPLE_NAME.upper()}] Saved {len(labels_df)} records -> {out_dir}')
    print(f'           Shape: {waveforms_np.shape}  dtype: {waveforms_np.dtype}')


def main() -> None:
    for name in SHAPES:
        generate_dataset(name)
    generate_simple_dataset()


if __name__ == '__main__':
    main()
