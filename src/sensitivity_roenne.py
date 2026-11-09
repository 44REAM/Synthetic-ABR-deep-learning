"""Sensitivity analysis data generation using the Roenne 2012 MATLAB model."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm
import matlab.engine

from src.synthetic_shared import (
    compute_label,
    save_subject_dataset,
    subject_dataset_exists,
)

# AMT 1.6.0 as used for the paper (see src/synthetic_roenne.py and README.md)
AMT_ROOT = str(REPOSITORY_ROOT / 'amtoolbox-1.6.0')
HELPER_ROOT = str(Path(__file__).resolve().parent / 'matlab')
OUT_DIR = str(REPOSITORY_ROOT / 'datasets' / 'sensitivity_roenne')
STIM_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

_LABEL_COLUMNS = [
    'record_id',
    'subject_id',
    'intensity_db',
    'wave_v_present',
    'wave_v_latency_ms',
    'wave_v_amplitude_uv',
    'snr_db',
    'model',
    # Dataset-level constants
    'native_sample_rate_hz',
    'pre_samp',
    'n_avg',
    # Stimulus / cochlear-coverage settings
    'stimulus_family',
    'flow_hz',
    'fhigh_hz',
    'num_cf',
    # Single-kernel readout setting
    'ur_delay_ms',
    # Per-subject generation parameters
    'cohc',
    'cihc',
    'spont',
    'sigma_eeg',
]

def main() -> None:
    print("[SENSITIVITY ROENNE] Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(AMT_ROOT), nargout=0)
    eng.addpath(HELPER_ROOT, nargout=0)
    eng.amt_start(nargout=0)
    print("[SENSITIVITY ROENNE] MATLAB engine ready.")

    # Generate 21 unique combinations of cohc and cihc (spont = 50)
    combinations = []
    # Vary cohc from 0.0 to 1.0 (step 0.1) with cihc = 1.0 (11 samples)
    for cohc_val in np.arange(0.0, 1.1, 0.1):
        combinations.append((round(float(cohc_val), 1), 1.0))
    # Vary cihc from 0.0 to 0.9 (step 0.1) with cohc = 1.0 (10 samples)
    for cihc_val in np.arange(0.0, 1.0, 0.1):
        combinations.append((1.0, round(float(cihc_val), 1)))

    print(f"[SENSITIVITY ROENNE] Prepared {len(combinations)} parameter configurations.")

    os.makedirs(OUT_DIR, exist_ok=True)

    for subject_index, (cohc, cihc) in enumerate(tqdm(combinations, desc="Simulating configurations")):
        subject_id = f'roenne_{subject_index:05d}'
        if subject_dataset_exists(OUT_DIR, subject_id):
            continue
        
        levels_mat = matlab.double([[float(level) for level in STIM_LEVELS]])
        
        try:
            # Run simulation
            simpot_matrix, fs_out, pre_samp = eng.roenne2012_varied_batch(
                levels_mat,
                'click',     # stimulus_family
                100.0,       # flow
                16000.0,     # fhigh
                500.0,       # num_cf
                float(cohc),
                float(cihc),
                50.0,        # spont
                0.0,         # ur_delay_ms
                nargout=3,
            )
        except Exception as exc:
            print(f"\n[SENSITIVITY ROENNE] MATLAB simulation failed for {subject_id}: {exc}")
            continue

        simpot_np = np.asarray(simpot_matrix, dtype=np.float32)
        if simpot_np.ndim == 1:
            simpot_np = simpot_np[:, None]

        native_sample_rate = float(np.asarray(fs_out).squeeze())
        pre_samp_int = int(round(float(np.asarray(pre_samp).squeeze())))
        dt_ms = 1000.0 / native_sample_rate

        subject_waveforms: list[np.ndarray] = []
        subject_records: list[dict[str, Any]] = []

        for intensity_index, level in enumerate(STIM_LEVELS):
            clean_post_source = simpot_np[:, intensity_index].astype(np.float32)
            t_post = (np.arange(len(clean_post_source), dtype=np.float32) - pre_samp_int) * dt_ms

            # No noise and no UR delay is added
            waveform = clean_post_source

            # Compute label with threshold_uv=0.0 to capture latency and amplitude for all waves
            _, latency_ms, amplitude_uv = compute_label(
                clean_post_source,
                t_post,
                threshold_uv=0.0,
            )

            # Mark present if amplitude is positive
            label = int(amplitude_uv > 0.0)

            record_id = subject_index * len(STIM_LEVELS) + intensity_index

            subject_waveforms.append(waveform)
            subject_records.append({
                'record_id': record_id,
                'subject_id': subject_id,
                'intensity_db': float(level),
                'wave_v_present': label,
                'wave_v_latency_ms': round(latency_ms, 3) if label else float('nan'),
                'wave_v_amplitude_uv': round(amplitude_uv, 4) if label else 0.0,
                'snr_db': 99.0,  # No noise, so SNR is effectively perfect/arbitrary
                'model': 'roenne',
                'native_sample_rate_hz': native_sample_rate,
                'pre_samp': pre_samp_int,
                'n_avg': 1,
                'stimulus_family': 'click',
                'flow_hz': 100.0,
                'fhigh_hz': 16000.0,
                'num_cf': 500,
                'ur_delay_ms': 0.0,
                'cohc': round(cohc, 4),
                'cihc': round(cihc, 4),
                'spont': 50.0,
                'sigma_eeg': 0.0,
            })

        # Save the subject dataset
        ordered_records = [
            {column: record[column] for column in _LABEL_COLUMNS}
            for record in subject_records
        ]
        save_subject_dataset(
            OUT_DIR,
            subject_id,
            subject_waveforms,
            ordered_records,
            subject_clean_waveforms=subject_waveforms,
        )

    eng.quit()
    print("\n[SENSITIVITY ROENNE] Sensitivity data generation complete.")

if __name__ == '__main__':
    main()
