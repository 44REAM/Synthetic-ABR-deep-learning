"""Verhulst subjects with randomised brainstem stages.

Each of N_SUBJECTS simulated subjects draws the ten cochlear-nucleus (CN) and inferior-colliculus
(IC) parameters of verhulst2018 (Tex, Tin, dly, A and S of each stage) uniformly within +-SPREAD of
their AMT 1.6.0 defaults, and a cochlear-irregularity seed from the range of the training data; all
other parameters keep their defaults (hearing profile 'Normal', numH/M/L = 13/3/3, no extra ABR
delay). The waveforms are made with the generator of the training data (src/synthetic_verhulst.py:
Elberling click, brainstem delays, no noise) at STIM_LEVELS. Subject i draws from
np.random.default_rng([SEED, i]).
Writes datasets/verhulst_brainstem/verhulst_<index>/ (existing subjects are skipped); labels.csv
also holds the drawn parameters.

Run from the repository root, with the Python that has the Verhulst model's dependencies on
PATH (AMT's Verhulst model calls `python`):
  python src/synthetic_verhulst_brainstem.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from tqdm import tqdm

import src.synthetic_verhulst as verhulst
from src.synthetic_shared import load_generated_dataset, save_subject_dataset, subject_dataset_exists

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUT_DIR = REPOSITORY_ROOT / 'datasets' / 'verhulst_brainstem'
N_SUBJECTS = 20
STIM_LEVELS = [20, 50, 70]               # dB peSPL
SEED = 20261024
SPREAD = 0.20                            # uniform within default x (1 -+ SPREAD)
# verhulst2018 defaults of the CN and IC stages (amtoolbox-1.6.0/defaults/arg_verhulst2018.m)
BRAINSTEM_DEFAULTS = {'Tex_cn': 0.5e-3, 'Tin_cn': 2e-3, 'dly_cn': 1e-3, 'Acn': 1.5, 'Scn': 0.6,
                      'Tex_ic': 0.5e-3, 'Tin_ic': 2e-3, 'dly_ic': 2e-3, 'Aic': 1.0, 'Sic': 1.5}
FIBRES = (13, 3, 3)                      # numH, numM, numL


def subject_parameters(index: int) -> tuple[int, dict[str, float]]:
    """Cochlear-irregularity seed and brainstem parameters of subject `index`."""
    rng = np.random.default_rng([SEED, index])
    brainstem = {name: float(rng.uniform(value * (1.0 - SPREAD), value * (1.0 + SPREAD)))
                 for name, value in BRAINSTEM_DEFAULTS.items()}
    lo, hi = verhulst.SUBJECT_SEED_RANGE
    return int(rng.integers(lo, hi + 1)), brainstem


def main() -> None:
    pending = [i for i in range(N_SUBJECTS) if not subject_dataset_exists(str(OUT_DIR), f'verhulst_{i:05d}')]
    print(f'{N_SUBJECTS} subjects, {len(pending)} to simulate -> {OUT_DIR}')
    if pending:
        if not verhulst._init_engine():
            raise RuntimeError('Could not start the MATLAB engine')
        h, m, l = FIBRES
        config = {
            'intensities': [float(v) for v in STIM_LEVELS],
            'stimulus_family': verhulst.DEFAULT_STIMULUS_FAMILY,
            'hearing_profiles': verhulst.DEFAULT_HEARING_PROFILES,
            'num_h_range': verhulst.NUM_H_RANGE,
            'num_m_range': verhulst.NUM_M_RANGE,
            'num_l_range': verhulst.NUM_L_RANGE,
            'subject_seed_range': verhulst.SUBJECT_SEED_RANGE,
            'abr_delay_ms_range': verhulst.ABR_DELAY_MS_RANGE,
            'subject_param_modes': dict(verhulst.SUBJECT_PARAM_MODES),
            'label_criterion': verhulst.LABEL_CRITERION,
            'db_threshold': verhulst.DB_THRESHOLD,
            'amplitude_threshold': verhulst.ABSENT_LABEL_THRESHOLD,
            'seed': SEED,
        }
        for i in tqdm(pending, desc='Simulating subjects (Verhulst brainstem)'):
            subject_seed, brainstem = subject_parameters(i)
            fixed = {'hearing_profile': 'Normal', 'num_h': h, 'num_m': m, 'num_l': l,
                     'subject_seed': subject_seed, 'abr_delay_ms': 0.0}
            result = verhulst.simulate_subject(i, dict(config, fixed_params=fixed, model_options=brainstem))
            if result is None:
                raise RuntimeError(f'Subject {i} failed')
            subject_id, waveforms, records, an_populations = result
            rows = [{**{c: r[c] for c in verhulst._LABEL_COLUMNS}, **brainstem} for r in records]
            save_subject_dataset(str(OUT_DIR), subject_id, waveforms, rows, None, an_populations)

    waveforms, labels = load_generated_dataset(str(OUT_DIR))
    print(f'Saved {len(labels)} records, waveforms {waveforms.shape}')


if __name__ == '__main__':
    main()
