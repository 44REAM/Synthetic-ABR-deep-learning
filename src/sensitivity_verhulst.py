"""Simulated impairment sweeps of the Verhulst model for the sensitivity analysis.

One simulated subject per parameter setting (cochlear-irregularity seed 1), made with the
generator of the training data (src/synthetic_verhulst.py: Elberling click calibrated as in
the Roenne generator, brainstem delays, no noise, wave V measured at every level), at
0-90 dB in 10 dB steps. Settings:
  synaptopathy  hearing profile 'Normal'; numH 13 -> 1 (numM = numL = 3), then numM 3 -> 0
                (numH = 1), then numL 3 -> 0 (numH = 1, numM = 0): 19 settings
  OHC loss      hearing profiles Flat00 ... Flat35 in 5 dB steps (numH/M/L = 13/3/3): 8 settings
Writes datasets/sensitivity_verhulst/verhulst_<index>/ (existing subjects are skipped).

Run from the repository root, with the Python that has the Verhulst model's dependencies on
PATH (AMT's Verhulst model calls `python`):
  python src/sensitivity_verhulst.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tqdm import tqdm

import src.synthetic_verhulst as verhulst
from src.synthetic_shared import load_generated_dataset, save_subject_dataset, subject_dataset_exists

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUT_DIR = REPOSITORY_ROOT / 'datasets' / 'sensitivity_verhulst'
STIM_LEVELS = list(range(0, 100, 10))    # dB peSPL
SUBJECT_SEED = 1                         # cochlear-irregularity seed of every setting
FULL_FIBRES = (13, 3, 3)                 # numH, numM, numL without synaptopathy
FLAT_PROFILES = [f'Flat{v:02d}' for v in range(0, 40, 5)]


def settings() -> list[dict]:
    """The 27 parameter settings, in the order of the subject indices."""
    fibres = ([(h, 3, 3) for h in range(13, 0, -1)] + [(1, m, 3) for m in range(2, -1, -1)]
              + [(1, 0, l) for l in range(2, -1, -1)])
    out = [{'hearing_profile': 'Normal', 'num_h': h, 'num_m': m, 'num_l': l} for h, m, l in fibres]
    h, m, l = FULL_FIBRES
    out += [{'hearing_profile': p, 'num_h': h, 'num_m': m, 'num_l': l} for p in FLAT_PROFILES]
    return [dict(s, subject_seed=SUBJECT_SEED, abr_delay_ms=0.0) for s in out]


def main() -> None:
    sweep = settings()
    pending = [i for i in range(len(sweep)) if not subject_dataset_exists(str(OUT_DIR), f'verhulst_{i:05d}')]
    print(f'{len(sweep)} settings, {len(pending)} to simulate -> {OUT_DIR}')
    if pending:
        if not verhulst._init_engine():
            raise RuntimeError('Could not start the MATLAB engine')
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
            'seed': verhulst.SEED,
        }
        for i in tqdm(pending, desc='Simulating settings (Verhulst)'):
            result = verhulst.simulate_subject(i, dict(config, fixed_params=sweep[i]))
            if result is None:
                raise RuntimeError(f'Setting {i} ({sweep[i]}) failed')
            subject_id, waveforms, records, an_populations = result
            save_subject_dataset(str(OUT_DIR), subject_id, waveforms,
                                 [{c: r[c] for c in verhulst._LABEL_COLUMNS} for r in records], None, an_populations)

    waveforms, labels = load_generated_dataset(str(OUT_DIR))
    print(f'Saved {len(labels)} records, waveforms {waveforms.shape}')


if __name__ == '__main__':
    main()
