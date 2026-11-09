"""Real versus synthetic training, with the non-physiological controls.

SETTING selects the training setting:
  noaug   no augmentation: real, Roenne, Verhulst, Roenne+Verhulst, controls
  aug     the tuned augmentation (results/tune_augmentation/selected.json):
          the same sources plus MixUp and CutMix
Every source has 8 development subjects, the size of each real dataset: 6 training + 2
validation (two-source and mixing conditions 3 + 3 and 1 + 1). EEGConformer, early stopping
60 steps (as in experiments/cutpoints.py), N_RUNS runs. Synthetic sources and controls use the
default label rule (src.run.LABEL_RULE, cut20); real-data sources use their expert labels (the
label rule in their run names is not applied). Real-data runs are not tested on their own
dataset.

Summary: analysis/real_vs_synthetic.py. Run from the repository root:
  python experiments/real_vs_synthetic.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.run import grid, run_grid

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 'noaug' or 'aug'; 2_run_experiments.py sets it through ABR_SETTING
SETTING = os.environ.get('ABR_SETTING', 'aug')
HUMAN = [('wang',), ('southampton',), ('physionet',)]
SYNTHETIC = [('roenne',), ('verhulst',), ('roenne', 'verhulst')]
MIXING = [('mixup',), ('cutmix',)]
CONTROLS = [('gaussian',), ('simple_gaussian',), ('gaussian_latency',)]
SETTINGS = {
    'noaug': {'augmentation': 'none', 'sources': HUMAN + SYNTHETIC + CONTROLS},
    'aug': {'augmentation': 'tuned', 'sources': HUMAN + SYNTHETIC + MIXING + CONTROLS},
}
N_DEV = 8                            # 6 training + 2 validation subjects
ARCHS = ['eegconformer']
N_RUNS = 30                          # Monte Carlo runs cv = 0 .. N_RUNS - 1; raising it adds runs
HPARAMS = {'early_stopping_steps': 60}   # overrides of src.train.HPARAMS
# Run names do not include HPARAMS, so a change of HPARAMS needs a new folder
OUT_DIR = REPOSITORY_ROOT / 'results' / f'real_vs_synthetic_{SETTING}'


def main() -> None:
    setting = SETTINGS[SETTING]
    specs = grid(setting['sources'], [N_DEV], ARCHS, augmentations=[setting['augmentation']], n_runs=N_RUNS)
    print(f'{len(specs)} runs -> {OUT_DIR}')
    run_grid(specs, OUT_DIR, hparams=HPARAMS)


if __name__ == '__main__':
    main()
