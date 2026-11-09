"""Synthetic development-pool size, with the non-physiological controls.

EEGConformer trained on N_DEVS synthetic development subjects (75 % training, 25 % validation;
Roenne+Verhulst half from each), without augmentation and with the tuned augmentation
(results/tune_augmentation/selected.json). Early stopping 60 steps, N_RUNS runs, the default
label rule (src.run.LABEL_RULE, cut20). For every run a new subset is drawn and split.
The n = 8 runs are the same as in experiments/real_vs_synthetic.py (identical specifications).

Summary: analysis/dev_size.py. Run from the repository root (finished runs are skipped):
  python experiments/dev_size.py
"""

from __future__ import annotations

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
SOURCES = [('roenne',), ('verhulst',), ('roenne', 'verhulst'), ('gaussian',)]
N_DEVS = [8, 16, 32, 64, 96, 128]
AUGMENTATIONS = ['none', 'tuned']
ARCH = 'eegconformer'
N_RUNS = 30                          # Monte Carlo runs cv = 0 .. N_RUNS - 1; raising it adds runs
HPARAMS = {'early_stopping_steps': 60}   # overrides of src.train.HPARAMS
# Run names do not include HPARAMS, so a change of HPARAMS needs a new folder
OUT_DIR = REPOSITORY_ROOT / 'results' / 'dev_size'


def main() -> None:
    # Smallest pools first, so an interrupted run has every source for the finished sizes
    specs = [spec for n in N_DEVS for aug in AUGMENTATIONS
             for spec in grid(SOURCES, [n], [ARCH], augmentations=[aug], n_runs=N_RUNS)]
    print(f'{len(specs)} runs -> {OUT_DIR}')
    run_grid(specs, OUT_DIR, hparams=HPARAMS)


if __name__ == '__main__':
    main()
