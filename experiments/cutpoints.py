"""Label cutpoint: training on every level with a dB cutpoint instead of the extreme levels only (0-20 vs 70-90 dB).

Label rules (src.data.apply_label_rule):
  paper     0-20 dB absent, 70-90 dB present, other levels left out
  cut<X>    all 19 levels (0-90 dB), present if level > X dB
Every rule uses the same subjects per run (the split depends only on sources, n_dev and cv).
Roenne+Verhulst, EEGConformer, no augmentation.

Summary: analysis/cutpoints.py. Run from the repository root:
  python experiments/cutpoints.py
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
SOURCES = [('roenne', 'verhulst')]
N_DEV = 64                           # 48 training + 16 validation subjects
ARCHS = ['eegconformer']
LABEL_RULES = ['paper', 'cut50', 'cut40', 'cut30', 'cut20']
AUGMENTATION = 'none'
N_RUNS = 30                          # Monte Carlo runs cv = 0 .. N_RUNS - 1; raising it adds runs
HPARAMS = {'early_stopping_steps': 60}   # overrides of src.train.HPARAMS
# Run names do not include HPARAMS, so a change of HPARAMS needs a new folder
OUT_DIR = REPOSITORY_ROOT / 'results' / 'cutpoints'


def main() -> None:
    specs = grid(SOURCES, [N_DEV], ARCHS, LABEL_RULES, [AUGMENTATION], N_RUNS)
    print(f'{len(specs)} runs -> {OUT_DIR}')
    run_grid(specs, OUT_DIR, hparams=HPARAMS)


if __name__ == '__main__':
    main()
