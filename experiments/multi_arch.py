"""Several architectures, with the non-physiological controls.

The five waveform architectures, each trained with the tuned augmentation (selected
with EEGConformer, results/tune_augmentation/selected.json, applied unchanged to every
waveform architecture), and CWT-ViT, trained with its own tuned augmentation
(results/tune_augmentation_cwt_vit/selected.json, with early stopping 240 steps and validation
subjects with the same augmentation; experiments/tune_augmentation.py). Real-data sources use
their 8 development subjects (6 training + 2 validation); synthetic sources, mixing conditions
and controls use 64 (48 + 16). Early stopping 60 steps (CWT-ViT 240), N_RUNS runs, the default
label rule (src.run.LABEL_RULE, cut20; real-data sources keep their expert labels). The EEGConformer runs are
the same as in experiments/real_vs_synthetic.py (n = 8) and experiments/dev_size.py (n = 64).

Summary: analysis/multi_arch.py. Run from the repository root (finished runs are skipped):
  python experiments/multi_arch.py
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
ARCHS = ['eegconformer', 'deep4', 'atcnet', 'shallow', 'eegnet', 'cwt_vit']
HUMAN = [('wang',), ('southampton',), ('physionet',)]
SYNTHETIC = [('roenne',), ('verhulst',), ('roenne', 'verhulst'), ('mixup',), ('cutmix',), ('gaussian',)]
N_DEV_HUMAN = 8
N_DEV_SYNTHETIC = 64
AUGMENTATION = 'tuned'
N_RUNS = 30                          # Monte Carlo runs cv = 0 .. N_RUNS - 1; raising it adds runs
HPARAMS = {'early_stopping_steps': 60}   # overrides of src.train.HPARAMS
# Per-architecture overrides of HPARAMS: CWT-ViT stops before it has learned anything with 60
ARCH_HPARAMS = {'cwt_vit': {'early_stopping_steps': 240}}
# Run names do not include HPARAMS, so a change of HPARAMS needs a new folder
OUT_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'


def main() -> None:
    # Architecture by architecture, so an interrupted run has every source for the finished ones
    for arch in ARCHS:
        specs = (grid(HUMAN, [N_DEV_HUMAN], [arch], augmentations=[AUGMENTATION], n_runs=N_RUNS)
                 + grid(SYNTHETIC, [N_DEV_SYNTHETIC], [arch], augmentations=[AUGMENTATION], n_runs=N_RUNS))
        print(f'{arch}: {len(specs)} runs -> {OUT_DIR}')
        run_grid(specs, OUT_DIR, hparams={**HPARAMS, **ARCH_HPARAMS.get(arch, {})})


if __name__ == '__main__':
    main()
