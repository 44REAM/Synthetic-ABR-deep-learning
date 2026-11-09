"""Augmentation tuning.

ARCH selects the architecture and its TUNE_SETTINGS. EEGConformer's setting is used by the five
waveform architectures; CWT-ViT is tuned separately (results/tune_augmentation_cwt_vit/,
selected.json there; src/run.py reads the file of the run's architecture).

Stage 1: one augmentation at a time from CONFIGS (none, time shift, pink, Gaussian and uniform
noise), for each training source separately: 8 development subjects (6 training + 2
validation), the default label rule (src.run.LABEL_RULE, cut20), and the architecture's early
stopping, number of runs and validation noise. The selected settings are in
<folder>/selected.json, which augmentation='tuned' reads.

EEGConformer
Sources: Roenne, Verhulst, the two matched controls and the simple Gaussian control, which is
used in experiments/real_vs_synthetic.py only and so tuned here only (Roenne+Verhulst uses the
settings of its two sources), and the real-data sources listed in 'tune_human'. RUN_STAGES runs one
stage at a time: stage 2 needs the stage-1 choice in selected.json first.

Stage 2: DC shift added on top of each source's selected setting ('tuned+dc<value>'), next to
the selected setting alone ('tuned'), with the same training. It covers the real-data sources too.
Run names do not record the selected setting. The stage-2 runs use the stage-1 selection: selected.json with its 'dc_shift_max' values removed
(build_specs), so they can be re-run from the final file.

CWT-ViT
Stage 1 only, on the three synthetic sources and the three real-data sources, 10 runs per
configuration, early stopping 240 steps, validation subjects with the same augmentation.
selected.json also sets 'noisy_validation' for the synthetic sources, as in the tuning.

Run from the repository root (finished runs are skipped):
  python experiments/tune_augmentation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.run import TUNED_AUGMENTATION_FILE, grid, run_grid

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 'eegconformer' or 'cwt_vit' (a key of TUNE_SETTINGS); 2_run_experiments.py sets it through ABR_ARCH
ARCH = os.environ.get('ABR_ARCH', 'eegconformer')
SYNTHETIC_SOURCES = ['roenne', 'verhulst', 'gaussian']
HUMAN_SOURCES = ['wang', 'southampton', 'physionet']
N_DEV = 8                            # 6 training + 2 validation subjects
# Per architecture:
#   hparams           overrides of src.train.HPARAMS
#   n_runs            runs per configuration
#   tune_human        human sources tuned in stage 1 with the same grid (the others keep their
#                     setting in selected.json)
#   noisy_validation  apply each configuration once to the validation subjects as well
#   dc_stage          run stage 2 (DC shift on the selected setting)
#   extra_synthetic   synthetic sources tuned for this architecture only: the simple and the
#                     latency-only Gaussian controls are used in experiments/real_vs_synthetic.py
#                     (EEGConformer) only
TUNE_SETTINGS = {
    'eegconformer': {'hparams': {'early_stopping_steps': 60}, 'n_runs': 30, 'tune_human': HUMAN_SOURCES,
                     'noisy_validation': False, 'dc_stage': True, 'folder': 'tune_augmentation',
                     'extra_synthetic': ['simple_gaussian', 'gaussian_latency']},
    # CWT-ViT: images rather than waveforms, so tuned separately. It needs early stopping 240
    # (with 60 it stops before it has learned anything) and a noisy validation set (with a
    # clean one, noise configurations stop at the start). No DC stage: CWT-ViT removes each
    # waveform's mean before the CWT, which cancels a DC shift.
    'cwt_vit': {'hparams': {'early_stopping_steps': 240}, 'n_runs': 10, 'tune_human': HUMAN_SOURCES,
                'noisy_validation': True, 'dc_stage': False, 'folder': 'tune_augmentation_cwt_vit'},
}
OUT_DIR = REPOSITORY_ROOT / 'results' / TUNE_SETTINGS[ARCH]['folder']
# Stages to run: {1} for the grid only (then choose and write selected.json), {2} for the DC
# stage on that choice, {1, 2} when selected.json already holds the stage-1 choice
RUN_STAGES = {1, 2}

# Stage-1 grid; 'randn' is Gaussian white noise. Maximum values: the amount of each draw
# is uniform between 0 and the value (DC shift: between -value and value; src/augment.py).
SHIFT_MS = [1.0, 3.0, 5.0, 7.0]
NOISE_LEVELS = [0.05, 0.1, 0.2, 0.4]
CONFIGS: dict[str, dict | None] = {
    'none': None,
    **{f'shift{v:g}': {'shift_limit_ms': v} for v in SHIFT_MS},
    **{f'pink{v:g}': {'pink_noise_sigma': v} for v in NOISE_LEVELS},
    **{f'randn{v:g}': {'randn_noise_sigma': v} for v in NOISE_LEVELS},
    **{f'uniform{v:g}': {'uniform_noise_amplitude': v} for v in NOISE_LEVELS},
}
# Stage 2: maximum DC offsets added to the selected setting
DC_SHIFTS = [0.1, 0.2, 0.5]
STACKED = ['tuned'] + [f'tuned+dc{v:g}' for v in DC_SHIFTS]


def build_specs(arch: str, stages: set[int] = frozenset({1, 2})) -> tuple[list, dict[str, dict], dict]:
    """The runs of the given stages of the architecture's tuning and the augmentation settings
    they use, with its setting."""
    setting = TUNE_SETTINGS[arch]
    n_runs = setting['n_runs']
    synthetic = SYNTHETIC_SOURCES + list(setting.get('extra_synthetic', []))
    sources = synthetic + list(setting['tune_human'])
    extra = {'noisy_validation': True} if setting['noisy_validation'] else {}
    augmentation_sets = {name: {s: {**cfg, **extra} for s in sources}
                         for name, cfg in CONFIGS.items() if cfg is not None}
    specs = []
    # Configuration by configuration, so an interrupted run has every source for the finished ones
    if 1 in stages:
        specs += [spec for name in CONFIGS
                  for spec in grid([(s,) for s in sources], [N_DEV], [arch], augmentations=[name], n_runs=n_runs)]
    if setting['dc_stage'] and 2 in stages:
        stacked_sources = synthetic + HUMAN_SOURCES
        # Stage 2 starts from the stage-1 selection: selected.json without the DC values that
        # stage 2 itself added, so the stage-2 runs are reproducible from the final file
        selected = json.loads(TUNED_AUGMENTATION_FILE.read_text())
        stage1 = {s: {k: v for k, v in selected[s].items() if k != 'dc_shift_max'} for s in stacked_sources}
        augmentation_sets['tuned'] = stage1
        augmentation_sets.update({f'tuned+dc{v:g}': {s: {**stage1[s], 'dc_shift_max': v} for s in stacked_sources}
                                  for v in DC_SHIFTS})
        specs += [spec for name in STACKED
                  for spec in grid([(s,) for s in stacked_sources], [N_DEV], [arch], augmentations=[name],
                                   n_runs=n_runs)]
    return specs, augmentation_sets, setting


def main() -> None:
    specs, augmentation_sets, setting = build_specs(ARCH, RUN_STAGES)
    print(f'{len(specs)} runs (stages {sorted(RUN_STAGES)}) -> {OUT_DIR}')
    run_grid(specs, OUT_DIR, augmentation_sets=augmentation_sets, hparams=setting['hparams'])


if __name__ == '__main__':
    main()
