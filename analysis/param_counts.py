"""Parameter counts of the architectures as trained in this study.

Each architecture is built with src.models.build_model, as in training (1 channel, 300
samples at 20 kHz, 1 output). Counted: trainable parameters, all parameters, and buffers
(e.g. BatchNorm running statistics, CWT-ViT's CWT scaling range), which are not parameters.
Each count is checked against a trained model: one checkpoint per architecture from
results/multi_arch is loaded into the counted model with strict=True.
Writes param_counts.csv and summary.txt into results/param_counts/.

Run from the repository root:
  python analysis/param_counts.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd
import torch

from src.models import ARCHS, build_model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CHECKPOINT_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch' / 'checkpoints'
CHECK_RUN = 'roenne__n64__{arch}__cut20__tuned__cv0'      # checkpoint loaded for each architecture
OUT_DIR = REPOSITORY_ROOT / 'results' / 'param_counts'
NAMES = {'eegconformer': 'EEGConformer', 'deep4': 'DeepConvNet', 'atcnet': 'ATCNet',
         'shallow': 'ShallowConvNet', 'eegnet': 'EEGNet', 'cwt_vit': 'CWT-ViT'}
SUBMITTED = {'eegconformer': 274593, 'deep4': 161751, 'atcnet': 29458, 'shallow': 3281, 'eegnet': 1265}


def main() -> None:
    rows = []
    for arch in NAMES:
        if arch not in ARCHS:
            raise ValueError(f'{arch} is not in src.models.ARCHS')
        model = build_model(arch)
        run = CHECK_RUN.format(arch=arch)
        state = torch.load(CHECKPOINT_DIR / f'{run}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)             # raises if the trained model differs
        rows.append({'arch': arch, 'name': NAMES[arch],
                     'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'all_params': sum(p.numel() for p in model.parameters()),
                     'buffer_values': sum(b.numel() for b in model.buffers()),
                     'submitted_table': SUBMITTED.get(arch),
                     'checked_against': run})
    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / 'param_counts.csv', index=False)
    shown = table.drop(columns='arch').set_index('name')
    shown['submitted_table'] = shown['submitted_table'].map(lambda v: '-' if pd.isna(v) else f'{int(v):,}')
    for col in ('trainable_params', 'all_params', 'buffer_values'):
        shown[col] = shown[col].map('{:,}'.format)
    pd.set_option('display.width', 200)
    text = (f'Parameter counts as trained (1 channel, 300 samples, 1 output); each loaded from its checkpoint '
            f'with strict=True\n{shown.to_string()}\n')
    print(text)
    (OUT_DIR / 'summary.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
