"""Figure of analysis/real_vs_synthetic_cwt_vit.py (CWT-ViT, 8 development subjects per source),
drawn with figures/real_vs_synthetic.py. Rows: no augmentation and the tuned augmentation;
columns: the three test sets. The y axis starts at 0.
Writes real_vs_synthetic_cwt_vit.png (and .pdf) into results/real_vs_synthetic_cwt_vit/figures/.

Run from the repository root:
  python figures/real_vs_synthetic_cwt_vit.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from figures.real_vs_synthetic import SETTINGS, draw

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'real_vs_synthetic_cwt_vit'
Y_LIM = (0.0, 1.0)


def main() -> None:
    draw({s: RESULTS_DIR / s for s in SETTINGS}, RESULTS_DIR / 'figures', 'real_vs_synthetic_cwt_vit', Y_LIM)


if __name__ == '__main__':
    main()
