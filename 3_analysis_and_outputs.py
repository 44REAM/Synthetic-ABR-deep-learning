"""Step 3 of 3: analyses, figures and tables from the runs in results/.

Each figure and table script writes into results/<name>/. Run from the repository root:

    uv run python 3_analysis_and_outputs.py
"""

import os
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent


def run(script: str, **settings: str) -> None:
    """Run one script from the repository root, with settings as environment variables."""
    print(f'\n=== {" ".join([script] + [f"{k}={v}" for k, v in settings.items()])}', flush=True)
    subprocess.run([sys.executable, script], cwd=REPOSITORY_ROOT, env={**os.environ, **settings}, check=True)


def main() -> None:
    # Label rule (Supplementary Fig. 2)
    run('analysis/cutpoints.py')
    run('figures/cutpoints.py')

    # Augmentation (Supplementary Figs. 3 and 4, Supplementary Table 7)
    run('analysis/tune_augmentation.py', ABR_ARCH='eegconformer')
    run('analysis/tune_augmentation.py', ABR_ARCH='cwt_vit')
    run('figures/tune_augmentation.py')

    # Equal development size (Fig. 2, Supplementary Fig. 5)
    run('analysis/real_vs_synthetic.py', ABR_SETTING='noaug')
    run('analysis/real_vs_synthetic.py', ABR_SETTING='aug')
    run('figures/real_vs_synthetic.py')
    run('analysis/real_vs_synthetic_cwt_vit.py')
    run('figures/real_vs_synthetic_cwt_vit.py')

    # Development-pool size (Supplementary Fig. 6)
    run('analysis/dev_size.py')
    run('figures/dev_size.py')

    # Architectures and conventional detectors (Fig. 3, Supplementary Fig. 7, Supplementary Tables 8 and 9)
    run('analysis/multi_arch.py')
    run('analysis/template.py')
    run('analysis/adjacent.py')
    run('analysis/physionet_frequency.py')
    run('analysis/threshold_metrics.py')
    run('figures/multi_arch.py')
    run('tables/threshold_metrics.py')

    # Feature attribution (Figs. 4 and 5)
    run('analysis/attribution.py')
    run('analysis/attribution_image.py')
    run('figures/attribution.py')

    # Simulated auditory dysfunction (Fig. 6)
    run('analysis/sensitivity.py')
    run('figures/sensitivity.py')

    # Waveform variability (Table 1, Supplementary Table 2, Supplementary Fig. 1)
    run('analysis/variability.py')
    run('figures/variability.py')

    # Synthetic waveform examples (Supplementary Figs. 8 and 9)
    run('figures/synthetic_waveforms.py')

    # Label and parameter counts (Supplementary Tables 1 and 5)
    run('analysis/label_counts.py')
    run('analysis/param_counts.py')


if __name__ == '__main__':
    main()
