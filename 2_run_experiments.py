"""Step 2 of 3: train every experiment grid into results/.

Runs already on disk are skipped, so this can be stopped and restarted. The augmentation runs use the
settings in results/tune_augmentation/selected.json and results/tune_augmentation_cwt_vit/selected.json
(included; README.md). Run from the repository root:

    uv run python 2_run_experiments.py
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
    run('experiments/cutpoints.py')                                   # label rule
    run('experiments/tune_augmentation.py', ABR_ARCH='eegconformer')  # augmentation tuning
    run('experiments/tune_augmentation.py', ABR_ARCH='cwt_vit')
    run('experiments/real_vs_synthetic.py', ABR_SETTING='noaug')      # equal development size
    run('experiments/real_vs_synthetic.py', ABR_SETTING='aug')
    run('experiments/dev_size.py')                                    # development-pool size
    run('experiments/multi_arch.py')                                  # architectures


if __name__ == '__main__':
    main()
