"""Step 1 of 3: prepare the real data and generate every synthetic dataset.

Needs the real recordings in datasets/, MATLAB R2025a and `uv sync --extra matlab` (README.md).
Subjects already on disk are skipped. Run from the repository root:

    uv run python 1_generate_data.py
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
    run('src/prepare_physionet.py')      # PhysioNet averages in µV: datasets/physionet_uv.npz
    run('src/synthetic_roenne.py')       # datasets/synthetic_roenne
    run('src/synthetic_verhulst.py')     # datasets/synthetic_verhulst
    run('src/synthetic_controls.py')     # the three Gaussian controls (from the two sets above)
    run('src/sensitivity_roenne.py')     # impairment sweeps for Fig. 6
    run('src/sensitivity_verhulst.py')
    run('src/synthetic_verhulst_brainstem.py')   # brainstem-parameter examples for Supplementary Fig. 9


if __name__ == '__main__':
    main()
