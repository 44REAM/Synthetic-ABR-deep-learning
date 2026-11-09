# Synthetic-to-real wave V classification of auditory brainstem responses

Code for *Physiologically Informed Synthetic Auditory Brainstem Responses Enable Synthetic-to-Real
Wave V Classification*. Classifiers are trained on ABRs simulated with the Rønne and Verhulst models
(Auditory Modeling Toolbox, AMT 1.6.0) and tested on three real datasets (Mendeley, Southampton,
PhysioNet).

## Setup

```bash
uv sync                  # Python 3.12 and the exact package versions (uv.lock)
uv sync --extra matlab   # adds the MATLAB Engine (MATLAB R2025a), needed only to generate synthetic data
```

PyTorch comes from the CUDA 12.8 index set in `pyproject.toml`; change it for another CUDA version or CPU.

**AMT** (only to generate synthetic data): download AMT 1.6.0 from <https://amtoolbox.org>, unpack it
into `amtoolbox-1.6.0/` at the repository root and copy `amt_patch/models` and `amt_patch/environments`
over it. The two patched files let `zilany2007.m` take `cohc`, `cihc` and `spont`, and make
`run_cochlear_model.py` work with NumPy 2.

## Data

The labels (Liu et al., 2025, *Trends Hear.* 29:23312165251347773; majority vote of the labellers,
ties counted as absent) are included; step 1 generates the synthetic data. Download the recordings:

| Dataset | Download | Put it at |
|---|---|---|
| Southampton | <http://doi.org/10.5258/SOTON/D0168> | `datasets/Subject_ABR_Data.mat` |
| Mendeley | <https://data.mendeley.com/datasets/4yb9772dff/1> | `datasets/wang_single/human_raw_data/P1/` … `P8/` |
| PhysioNet | <https://physionet.org/content/earndb/1.0.0/> | `datasets/physionet.org/files/earndb/1.0.0/` (`wget -r -N -c -np https://physionet.org/files/earndb/1.0.0/` in `datasets/`), or set `PHYSIONET_DIR` to that folder |

## Run

```bash
uv run python 1_generate_data.py          # PhysioNet conversion and synthetic data (MATLAB)
uv run python 2_run_experiments.py        # trains every experiment (GPU)
uv run python 3_analysis_and_outputs.py   # analyses, figures and tables
```

Finished subjects and runs are skipped, so each step can be stopped and restarted. Single scripts
run from the repository root (e.g. `uv run python analysis/multi_arch.py`); the runner files list
them in order. The Rønne subjects are rebuilt from the stored parameters in
`datasets/roenne_subject_params.csv`.

## Repository

| Folder | Contents |
|---|---|
| `src/` | data loading and preprocessing, synthetic-data generators (MATLAB helpers in `src/matlab/`), augmentation, models, training |
| `experiments/` | one training grid per experiment, 30 Monte Carlo runs per setting (10 for the CWT-ViT augmentation tuning) |
| `analysis/`, `figures/`, `tables/` | statistics, figures and supplementary tables |
| `datasets/` | labels and Rønne subject parameters (included), the downloaded recordings, the synthetic data written by step 1 |
| `results/` | written by steps 2 and 3, one folder per experiment or analysis (about 30 GB, mostly checkpoints); only the selected augmentation settings (`selected.json`) are included |
| `amt_patch/` | the two patched AMT files |

An experiment folder holds `runs/*.json` (settings, validation loss, augmentation), `predictions/*.csv.gz`
(every evaluation waveform) and `checkpoints/*.pt`. Analyses read the predictions and never retrain;
attribution, the impairment sweeps and the parameter counts also load the checkpoints. Each analysis
writes a `summary.txt` with the numbers reported in the paper. Statistics use a participant-level
bootstrap (`analysis/bootstrap.py`).

The results were produced on Windows 11 with Python 3.12.13, PyTorch 2.11.0+cu128, an NVIDIA
GeForce RTX 4060 and MATLAB R2025a.

## Names in the code

| Code | Paper |
|---|---|
| `wang`, `southampton`, `physionet` | Mendeley, Southampton, PhysioNet |
| `roenne`, `verhulst`, `mixup`, `cutmix` | Rønne, Verhulst, MixUp, CutMix |
| `gaussian`, `gaussian_latency`, `simple_gaussian` | matched, latency-matched, unmatched Gaussian control |
| `deep4`, `shallow` | DeepConvNet, ShallowConvNet |
| `cut20` | synthetic label rule: wave V present above 20 dB peSPL |
| `near` | mixed-label levels |
| `southampton_augdev` | the four held-out Southampton participants |

The positive class is wave V **absent**. CWT-ViT (Liu et al., 2026) is reimplemented in
`src/cwt_vit.py`, where its deviations and assumptions are marked.

## License

MIT (`LICENSE`), except `amt_patch/`, whose two changed AMT files keep their licenses: GNU GPL v3
(`zilany2007.m`) and the UGent Academic License, non-commercial academic research only
(`run_cochlear_model.py`); texts in `amt_patch/licenses/`. The recordings and labels keep the terms
of their sources.
