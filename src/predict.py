"""Predict new waveforms with trained runs, without retraining.

For data other than the evaluation sets that run_grid predicts automatically, e.g. the
simulated impairment sweeps, filtered or masked test waveforms, or a new test set.
Records are dicts with 'waveform_proc' (300 samples at 20 kHz after the stimulus), 'subject'
and 'label', as produced by src/data.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import normalise
from src.run import load_model
from src.train import DEVICE, predict_logits


def predict_runs(out_dir: str | Path, record_sets: dict[str, list[dict]], runs: list[str] | None = None,
                 normalise_records: bool = True, device: torch.device = DEVICE) -> pd.DataFrame:
    """Logits of every run (default: all runs in out_dir) for every record set.

    normalise_records=True applies the min-max normalisation used in training (per subject,
    and per 'freq_khz' within a subject where records carry it; src.data.NORM_GROUP);
    pass False for records that are already normalised.
    Returns one row per (run, set, record index).
    """
    out = Path(out_dir)
    if runs is None:
        runs = sorted(p.stem for p in (out / 'runs').glob('*.json'))
    sets = {name: normalise(recs) if normalise_records else recs for name, recs in record_sets.items()}
    frames = []
    for run in runs:
        model = load_model(out, run, device)
        for name, recs in sets.items():
            logits = predict_logits(model, recs, device)
            frames.append(pd.DataFrame({'run': run, 'set': name, 'index': np.arange(len(logits)), 'logit': logits}))
    preds = pd.concat(frames, ignore_index=True)
    preds['prob_absent'] = 1.0 / (1.0 + np.exp(-preds['logit'].astype(np.float64)))
    return preds
