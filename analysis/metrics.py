"""Per-run metrics of a results folder written by src.run.run_grid.

One row per (run, evaluation set) with the threshold metrics at 0.5 (accuracy, AUROC,
F1, MCC, sensitivity, specificity; positive class = wave V absent), the unweighted log-loss of
P(absent), and the AUROC near threshold
(levels at which both labels occur in that set). Run-level summaries (mean and SD over runs)
start from this table; the participant-level bootstrap (analysis/bootstrap.py) starts from the
per-waveform predictions and the 'subject' column of waveforms.csv instead.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, matthews_corrcoef, roc_auc_score

RUN_COLUMNS = ['run', 'sources', 'n_dev', 'arch', 'label_rule', 'augmentation', 'cv',
               'val_loss', 'val_auc', 'best_step']


def binary_metrics(prob_absent: np.ndarray, label_absent: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, AUROC, F1, MCC, sensitivity, specificity and log-loss at the threshold."""
    labels = label_absent.astype(int)
    preds = (prob_absent >= threshold).astype(int)
    two_classes = len(np.unique(labels)) > 1
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        'acc': (tp + tn) / len(labels),
        'auc': roc_auc_score(labels, prob_absent) if two_classes else np.nan,
        'f1': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0,
        'mcc': matthews_corrcoef(labels, preds) if two_classes else np.nan,
        'sens': tp / (tp + fn) if tp + fn > 0 else np.nan,
        'spec': tn / (tn + fp) if tn + fp > 0 else np.nan,
        'logloss': log_loss(labels, np.clip(prob_absent, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold maximising Youden's J (sensitivity + specificity - 1; label 1 = positive, predicted
    positive if score >= threshold), and that J. Thresholds tying on J form ranges of scores; the
    middle of the widest range is returned."""
    u = np.unique(scores)
    pos, neg = labels == 1, labels == 0
    j = np.array([(scores[pos] >= t).mean() + (scores[neg] < t).mean() - 1 for t in u])
    best = np.flatnonzero(np.isclose(j, j.max()))
    # Contiguous blocks of best cut indices; threshold u[k] covers the scores in (u[k-1], u[k]]
    blocks = np.split(best, np.flatnonzero(np.diff(best) > 1) + 1)
    spans = [(u[b[0] - 1] if b[0] > 0 else u[0], u[b[-1]]) for b in blocks]
    lo, hi = max(spans, key=lambda s: s[1] - s[0])
    return float((lo + hi) / 2), float(j.max())


def near_threshold_ids(waveforms: pd.DataFrame) -> set[int]:
    """Waveforms at levels where both labels occur within their evaluation set."""
    g = waveforms.groupby(['eval_set', 'intensity_db'])['label_absent']
    mixed = g.transform('min') != g.transform('max')
    return set(waveforms.loc[mixed, 'waveform_id'])


def physionet_by_frequency(waveforms: pd.DataFrame) -> pd.DataFrame:
    """The PhysioNet waveforms with one evaluation set per tone-burst frequency ('1 kHz', '4 kHz')."""
    physionet = waveforms[waveforms['eval_set'] == 'physionet']
    return physionet.assign(eval_set=physionet['freq_khz'].map(lambda f: f'{int(f)} kHz'))


def per_run_metrics(runs: pd.DataFrame, preds: pd.DataFrame, waveforms: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, evaluation set): run settings, metrics on all waveforms, AUROC near threshold."""
    near = near_threshold_ids(waveforms)
    d = preds.merge(waveforms[['waveform_id', 'eval_set', 'role', 'label_absent']], on='waveform_id')
    rows = []
    for (run, eval_set), g in d.groupby(['run', 'eval_set'], sort=False):
        gn = g[g['waveform_id'].isin(near)]
        auc_near = (roc_auc_score(gn['label_absent'], gn['prob_absent'])
                    if gn['label_absent'].nunique() > 1 else np.nan)
        rows.append({'run': run, 'eval_set': eval_set, 'role': g['role'].iloc[0], 'n_waveforms': len(g),
                     **binary_metrics(g['prob_absent'].to_numpy(), g['label_absent'].to_numpy()),
                     'auc_near': auc_near, 'n_near': len(gn)})
    return runs[RUN_COLUMNS].merge(pd.DataFrame(rows), on='run')
