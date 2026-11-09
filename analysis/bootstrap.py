"""Participant-level bootstrap of AUROC averaged over training runs.

Test participants are drawn with replacement and every waveform of a drawn participant is
kept, so the dependence between a participant's waveforms is preserved. The statistic is the
AUROC of each run on the resampled waveforms, averaged over the runs of a setting (the mean the
tables report); the CI is the 2.5-97.5 percentile range over draws. Draws with only one class
are redrawn. Within an evaluation set every setting is scored on the same draws (and the same
seed gives the same draws), so differences between settings are paired draw by draw
(paired_difference); benjamini_hochberg adjusts a family of their p-values.

bootstrap_threshold_metrics does the same for accuracy, F1, sensitivity and specificity at a fixed
probability threshold or one threshold per run (positive class = label 1, wave V absent), on
the same draws as bootstrap_auroc for the same waveforms, subsets and seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_BOOT = 10000
SEED = 20261024


def auroc_rows(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC of each row of scores (runs x waveforms), Mann-Whitney statistic with ties averaged."""
    ranks = rankdata(scores, axis=1)
    n_pos = labels.sum()
    n_neg = labels.size - n_pos
    return (ranks[:, labels == 1].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def participant_draws(subjects: np.ndarray, labels: np.ndarray, n_boot: int,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """Waveform indices of n_boot participant resamples, each containing both classes."""
    ids = np.unique(subjects)
    rows_of = {s: np.flatnonzero(subjects == s) for s in ids}
    draws = []
    while len(draws) < n_boot:
        idx = np.concatenate([rows_of[s] for s in rng.choice(ids, len(ids))])
        if 0 < labels[idx].sum() < idx.size:
            draws.append(idx)
    return draws


def bootstrap_auroc(runs: pd.DataFrame, preds: pd.DataFrame, waveforms: pd.DataFrame, group_cols: list[str],
                    subsets: dict[str, set[int] | None], n_boot: int = N_BOOT, seed: int = SEED,
                    return_draws: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Mean AUROC over the runs of each setting with its participant-bootstrap CI.

    runs: 'run' plus the group_cols that define a setting; preds: run, waveform_id, prob_absent;
    waveforms: the evaluation waveforms to use (waveform_id, eval_set, subject, label_absent).
    subsets: name -> waveform_ids to restrict to (None = all), e.g. {'all': None, 'near': ids}.
    Returns one row per (setting, evaluation set, subset). Runs without predictions for an
    evaluation set (e.g. real-data runs on their own dataset) are left out of that set.
    return_draws=True also returns {(setting key tuple, eval_set, subset): bootstrap means}, for
    paired_difference.
    """
    rng = np.random.default_rng(seed)
    table = preds.pivot(index='run', columns='waveform_id', values='prob_absent')
    rows, draws_out = [], {}
    for eval_set, wf_set in waveforms.groupby('eval_set', sort=False):
        tested = runs[runs['run'].isin(preds.loc[preds['waveform_id'].isin(wf_set['waveform_id']), 'run'])]
        for subset, ids in subsets.items():
            wf = (wf_set if ids is None else wf_set[wf_set['waveform_id'].isin(ids)]).sort_values('waveform_id')
            labels = wf['label_absent'].to_numpy()
            subjects = wf['subject'].to_numpy()
            draws = participant_draws(subjects, labels, n_boot, rng)
            for key, grp in tested.groupby(group_cols, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                scores = table.loc[grp['run'], wf['waveform_id']].to_numpy()
                boot = np.array([auroc_rows(scores[:, idx], labels[idx]).mean() for idx in draws])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                draws_out[(key, eval_set, subset)] = boot
                rows.append({**dict(zip(group_cols, key)),
                             'eval_set': eval_set, 'subset': subset, 'n_runs': len(grp),
                             'n_waveforms': len(wf), 'n_participants': len(np.unique(subjects)),
                             'auroc': auroc_rows(scores, labels).mean(), 'ci_low': lo, 'ci_high': hi})
    return (pd.DataFrame(rows), draws_out) if return_draws else pd.DataFrame(rows)


def threshold_rows(scores: np.ndarray, labels: np.ndarray, threshold: float | np.ndarray) -> dict[str, np.ndarray]:
    """Accuracy, F1, sensitivity and specificity of each row of scores (runs x waveforms) at the
    threshold (one value, or one per row as a column vector)."""
    pred, pos = scores >= threshold, labels == 1
    tp, fp = (pred & pos).sum(axis=1), (pred & ~pos).sum(axis=1)
    fn, tn = (~pred & pos).sum(axis=1), (~pred & ~pos).sum(axis=1)
    return {'acc': (tp + tn) / labels.size, 'f1': 2 * tp / (2 * tp + fp + fn), 'sens': tp / (tp + fn),
            'spec': tn / (tn + fp)}


def bootstrap_threshold_metrics(runs: pd.DataFrame, preds: pd.DataFrame, waveforms: pd.DataFrame,
                                group_cols: list[str], subsets: dict[str, set[int] | None], threshold: float = 0.5,
                                run_thresholds: dict[str, float] | None = None, n_boot: int = N_BOOT,
                                seed: int = SEED) -> pd.DataFrame:
    """Mean accuracy, F1, sensitivity and specificity over the runs of each setting with participant-bootstrap CIs.

    Arguments as bootstrap_auroc; run_thresholds (run -> threshold), if given, replaces the one
    threshold for every run. Returns one row per (setting, evaluation set, subset, metric) with
    the mean over runs ('value'), its 95% CI and the SD over runs.
    """
    rng = np.random.default_rng(seed)
    table = preds.pivot(index='run', columns='waveform_id', values='prob_absent')
    rows = []
    for eval_set, wf_set in waveforms.groupby('eval_set', sort=False):
        tested = runs[runs['run'].isin(preds.loc[preds['waveform_id'].isin(wf_set['waveform_id']), 'run'])]
        for subset, ids in subsets.items():
            wf = (wf_set if ids is None else wf_set[wf_set['waveform_id'].isin(ids)]).sort_values('waveform_id')
            labels = wf['label_absent'].to_numpy()
            subjects = wf['subject'].to_numpy()
            draws = participant_draws(subjects, labels, n_boot, rng)
            for key, grp in tested.groupby(group_cols, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                scores = table.loc[grp['run'], wf['waveform_id']].to_numpy()
                thr = (threshold if run_thresholds is None
                       else np.array([run_thresholds[r] for r in grp['run']])[:, None])
                point = threshold_rows(scores, labels, thr)
                boot = [threshold_rows(scores[:, idx], labels[idx], thr) for idx in draws]
                for metric, values in point.items():
                    lo, hi = np.percentile([b[metric].mean() for b in boot], [2.5, 97.5])
                    rows.append({**dict(zip(group_cols, key)), 'eval_set': eval_set, 'subset': subset,
                                 'metric': metric, 'n_runs': len(grp), 'n_waveforms': len(wf),
                                 'n_participants': len(np.unique(subjects)), 'value': values.mean(),
                                 'ci_low': lo, 'ci_high': hi, 'sd': values.std(ddof=1)})
    return pd.DataFrame(rows)


def paired_difference(point_a: float, point_b: float, boot_a: np.ndarray, boot_b: np.ndarray) -> dict:
    """Mean AUROC of setting a minus setting b, both scored on the same participant draws.

    Returns the difference, its 95% percentile CI and a two-sided bootstrap p-value
    (2 x the smaller tail share of the draw-wise differences, with the +1 correction).
    """
    diff = boot_a - boot_b
    lo, hi = np.percentile(diff, [2.5, 97.5])
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    return {'diff': point_a - point_b, 'ci_low': lo, 'ci_high': hi,
            'p': min(1.0, 2 * (tail + 1) / (len(diff) + 1))}


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q) for one family of comparisons."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return q
