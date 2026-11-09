"""Label counts of the three real test sets by stimulus level (no model output involved).

Per test set (PhysioNet split by stimulus frequency) and stimulus level: the number of
wave-V-present and wave-V-absent waveforms, the number of participants, whether the level
belongs to the 'near' subset of the analyses (levels at which both labels occur within the
test set; analysis.metrics.near_threshold_ids) and, for Mendeley (6 labellers) and
Southampton (3 labellers; 2 at 0 dB), the number of waveforms on which the labellers
disagree. The majority vote of the labellers is the label used everywhere else (ties are
absent); the script checks this against the loaded labels. PhysioNet has a single label per
waveform. Per test set it also counts the participant (x frequency) series without any
present label.
Writes label_counts.csv and summary.txt into results/label_counts/.

Run from the repository root:
  python analysis/label_counts.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from analysis.metrics import near_threshold_ids
from src.data import DATASET_NAMES, SOTON_LABEL_CSV, WANG_LABEL_CSV, load_eval_sets, waveform_table

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUT_DIR = REPOSITORY_ROOT / 'results' / 'label_counts'
TEST_SETS = ['wang', 'southampton', 'physionet']
# Raw labeller files: value 1 = wave V present, one row per labeller and level, one column per subject
LABELLER_FILES = {'wang': (WANG_LABEL_CSV, lambda col: f'p{int(col[1:])}'),
                  'southampton': (SOTON_LABEL_CSV, lambda col: f'soton_s{int(col[1:]) - 1:02d}')}


def labeller_votes(eval_set: str) -> pd.DataFrame:
    """Per (subject, intensity_db): number of labellers and the share voting absent."""
    path, subject_of = LABELLER_FILES[eval_set]
    df = pd.read_csv(path)
    level = df.columns[1]
    long = df.melt(id_vars=['Labeler', level], var_name='column', value_name='present')
    long['present'] = pd.to_numeric(long['present'], errors='coerce')
    long = long.dropna(subset=['present'])
    votes = long.groupby(['column', level])['present'].agg(n_labellers='size', present_share='mean').reset_index()
    return pd.DataFrame({'eval_set': eval_set, 'subject': votes['column'].map(subject_of),
                         'intensity_db': votes[level].astype(float), 'n_labellers': votes['n_labellers'],
                         'absent_share': 1 - votes['present_share']})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf = waveform_table(load_eval_sets())
    wf = wf[wf['role'] == 'test'].copy()
    wf['freq_khz'] = wf['freq_khz'].fillna(0.0)
    wf['near'] = wf['waveform_id'].isin(near_threshold_ids(wf))

    votes = pd.concat([labeller_votes(s) for s in LABELLER_FILES], ignore_index=True)
    wf = wf.merge(votes, on=['eval_set', 'subject', 'intensity_db'], how='left')
    voted = wf.dropna(subset=['absent_share'])
    assert len(voted) == wf['eval_set'].isin(LABELLER_FILES).sum(), 'every Mendeley/Southampton waveform needs its votes'
    majority = (voted['absent_share'] >= 0.5).astype(int)         # ties are absent, as in src/data_loader.py
    assert (majority == voted['label_absent']).all(), 'majority vote of the labellers differs from the loaded label'
    wf['disputed'] = (wf['absent_share'] > 0) & (wf['absent_share'] < 1)

    counts = (wf.groupby(['eval_set', 'freq_khz', 'intensity_db'])
              .agg(n_present=('label_absent', lambda s: int((s == 0).sum())), n_absent=('label_absent', 'sum'),
                   n_participants=('subject', 'nunique'), near=('near', 'first'), n_disputed=('disputed', 'sum'))
              .reset_index())
    counts['n_disputed'] = counts['n_disputed'].astype('Int64')
    counts.loc[~counts['eval_set'].isin(LABELLER_FILES), 'n_disputed'] = pd.NA
    counts.to_csv(OUT_DIR / 'label_counts.csv', index=False)

    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)

    def levels(d: pd.DataFrame) -> str:
        return ', '.join(f'{v:g}' for v in sorted(d['intensity_db'].unique()))

    for eval_set in TEST_SETS:
        d = wf[wf['eval_set'] == eval_set]
        near = d[d['near']]
        labellers = f', labellers per waveform {int(d["n_labellers"].min())}-{int(d["n_labellers"].max())}' \
            if eval_set in LABELLER_FILES else ', single label'
        out(f'=== {DATASET_NAMES[eval_set]}: {len(d)} waveforms, {d["subject"].nunique()} participants, '
            f'{d["label_absent"].sum()} absent{labellers} ===')
        series = d.groupby(['subject', 'freq_khz'])['label_absent']
        out(f'  participant (x frequency) series without any present label: {int((series.min() == 1).sum())} of '
            f'{series.ngroups}')
        out(f'  near subset (levels with both labels): {len(near)} waveforms ({near["label_absent"].sum()} absent), '
            f'{near["subject"].nunique()} participants, levels {levels(near)} dB')
        if eval_set in LABELLER_FILES:
            dis = d[d['disputed']]
            out(f'  waveforms with disagreeing labellers: {len(dis)} ({dis["subject"].nunique()} participants), levels '
                f'{levels(dis)} dB; of these in the near subset: {int(dis["near"].sum())}')
        c = counts[counts['eval_set'] == eval_set].copy()
        c['near'] = c['near'].map({True: '*', False: ''})
        for freq, cf in c.groupby('freq_khz'):
            out(f'  stimulus level (dB){f", {freq:g} kHz" if freq else ""}; * = near subset')
            table = cf.set_index('intensity_db')[['n_present', 'n_absent', 'n_participants', 'near']
                                                 + (['n_disputed'] if eval_set in LABELLER_FILES else [])].T
            table.columns = [f'{v:g}' for v in table.columns]
            out('    ' + table.to_string().replace('\n', '\n    '))
        out()

    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
