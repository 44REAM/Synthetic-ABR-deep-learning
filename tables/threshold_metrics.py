"""Threshold-metric tables from analysis/threshold_metrics.py, as LaTeX longtables.

Two tables with the same rows (test set, then PhysioNet by tone-burst frequency, then
architecture, then training source; real-data models are not tested on their own dataset)
and columns (AUROC and the threshold metrics): one at the fixed threshold of 0.5, and one with
the held-out Youden threshold of each run, which also has the conventional detectors. Each metric is
the mean over runs with its 95% participant-bootstrap CI, read from
results/threshold_metrics/threshold_metrics.csv (positive class = wave V absent).
Writes threshold_fixed.tex and threshold_heldout.tex into results/threshold_metrics/tables/.

Run from the repository root:
  python tables/threshold_metrics.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from analysis.threshold_metrics import ARCHS, FREQUENCY_SETS, SOURCES, TEST_SETS

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RESULTS_DIR = REPOSITORY_ROOT / 'results' / 'threshold_metrics'
OUT_DIR = RESULTS_DIR / 'tables'
EVAL_SETS = {**TEST_SETS, **FREQUENCY_SETS}           # table groups, in this order
NAMES = {**SOURCES, 'roenne': 'R{\\o}nne', 'roenne+verhulst': 'R{\\o}nne+Verhulst',
         'gaussian': 'Matched Gaussian'}       # the control name of Table 1
ARCH_NAMES = {**ARCHS, 'template': 'Template cross-correlation (no training)',
              'adjacent': 'Adjacent-level correlation (no training)',
              'template_adjacent': 'Template + adjacent-level correlation (no training)'}
METRICS = {'sens': 'Sensitivity', 'spec': 'Specificity', 'f1': 'F1', 'acc': 'Accuracy'}
CAPTIONS = {
    'fixed': (
        'Threshold-dependent performance at a fixed threshold of 0.5 (8 development subjects for the real '
        'datasets and 64 for the synthetic sources and the matched Gaussian control, each with its selected '
        'augmentation). Wave~V absent is the positive class. Values are means over 30 runs with 95\\% '
        'participant-level bootstrap confidence intervals.'),
    'heldout': (
        'Threshold-dependent performance with a threshold selected separately for each run by the Youden index on '
        'the four held-out Southampton participants and applied to that run\'s test predictions (8 development '
        'subjects for the real datasets and 64 for the synthetic sources and the matched Gaussian control, each '
        'with its selected augmentation). Wave~V absent is the positive class. Values are means over 30 runs with '
        '95\\% participant-level bootstrap confidence intervals.'),
}


def cell(value: float, low: float, high: float) -> str:
    return f'{value:.2f} [{low:.2f}, {high:.2f}]'


def longtable(name: str, columns: list[str], rows: dict[tuple[str, str, str], list[str]]) -> str:
    n = len(columns) + 1
    header = ('\\textbf{Training source} & ' + ' & '.join(f'\\textbf{{{c}}}' for c in columns) + ' \\\\\n'
              '\\midrule\n')
    lines = ['{', '\\setlength{\\tabcolsep}{4pt}', '\\setlength{\\LTcapwidth}{\\textwidth}', '\\footnotesize',
             f'\\begin{{longtable}}{{l{"c" * len(columns)}}}',
             f'\\caption{{{CAPTIONS[name]}}} \\label{{tab:threshold_{name}}} \\\\',
             '\\toprule', header + '\\endfirsthead',
             f'\\multicolumn{{{n}}}{{c}}{{\\small\\textbf{{Supplementary Table \\thetable{{}}}} '
             f'\\textit{{(Continued)}}}} \\\\',
             '\\toprule', header + '\\endhead',
             '\\midrule', f'\\multicolumn{{{n}}}{{r}}{{\\small\\itshape Continued on next page --}} \\\\',
             '\\endfoot', '\\bottomrule', '\\endlastfoot']
    for k, (test_set, test_name) in enumerate(EVAL_SETS.items()):
        if k:
            lines.append('\\midrule')
        lines.append(f'\\multicolumn{{{n}}}{{l}}{{\\textbf{{Test set: {test_name}}}}} \\\\')
        for arch, arch_name in ARCH_NAMES.items():
            body = [f'{NAMES[s]} & ' + ' & '.join(rows[test_set, arch, s]) + ' \\\\'
                    for s in SOURCES if (test_set, arch, s) in rows]
            if body:                                  # the template is in the held-out table only
                lines += ['\\addlinespace[0.3em]', f'\\multicolumn{{{n}}}{{l}}{{\\textit{{{arch_name}}}}} \\\\', *body]
    lines += ['\\end{longtable}', '}']
    return '\n'.join(lines) + '\n'


def main() -> None:
    table = pd.read_csv(RESULTS_DIR / 'threshold_metrics.csv')
    table = table[table['subset'] == 'all']
    cells = {(r.threshold_rule, r.eval_set, r.arch, r.sources, r.metric): cell(r.value, r.ci_low, r.ci_high)
             for r in table.itertuples()}

    def rows(rule: str) -> dict[tuple[str, str, str], list[str]]:
        keys = sorted({(e, a, s) for r, e, a, s, _ in cells if r == rule and e in EVAL_SETS})
        return {k: [cells['none', *k, 'auroc']] + [cells[rule, *k, m] for m in METRICS] for k in keys}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rule in (('fixed', '0.5'), ('heldout', 'youden_heldout')):
        path = OUT_DIR / f'threshold_{name}.tex'
        path.write_text(longtable(name, ['AUROC', *METRICS.values()], rows(rule)), encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
