"""Convert the PhysioNet ABR averages (EARNDB, Epstein and Silva) to uV in one small file.

Reads the averaged records in the 'average' folder of the PhysioNet release
(https://physionet.org/content/earndb/1.0.0/, Nx_evoked_ave<level>_F<freq>_R<rep>; its
'ave_bck' subfolder is not read), keeps the ABR channel of the 1 and 4 kHz records and writes
OUT_PATH, which src/data_loader.load_physionet_abr reads. The release folder is the environment
variable PHYSIONET_DIR, or by default datasets/physionet.org/files/earndb/1.0.0 (where
`wget -r -N -c -np https://physionet.org/files/earndb/1.0.0/` run in datasets/ puts it).
Band-pass, resampling, the 15 ms crop and normalisation stay in the loader, as for the other
real datasets.

Units: every averaged record stores its ABR channel scaled to the int32 full scale
(max |d| = 2^31), and the header gain is the inverse of the usual WFDB gain, so
uV = d_signal x adc_gain. The two half averages
(R1, R2) have different gains, so they are converted before their mean is taken.

Output arrays, one row per (subject, level, frequency), in file-name order:
  record                      record name without the half, e.g. N1_evoked_ave100_F1
  subject, subject_idx        'n1', 1
  level, freq_khz             dB peSPL, kHz
  sample_rate                 Hz (48 kHz)
  abr_uv                      mean of the available halves, uV, from the stimulus onset
  abr_r1_uv, abr_r2_uv        the half averages, uV (NaN where a half has no ABR channel)
  trials_r1, trials_r2        'Total Trials Used' of each half (-1 where missing)
"""

from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np

try:
    import wfdb
except ImportError as exc:
    raise ImportError('wfdb is required to read the PhysioNet records: uv add wfdb') from exc

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = Path(os.environ.get('PHYSIONET_DIR',
                                  REPOSITORY_ROOT / 'datasets' / 'physionet.org' / 'files' / 'earndb' / '1.0.0'))
SOURCE_DIR = RELEASE_DIR / 'average'
OUT_PATH = REPOSITORY_ROOT / 'datasets' / 'physionet_uv.npz'
FREQS_KHZ = (1, 4)

_FNAME_RE = re.compile(r'^(N\d+)_evoked_ave(\d+)_F(\d+)_(R\d+)(?:_x)?\.hea$', re.IGNORECASE)
_TRIALS_RE = re.compile(r'<Total Trials Used>:\s*(\d+)')


def read_half(record_path: Path) -> tuple[np.ndarray, float, float, int] | None:
    """ABR channel of one half average in uV, its sampling rate, gain and trial count (None if no ABR)."""
    rec =wfdb.rdrecord(str(record_path), physical=False)
    names = [str(s).upper() for s in rec.sig_name]
    if 'ABR' not in names:
        return None  # channel removed for artefacts; only the OAE remains
    i = names.index('ABR')
    if rec.units[i] != 'nV':
        raise ValueError(f'{record_path}: ABR units {rec.units[i]!r}, expected nV')
    match = _TRIALS_RE.search(' '.join(rec.comments))
    trials = int(match.group(1)) if match else -1
    return rec.d_signal[:, i].astype(np.float64) * rec.adc_gain[i], float(rec.fs), float(rec.adc_gain[i]), trials


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f'{SOURCE_DIR} not found: download the PhysioNet release (README.md) '
                                'or set PHYSIONET_DIR to its folder')
    halves: dict[tuple[str, int, int], dict[str, tuple]] = {}
    skipped = []
    for path in sorted(SOURCE_DIR.iterdir()):
        match = _FNAME_RE.match(path.name)
        if match is None or int(match.group(3)) not in FREQS_KHZ:
            continue
        subject, level, freq, rep = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4).upper()
        half = read_half(path.with_suffix(''))
        if half is None:
            skipped.append(path.name)
            continue
        halves.setdefault((subject, level, freq), {})[rep] = half

    rows = []
    for (subject, level, freq), reps in halves.items():
        rates = {reps[r][1] for r in reps}
        lengths = {len(reps[r][0]) for r in reps}
        if len(rates) > 1 or len(lengths) > 1:
            raise ValueError(f'{subject} {level} dB F{freq}: halves differ in rate {rates} or length {lengths}')
        n = lengths.pop()
        r1, r2 = (reps[r][0] if r in reps else np.full(n, np.nan) for r in ('R1', 'R2'))
        rows.append({
            'record': f'{subject}_evoked_ave{level}_F{freq}', 'subject': subject.lower(),
            'subject_idx': int(subject[1:]), 'level': float(level), 'freq_khz': freq, 'sample_rate': rates.pop(),
            'abr_uv': np.mean([reps[r][0] for r in sorted(reps)], axis=0),
            'abr_r1_uv': r1, 'abr_r2_uv': r2,
            'trials_r1': reps['R1'][3] if 'R1' in reps else -1, 'trials_r2': reps['R2'][3] if 'R2' in reps else -1,
            'n_halves': len(reps), 'gain_ratio': reps['R1'][2] / reps['R2'][2] if len(reps) == 2 else np.nan,
        })
    if len({len(row['abr_uv']) for row in rows}) > 1:
        raise ValueError('records differ in length')

    out = {key: np.array([row[key] for row in rows]) for key in rows[0] if key not in ('n_halves', 'gain_ratio')}
    for key in ('abr_uv', 'abr_r1_uv', 'abr_r2_uv'):
        out[key] = out[key].astype(np.float32)
    np.savez_compressed(OUT_PATH, **out)

    ratio = np.array([row['gain_ratio'] for row in rows])
    print(f'{len(rows)} records from {SOURCE_DIR}; one half only: '
          f'{[row["record"] for row in rows if row["n_halves"] < 2] or "none"}')
    print('records per frequency:', {f: int((out['freq_khz'] == f).sum()) for f in FREQS_KHZ},
          'subjects', ', '.join(sorted({row['subject'] for row in rows}, key=lambda s: int(s[1:]))))
    print(f'half files without an ABR channel (skipped): {skipped if skipped else "none"}')
    print(f'R1/R2 gain ratio: {np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f}; '
          f'samples per record {out["abr_uv"].shape[1]} at {", ".join(f"{r:g}" for r in set(out["sample_rate"]))} Hz')
    print(f'abr_uv peak-to-peak: median {np.median(np.ptp(out["abr_uv"], axis=1)):.3f} uV')
    print(f'wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
