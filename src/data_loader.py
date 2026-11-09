from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal


Record = dict[str, Any]
LabelMap = dict[Any, int]

# Columns per sub-recording: Average, Average(uV), Buffer1, Buffer1(uV), Buffer2, Buffer2(uV)
_N_COLS_PER_REC = 6


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_wang_txt(path: str) -> list[Record]:
    """Parse one Wang-format SmartEP TXT file into one record holding its sub-recordings.

    The sub-recordings of a file share the stimulus level but each has its own sweep count,
    sampling period and zero position (one file, P6 55 dB, mixes 25 and 50 us sampling), so
    they are kept apart in 'parts' and combined after preprocessing (_preprocess_wang_record).
    """
    # Rows 0-19 are key/value metadata; row 20 is the column header; rows 21+ are data.
    meta_df = pd.read_csv(path, header=None, nrows=20, dtype=str,
                          keep_default_na=False, encoding='latin-1')
    data_df = pd.read_csv(path, header=20, dtype=str,
                          keep_default_na=False, encoding='latin-1')

    n_recs = (len(meta_df.columns) - 1) // _N_COLS_PER_REC

    meta: dict[str, list[str]] = {}
    for _, row in meta_df.iterrows():
        key = str(row.iloc[0]).rstrip(':').strip()
        meta[key] = [
            str(row.iloc[1 + i * _N_COLS_PER_REC]).strip()
            if (1 + i * _N_COLS_PER_REC) < len(row) else ''
            for i in range(n_recs)
        ]

    def _meta(key: str, index: int) -> str | None:
        value = (meta.get(key) or [''])[index] if index < n_recs else ''
        return value or None

    data = data_df.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

    subject = os.path.basename(os.path.dirname(path)).lower()

    # Per-sub-recording Average(uV) waveform, sampling rate, zero position and sweep count.
    parts: list[dict[str, Any]] = []
    for index in range(n_recs):
        base = 1 + index * _N_COLS_PER_REC
        sample_period_us = _to_float(_meta('Smp. Period', index))
        parts.append({
            'waveform': data.iloc[:, base + 1].to_numpy(dtype=np.float64),
            'sample_rate': 1e6 / sample_period_us if sample_period_us else 20000.0,
            'zero_pos': _to_int(_meta('Zero Position', index)) or 530,
            'sweeps': _to_int(_meta('Sweeps', index)) or 0,
        })

    return [{
        'path': path,
        'subject': subject,
        'file_name': _meta('File Name', 0) or os.path.basename(path),
        'intensity': _to_float(_meta('Intensity', 0)),
        'sweeps': sum(part['sweeps'] for part in parts),
        'parts': parts,
        'header': {'source': 'wang'},
    }]


def load_wang_dataset(root_dir: str) -> list[Record]:
    """Recursively load all Wang TXT files under ``root_dir``."""
    records: list[Record] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if filename.lower().endswith('.txt'):
                records.extend(parse_wang_txt(os.path.join(dirpath, filename)))
    return records


def load_southampton_mat(mat_path: str) -> list[Record]:
    """Load the Southampton ABR MAT file into one record per subject and intensity."""
    mat_data = scipy.io.loadmat(mat_path)
    records: list[Record] = []

    for key, value in mat_data.items():
        if key.startswith('__'):
            continue

        match = re.match(r'^DB(\d+)$', key)
        if match is None:
            continue

        intensity = float(match.group(1))
        arr = np.array(value)
        if arr.ndim != 2:
            continue

        for subj_idx in range(arr.shape[1]):
            try:
                sweeps = np.array(arr[0, subj_idx], dtype=np.float64)
                if sweeps.ndim != 2:
                    continue

                avg = sweeps.mean(axis=0).astype(np.float32)
                records.append({
                    'path': mat_path,
                    'subject': f'soton_s{subj_idx:02d}',
                    'intensity': intensity,
                    'sweeps': sweeps.shape[0],
                    'sample_period': None,
                    'zero_pos': 0,
                    'waveform': avg,
                    'header': {'mat_key': key, 'source': 'southampton'},
                })
            except Exception:
                continue

    return records


def resample_to_rate(waveform: np.ndarray, orig_rate: float, target_rate: float) -> np.ndarray:
    """Resample a 1-D waveform to ``target_rate`` using FFT-based resampling."""
    if orig_rate == target_rate:
        return waveform
    duration = len(waveform) / orig_rate
    target_samples = int(np.round(duration * target_rate))
    return scipy.signal.resample(waveform, target_samples).astype(np.float32)


def pad_or_truncate(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if len(waveform) == target_len:
        return waveform
    if len(waveform) > target_len:
        return waveform[:target_len]
    return np.pad(waveform, (0, target_len - len(waveform)), mode='constant')


def load_physionet_abr(npz_path: str) -> list[Record]:
    """Load the PhysioNet ABR records (F1=1 kHz and F4=4 kHz) written by src/prepare_physionet.py.

    Each record is the mean of the two half averages (R1, R2) in uV, at 48 kHz from the
    stimulus onset. Records whose ABR channel was removed (only OAE remains) are not in the file.
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'{npz_path} not found: run src/prepare_physionet.py first')

    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in ('record', 'subject', 'subject_idx', 'level', 'freq_khz',
                                             'sample_rate', 'abr_uv')}

    records: list[Record] = []
    for k, name in enumerate(arrays['record']):
        intensity = float(arrays['level'][k])
        freq_khz = int(arrays['freq_khz'][k])
        records.append({
            'path': f'{npz_path}::{name}',
            'subject': str(arrays['subject'][k]),
            'subject_idx': int(arrays['subject_idx'][k]),
            'intensity': intensity,
            'freq_khz': freq_khz,
            'sweeps': None,
            'sample_period': None,
            'sample_rate': float(arrays['sample_rate'][k]),
            'zero_pos': 0,
            'waveform': arrays['abr_uv'][k].astype(np.float32),
            'header': {
                'source': 'physionet',
                'level': intensity,
                'freq_khz': freq_khz,
            },
        })

    return records


def load_southampton_labels(csv_path: str) -> LabelMap:
    """Load Southampton labels and return majority-vote binary labels."""
    df = pd.read_csv(csv_path)
    subj_cols = [
        column for column in df.columns if re.match(r'^S\d+$', column)]

    label_map: LabelMap = {}
    for intensity, group in df.groupby('StimulusLevel(dB SL)'):
        for column in subj_cols:
            subj_idx = int(column[1:]) - 1
            votes = group[column].dropna().astype(int).values
            if len(votes) == 0:
                continue
            majority = int(round(votes.mean()))
            label_map[(subj_idx, float(intensity))] = majority

    return label_map


def load_physionet_labels(csv_path: str) -> LabelMap:
    """Load PhysioNet labels and return {(subj_idx, freq_khz, level): binary_label}.

    Subject columns S1-S8 map to subjects N1-N8. Entries marked '-' are skipped.
    """
    df = pd.read_csv(csv_path)
    subj_cols = [c for c in df.columns if re.match(r'^S\d+$', c)]

    label_map: LabelMap = {}
    for _, row in df.iterrows():
        freq_match = re.match(
            r'^(\d+)\s*kHz$', str(row['Stim Frequency']).strip(), re.IGNORECASE)
        if freq_match is None:
            continue
        freq_khz = int(freq_match.group(1))

        level = _to_float(row['Stimulus Level (dB SL)'])
        if level is None:
            continue

        for col in subj_cols:
            subj_idx = int(col[1:])
            val = str(row[col]).strip()
            if val in ('-', '', 'nan'):
                continue
            label = _to_int(val)
            if label is not None:
                label_map[(subj_idx, freq_khz, level)] = label

    return label_map


def load_wang_labels(csv_path: str) -> LabelMap:
    """Load Wang labels and return majority-vote binary labels."""
    df = pd.read_csv(csv_path)
    subj_cols = [
        column for column in df.columns if re.match(r'^S\d+$', column)]

    label_map: LabelMap = {}
    for intensity, group in df.groupby('StimulusLevel(dB SL)'):
        for column in subj_cols:
            subj_idx = int(column[1:]) - 1
            subj_name = f'p{subj_idx + 1}'
            values = pd.to_numeric(
                group[column], errors='coerce').dropna().values
            if len(values) == 0:
                continue
            majority = int(round(values.mean()))
            label_map[(subj_name, float(intensity))] = majority

    return label_map


def _match_physionet_label(record: Record, label_map: LabelMap) -> int | None:
    """Match a PhysioNet record to its binary label."""
    subj_idx = record.get('subject_idx')
    freq_khz = record.get('freq_khz')
    intensity = record.get('intensity')
    if subj_idx is None or freq_khz is None or intensity is None:
        return None
    return label_map.get((subj_idx, freq_khz, float(intensity)))


def _match_soton_label(record: Record, label_map: LabelMap) -> int | None:
    """Match a Southampton record to its majority-vote label."""
    subject = record.get('subject', '')
    match = re.match(r'soton_s(\d+)', subject)
    if match is None:
        return None

    intensity = record.get('intensity')
    if intensity is None:
        return None

    subj_idx = int(match.group(1))
    return label_map.get((subj_idx, float(intensity)))


def _match_wang_label(record: Record, label_map: LabelMap) -> int | None:
    """Match a Wang record to its majority-vote label."""
    path = record.get('path', '')
    match = re.search(r'[/\\](P\d+)[/\\]', path, re.IGNORECASE)
    if match is None:
        return None

    intensity = record.get('intensity')
    if intensity is None:
        return None

    return label_map.get((match.group(1).lower(), float(intensity)))


def _load_label_maps(
    soton_label_csv: str | None,
    wang_label_csv: str | None,
    physionet_label_csv: str | None = None,
) -> tuple[LabelMap, LabelMap, LabelMap]:
    soton_labels: LabelMap = {}
    if soton_label_csv and os.path.exists(soton_label_csv):
        soton_labels = load_southampton_labels(soton_label_csv)

    wang_labels: LabelMap = {}
    if wang_label_csv and os.path.exists(wang_label_csv):
        wang_labels = load_wang_labels(wang_label_csv)

    physionet_labels: LabelMap = {}
    if physionet_label_csv and os.path.exists(physionet_label_csv):
        physionet_labels = load_physionet_labels(physionet_label_csv)

    return soton_labels, wang_labels, physionet_labels


def _abr_bandpass_at_rate(waveform: np.ndarray, sample_rate: float) -> np.ndarray:
    """Apply the clinical 30–3 kHz ABR bandpass at the signal's native sample rate.

    The high-cut is clamped to 90 % of the Nyquist frequency so the filter
    remains well-conditioned even for low-rate sources (e.g. Southampton at
    5 kHz whose Nyquist is 2.5 kHz, yielding a 30–2250 Hz passband).
    """
    nyq = sample_rate / 2.0
    low = 30.0
    high = min(3000.0, nyq * 0.9)
    if low >= high:
        return waveform  # source rate too low to bandpass meaningfully
    sos = scipy.signal.butter(
        4, [low / nyq, high / nyq], btype='band', output='sos')
    return scipy.signal.sosfiltfilt(sos, waveform).astype(np.float32)


def preprocess_synthetic_waveform(
    waveform: np.ndarray,
    native_rate: float,
    target_rate: float,
) -> np.ndarray:
    """Bandpass then resample a synthetic waveform to the target rate.

    Mirrors the real-data pipeline: apply the clinical 30–3 kHz bandpass at
    the generator's native rate, then resample to ``target_rate``.  For
    generators already at ``target_rate`` the resample is a no-op.
    """
    waveform = _abr_bandpass_at_rate(waveform, native_rate)
    return resample_to_rate(waveform, orig_rate=native_rate, target_rate=target_rate)


def _preprocess_wang_record(record: Record, target_len: int, target_rate: float) -> np.ndarray | None:
    """Preprocess the sub-recordings of a Wang file one by one, then average them.

    Each sub-recording is cropped at its own stimulus onset (keeping the whole post-stimulus
    sweep), band-passed at its own rate, resampled and cut to target_len, like the Southampton
    and PhysioNet sweeps. The mean weighted by sweep count (equal weights if no counts) is taken
    per sample over the sub-recordings that reach it: 25 us sweeps end 11.9 ms after the
    stimulus, and samples no sub-recording reaches stay 0.
    """
    parts = record.get('parts') or []
    use_sweeps = any(part['sweeps'] > 0 for part in parts)
    waveforms, weights = [], []
    for part in parts:
        post = part['waveform'][part['zero_pos']:]
        if len(post) == 0:
            continue
        post = _abr_bandpass_at_rate(post, part['sample_rate'])
        post = resample_to_rate(post, orig_rate=part['sample_rate'], target_rate=target_rate)
        waveforms.append(pad_or_truncate(post, target_len))
        weight = part['sweeps'] if use_sweeps else 1.0
        weights.append(np.where(np.arange(target_len) < len(post), weight, 0.0))
    if not waveforms:
        return None

    weights = np.array(weights, dtype=np.float64)
    total = weights.sum(axis=0)
    combined = (np.array(waveforms, dtype=np.float64) * weights).sum(axis=0) / np.maximum(total, 1e-12)
    return np.where(total > 0, combined, 0.0).astype(np.float32)


def _preprocess_record(record: Record, target_len: int, target_rate: float) -> np.ndarray | None:
    source = record.get('header', {}).get('source')
    if source == 'wang':
        return _preprocess_wang_record(record, target_len, target_rate)

    waveform = record.get('waveform')
    if waveform is None or len(waveform) == 0:
        return None

    if source == 'southampton':
        orig_rate = 5000.0
        waveform = _abr_bandpass_at_rate(waveform, orig_rate)
        waveform_proc = resample_to_rate(
            waveform, orig_rate=orig_rate, target_rate=target_rate)
        return pad_or_truncate(waveform_proc, target_len)

    if source == 'physionet':
        orig_rate = record.get('sample_rate', 48000.0)
        waveform = _abr_bandpass_at_rate(waveform, orig_rate)
        waveform_proc = resample_to_rate(
            waveform, orig_rate=orig_rate, target_rate=target_rate)
        return pad_or_truncate(waveform_proc, target_len)

    return None


def _attach_binary_label(
    record: Record,
    soton_labels: LabelMap,
    wang_labels: LabelMap,
    physionet_labels: LabelMap | None = None,
) -> None:
    source = record.get('header', {}).get('source')

    # Label convention: 1 = wave V absent (positive/pathological class),
    #                    0 = wave V present (negative/normal class).
    if source == 'southampton' and soton_labels:
        label = _match_soton_label(record, soton_labels)
        if label is not None:
            record['label'] = 1 - label
    elif source == 'physionet' and physionet_labels:
        label = _match_physionet_label(record, physionet_labels)
        if label is not None:
            record['label'] = 1 - label
    elif source == 'wang' and wang_labels:
        label = _match_wang_label(record, wang_labels)
        if label is not None:
            record['label'] = 1 - label


def prepare_records_for_dataset(
    wang_root: str,
    mat_path: str | None = None,
    target_len: int = 300,
    target_rate: float = 20_000.0,
    soton_label_csv: str | None = None,
    wang_label_csv: str | None = None,
    physionet_path: str | None = None,
    physionet_label_csv: str | None = None,
) -> list[Record]:
    """Load, preprocess, and label ABR records from the real datasets."""
    soton_labels, wang_labels, physionet_labels = _load_label_maps(
        soton_label_csv, wang_label_csv, physionet_label_csv
    )

    raw_records: list[Record] = []
    raw_records.extend(load_wang_dataset(wang_root))
    if mat_path is not None and os.path.exists(mat_path):
        raw_records.extend(load_southampton_mat(mat_path))
    if physionet_path is not None:
        raw_records.extend(load_physionet_abr(physionet_path))

    prepared_records: list[Record] = []
    for record in raw_records:
        waveform_proc = _preprocess_record(
            record, target_len=target_len, target_rate=target_rate)
        if waveform_proc is None:
            continue

        prepared_record = record.copy()
        prepared_record['waveform_proc'] = waveform_proc
        _attach_binary_label(prepared_record, soton_labels,
                             wang_labels, physionet_labels)

        if 'label' not in prepared_record:
            source = prepared_record.get('header', {}).get('source')
            if source == 'physionet':
                # Physionet CSV uses '-' for unannotated combos; skip silently
                continue
            subject = prepared_record.get('subject', 'Unknown')
            intensity = prepared_record.get('intensity', 'Unknown')
            path = prepared_record.get('path', 'Unknown')
            raise ValueError(
                f"Missing binary label for record - Subject: {subject}, "
                f"Intensity: {intensity}dB (Path: {path})"
            )

        prepared_records.append(prepared_record)

    return prepared_records


def _normalize_signal_in_place(record: Record, norm_method: str, stats: dict[str, Any]) -> None:
    waveform = record['waveform_proc'].astype(np.float32)

    if norm_method == 'mean_std':
        mean = np.float32(stats['mean'])
        std = np.float32(stats['std'])
        record['waveform_proc'] = (waveform - mean) / \
            (std + 1e-8) if std > 0 else waveform - mean
        return

    if norm_method == 'min_max':
        min_val = np.float32(stats['min'])
        max_val = np.float32(stats['max'])
        record['waveform_proc'] = (
            (waveform - min_val) / (max_val - min_val)
            if max_val > min_val else waveform - min_val
        )


def normalize_records(
    records: list[Record],
    norm_method: str = 'mean_std',
    norm_group: str = 'subject',
    global_stats: dict[str, Any] | None = None,
) -> list[Record]:
    """Apply normalization in-place and return the same record list.

    norm_group: 'signal' (each waveform on its own), 'subject' (statistics over all
    waveforms of a subject), 'subject_frequency' (as 'subject', but separately per stimulus
    frequency where a record has 'freq_khz', i.e. PhysioNet's 1 and 4 kHz tone bursts;
    records without 'freq_khz' are grouped as with 'subject'), 'group' (all records).
    """
    if norm_group == 'signal':
        for record in records:
            waveform = record['waveform_proc'].astype(np.float32)
            stats = {
                'mean': np.mean(waveform),
                'std': np.std(waveform),
                'min': np.min(waveform),
                'max': np.max(waveform),
            }
            _normalize_signal_in_place(record, norm_method, stats)

    elif norm_group in ('subject', 'subject_frequency'):
        def group_key(record: Record) -> Any:
            subject = record.get('subject')
            if not subject:
                raise ValueError(
                    f"Missing 'subject' field for record from path: {record.get('path', 'unknown')}"
                )
            if norm_group == 'subject_frequency':
                return (subject, record.get('freq_khz'))
            return subject

        group_arrays: defaultdict[Any, list[np.ndarray]] = defaultdict(list)
        for record in records:
            group_arrays[group_key(record)].append(record['waveform_proc'])

        group_stats: dict[Any, dict[str, Any]] = {}
        for key, arrays in group_arrays.items():
            concatenated = np.concatenate(arrays)
            group_stats[key] = {
                'mean': np.mean(concatenated),
                'std': np.std(concatenated),
                'min': np.min(concatenated),
                'max': np.max(concatenated),
            }

        for record in records:
            _normalize_signal_in_place(
                record, norm_method, group_stats[group_key(record)])

    elif norm_group == 'group':
        if global_stats is not None and len(global_stats) > 0:
            stats = global_stats
        else:
            all_arrays = [record['waveform_proc'] for record in records]
            concatenated = np.concatenate(
                all_arrays) if all_arrays else np.array([])
            stats = {
                'mean': np.mean(concatenated) if len(concatenated) > 0 else 0.0,
                'std': np.std(concatenated) if len(concatenated) > 0 else 1.0,
                'min': np.min(concatenated) if len(concatenated) > 0 else 0.0,
                'max': np.max(concatenated) if len(concatenated) > 0 else 1.0,
            }
            if global_stats is not None:
                global_stats.update(stats)

        for record in records:
            _normalize_signal_in_place(record, norm_method, stats)

    else:
        raise ValueError(f"Unknown norm_group: {norm_group}")

    return records


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wang', default='datasets/wang_single/human_raw_data', help='Wang dataset root')
    parser.add_argument(
        '--mat', default='datasets/Subject_ABR_Data.mat', help='Southampton .mat path')
    parser.add_argument('--out', default=None, help='dump sample JSON')
    args = parser.parse_args()

    records = prepare_records_for_dataset(args.wang, args.mat)
    print(f'Loaded {len(records)} records')
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(
                [{'path': record['path'], 'subject': record['subject'],
                    'intensity': record.get('intensity')} for record in records],
                fh,
                indent=2,
            )
