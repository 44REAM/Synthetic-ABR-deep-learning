from __future__ import annotations

import os
import shutil
from typing import Any

import numpy as np
import pandas as pd


def _remove_legacy_dataset_files(out_dir: str) -> None:
    for legacy_file in ('waveforms.npy', 'labels.csv'):
        legacy_path = os.path.join(out_dir, legacy_file)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)


def _subject_dataset_paths(out_dir: str, subject_id: str) -> tuple[str, str, str]:
    subject_dir = os.path.join(out_dir, subject_id)
    waveform_path = os.path.join(subject_dir, 'waveforms.npy')
    labels_path = os.path.join(subject_dir, 'labels.csv')
    return subject_dir, waveform_path, labels_path


def subject_dataset_exists(out_dir: str, subject_id: str) -> bool:
    _, waveform_path, labels_path = _subject_dataset_paths(out_dir, subject_id)
    return os.path.exists(waveform_path) and os.path.exists(labels_path)


def save_subject_dataset(
    out_dir: str,
    subject_id: str,
    subject_waveforms: list[np.ndarray],
    subject_records: list[dict[str, Any]],
    subject_clean_waveforms: list[np.ndarray] | None = None,
    subject_an_populations: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Save a subject's `waveforms.npy` and `labels.csv` in `<out_dir>/<subject_id>/`.

    If given, `subject_clean_waveforms` is also saved as `waveforms_clean.npy` and
    `subject_an_populations` as `an_population.npy`, each of shape `(n_levels, post_len)`.
    """
    _remove_legacy_dataset_files(out_dir)

    subject_dir, waveform_path, labels_path = _subject_dataset_paths(
        out_dir, subject_id)
    if os.path.isdir(subject_dir):
        shutil.rmtree(subject_dir)

    os.makedirs(subject_dir, exist_ok=True)
    waveforms_np = np.stack(subject_waveforms, axis=0).astype(np.float32)
    labels_df = pd.DataFrame(subject_records)
    np.save(waveform_path, waveforms_np)
    labels_df.to_csv(labels_path, index=False)

    if subject_clean_waveforms is not None:
        # Deterministically align each clean waveform to match the saved
        # noisy waveform shape (rows and columns) without using exceptions.
        n_rows_wave, n_cols_wave = waveforms_np.shape

        clean_stack: list[np.ndarray] = []
        for x in subject_clean_waveforms:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            # If clean is shorter than the noisy full length, left-pad with zeros
            if arr.shape[0] < n_cols_wave:
                pre_pad = n_cols_wave - arr.shape[0]
                arr_full = np.concatenate(
                    [np.zeros(pre_pad, dtype=np.float32), arr])
            else:
                # If clean is longer, keep the most recent samples (right-aligned)
                arr_full = arr[-n_cols_wave:].astype(np.float32)
            clean_stack.append(arr_full)

        if not clean_stack:
            clean_arr = np.zeros((n_rows_wave, n_cols_wave), dtype=np.float32)
        else:
            clean_arr = np.stack(clean_stack, axis=0).astype(np.float32)
            # Align number of rows: pad with zero-rows or truncate as needed
            n_rows_clean = clean_arr.shape[0]
            if n_rows_clean < n_rows_wave:
                pad_rows = n_rows_wave - n_rows_clean
                pad_block = np.zeros((pad_rows, n_cols_wave), dtype=np.float32)
                clean_arr = np.vstack([clean_arr, pad_block])
            elif n_rows_clean > n_rows_wave:
                clean_arr = clean_arr[:n_rows_wave, :]

        np.save(os.path.join(subject_dir, 'waveforms_clean.npy'), clean_arr)
        # Also save a copy of the labels so clean waveforms have identical
        # metadata (rows correspond 1:1 to `waveforms_clean.npy`).
        labels_df.to_csv(os.path.join(
            subject_dir, 'labels_clean.csv'), index=False)

    if subject_an_populations is not None:
        n_rows_wave, n_cols_wave = waveforms_np.shape

        an_stack: list[np.ndarray] = []
        for x in subject_an_populations:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
            # If AN is shorter than the noisy full length, left-pad with zeros
            if arr.shape[0] < n_cols_wave:
                pre_pad = n_cols_wave - arr.shape[0]
                arr_full = np.concatenate(
                    [np.zeros(pre_pad, dtype=np.float32), arr])
            else:
                # If AN is longer, keep the most recent samples (right-aligned)
                arr_full = arr[-n_cols_wave:].astype(np.float32)
            an_stack.append(arr_full)

        if not an_stack:
            an_arr = np.zeros((n_rows_wave, n_cols_wave), dtype=np.float32)
        else:
            an_arr = np.stack(an_stack, axis=0).astype(np.float32)
            # Align number of rows: pad with zero-rows or truncate as needed
            n_rows_an = an_arr.shape[0]
            if n_rows_an < n_rows_wave:
                pad_rows = n_rows_wave - n_rows_an
                pad_block = np.zeros((pad_rows, n_cols_wave), dtype=np.float32)
                an_arr = np.vstack([an_arr, pad_block])
            elif n_rows_an > n_rows_wave:
                an_arr = an_arr[:n_rows_wave, :]

        np.save(os.path.join(subject_dir, 'an_population.npy'), an_arr)

    return waveforms_np, labels_df


def _has_legacy_dataset_files(dataset_dir: str) -> bool:
    waveform_path = os.path.join(dataset_dir, 'waveforms.npy')
    labels_path = os.path.join(dataset_dir, 'labels.csv')
    return os.path.exists(waveform_path) and os.path.exists(labels_path)


def _iter_subject_dataset_dirs(dataset_dir: str) -> list[str]:
    subject_dirs: list[str] = []
    if not os.path.isdir(dataset_dir):
        return subject_dirs

    for entry in sorted(os.scandir(dataset_dir), key=lambda item: item.name):
        if not entry.is_dir():
            continue

        waveform_path = os.path.join(entry.path, 'waveforms.npy')
        labels_path = os.path.join(entry.path, 'labels.csv')
        if os.path.exists(waveform_path) and os.path.exists(labels_path):
            subject_dirs.append(entry.path)

    return subject_dirs


def load_generated_dataset(dataset_dir: str, load_clean: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    filename = 'waveforms_clean.npy' if load_clean else 'waveforms.npy'
    if _has_legacy_dataset_files(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if not os.path.exists(path) and load_clean:
            raise FileNotFoundError(f"Clean waveforms not found in legacy dataset: {dataset_dir!r}")
        waveforms = np.load(path)
        labels_df = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'))
        return waveforms, labels_df

    subject_dirs = _iter_subject_dataset_dirs(dataset_dir)
    if not subject_dirs:
        raise FileNotFoundError(
            f'No generated dataset found in {dataset_dir!r}.')

    waveform_blocks: list[np.ndarray] = []
    label_blocks: list[pd.DataFrame] = []
    for subject_dir in subject_dirs:
        path = os.path.join(subject_dir, filename)
        if not os.path.exists(path) and load_clean:
            raise FileNotFoundError(f"Clean waveforms not found in subject directory: {subject_dir!r}")
        waveforms = np.load(path)
        labels_df = pd.read_csv(os.path.join(subject_dir, 'labels.csv'))
        if len(waveforms) != len(labels_df):
            raise RuntimeError(
                f'Shape mismatch in subject directory {subject_dir!r}: '
                f'{filename} has {len(waveforms)} rows but labels.csv has {len(labels_df)} rows.'
            )

        waveform_blocks.append(waveforms)
        label_blocks.append(labels_df)

    all_waveforms = np.concatenate(waveform_blocks, axis=0)
    all_labels = pd.concat(label_blocks, ignore_index=True)
    return all_waveforms, all_labels


def compute_label(
    clean_post: np.ndarray,
    t_post: np.ndarray,
    threshold_uv: float = 0.03,
) -> tuple[int, float, float]:
    mask = (t_post >= 4.0) & (t_post <= 12.0)
    window = clean_post[mask]
    if len(window) == 0:
        return 0, float('nan'), 0.0

    peak_index = int(np.argmax(window))
    peak_amplitude = float(window[peak_index])
    latency = float(t_post[mask][peak_index])
    if peak_amplitude >= threshold_uv:
        return 1, latency, peak_amplitude

    return 0, float('nan'), 0.0
