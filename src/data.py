"""Data for the experiments: real evaluation sets, synthetic training records, label rules,
train/validation splits and the torch datasets built from them.

File reading and preprocessing (band-pass, resampling, subject-level min-max normalisation;
for PhysioNet separately per tone-burst frequency, see NORM_GROUP) are in src/data_loader.py;
this module assembles them. A run's training subjects depend only on its training sources,
development size and Monte Carlo index, so different architectures, label rules and augmentations see
the same subjects.

Training sources:
  'wang' (Mendeley), 'southampton', 'physionet'    real datasets (6 train + 2 validation)
  'roenne', 'verhulst', 'gaussian', 'gaussian_latency', 'simple_gaussian'  datasets/synthetic_<name>
  'mixup', 'cutmix'                                Roenne/Verhulst waveform mixing
A tuple of synthetic sources, e.g. ('roenne', 'verhulst'), splits the development size equally.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from src.augment import cutmix_waveforms, make_transform, mixup_waveforms, shifted_augment, \
    add_pink_noise, add_uniform_noise, dc_shift_augment
from src.data_loader import normalize_records, pad_or_truncate, prepare_records_for_dataset, \
    preprocess_synthetic_waveform
from src.synthetic_shared import load_generated_dataset

Record = dict[str, Any]

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_ROOT = Path(__file__).resolve().parents[1] / 'datasets'
WANG_ROOT = DATA_ROOT / 'wang_single' / 'human_raw_data'
WANG_LABEL_CSV = DATA_ROOT / 'wang_single_label.csv'
MAT_PATH = DATA_ROOT / 'Subject_ABR_Data.mat'
SOTON_LABEL_CSV = DATA_ROOT / 'Subject_ABR_label.csv'
PHYSIONET_NPZ = DATA_ROOT / 'physionet_uv.npz'   # written by src/prepare_physionet.py
PHYSIONET_LABEL_CSV = DATA_ROOT / 'physionet_label.csv'

TARGET_RATE = 20_000
POST_LEN = 300                       # 15 ms after the stimulus at 20 kHz
NORM_METHOD = 'min_max'
# Min-max statistics per subject, and per stimulus frequency within a subject where records
# carry 'freq_khz' (PhysioNet: 1 and 4 kHz tone bursts, normalised separately so the smaller
# 1 kHz responses are not scaled by the 4 kHz maximum). 'subject' pools both frequencies;
# Mendeley, Southampton and the synthetic sources have one stimulus and are the same either way.
NORM_GROUP = 'subject_frequency'
BASE_SEED = 48431231                 # split and training seed of run cv = BASE_SEED + cv

HUMAN_SOURCES = ('wang', 'southampton', 'physionet')
MIX_SOURCES = ('mixup', 'cutmix')
DATASET_NAMES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
# Southampton subjects reserved for augmentation tuning: never trained or tested on
HELD_OUT_SOTON_SUBJECTS = {'soton_s08', 'soton_s09', 'soton_s10', 'soton_s11'}
AUGDEV_SET = 'southampton_augdev'
MIX_LABEL_TYPE = 'hard'              # MixUp/CutMix label: from a source drawn with probability equal to its share

# ==============================================================================
# HUMAN DATA
# ==============================================================================

def copy_records(records: list[Record]) -> list[Record]:
    out = []
    for r in records:
        rc = dict(r)
        if 'waveform_proc' in rc:
            rc['waveform_proc'] = rc['waveform_proc'].copy()
        out.append(rc)
    return out


def normalise(records: list[Record]) -> list[Record]:
    """Subject-level min-max normalisation of a copy of the records (NORM_GROUP: PhysioNet per
    subject and tone-burst frequency)."""
    return normalize_records(copy_records(records), NORM_METHOD, NORM_GROUP, {})


def load_human_datasets() -> dict[str, list[Record]]:
    """The three real datasets, preprocessed and labelled (Southampton without the tuning subjects)."""
    all_recs = prepare_records_for_dataset(
        wang_root=str(WANG_ROOT), mat_path=str(MAT_PATH), target_len=POST_LEN, target_rate=float(TARGET_RATE),
        soton_label_csv=str(SOTON_LABEL_CSV), wang_label_csv=str(WANG_LABEL_CSV),
        physionet_path=str(PHYSIONET_NPZ), physionet_label_csv=str(PHYSIONET_LABEL_CSV))
    by_source = {name: [r for r in all_recs if r.get('header', {}).get('source') == name] for name in HUMAN_SOURCES}
    return by_source


def load_eval_sets() -> dict[str, list[Record]]:
    """Normalised evaluation sets: the three test datasets and the Southampton tuning subjects."""
    by_source = load_human_datasets()
    sets = {
        'wang': by_source['wang'],
        'southampton': [r for r in by_source['southampton'] if r['subject'] not in HELD_OUT_SOTON_SUBJECTS],
        'physionet': by_source['physionet'],
        AUGDEV_SET: [r for r in by_source['southampton'] if r['subject'] in HELD_OUT_SOTON_SUBJECTS],
    }
    return {name: normalise(recs) for name, recs in sets.items()}


def waveform_table(eval_sets: dict[str, list[Record]]) -> pd.DataFrame:
    """One row per evaluation waveform; waveform_id is its position across the sets in order."""
    rows = []
    for eval_set, recs in eval_sets.items():
        for k, r in enumerate(recs):
            rows.append({
                'waveform_id': len(rows), 'eval_set': eval_set, 'index_in_set': k,
                'dataset': DATASET_NAMES['southampton' if eval_set == AUGDEV_SET else eval_set],
                'role': 'augdev' if eval_set == AUGDEV_SET else 'test',
                'subject': r['subject'], 'intensity_db': r['intensity'], 'freq_khz': r.get('freq_khz'),
                'label_absent': int(r['label']),
            })
    return pd.DataFrame(rows)

# ==============================================================================
# SYNTHETIC DATA AND LABEL RULES
# ==============================================================================
_synthetic_cache: dict[str, list[Record]] = {}


def load_synthetic_folder(folder: str | Path, name: str) -> tuple[list[Record], pd.DataFrame]:
    """Records of a generated dataset folder, preprocessed like the real data (not normalised),
    and its labels table (one row per record, in the same order)."""
    waveforms, labels_df = load_generated_dataset(str(folder))
    native_rate = float(labels_df['native_sample_rate_hz'].iloc[0])
    records = []
    for idx, row in labels_df.iterrows():
        wf = preprocess_synthetic_waveform(waveforms[idx], native_rate, float(TARGET_RATE))
        offset = int(round(int(row['pre_samp']) * TARGET_RATE / native_rate))
        records.append({
            'waveform_proc': pad_or_truncate(wf[offset:], POST_LEN),
            'label': 1 - int(row['wave_v_present']),
            'subject': f'{name}_{row["subject_id"]}',
            'intensity': float(row['intensity_db']),
            'source': name,
        })
    return records, labels_df


def load_synthetic(name: str) -> list[Record]:
    """Records of datasets/synthetic_<name>, preprocessed like the real data (not normalised)."""
    if name not in _synthetic_cache:
        _synthetic_cache[name] = load_synthetic_folder(DATA_ROOT / f'synthetic_{name}', name)[0]
    return _synthetic_cache[name]


def apply_label_rule(records: list[Record], rule: str) -> list[Record]:
    """Select levels and set labels (1 = wave V absent) of synthetic records.

    'paper'   0-20 dB absent, 70-90 dB present, other levels left out
    'stored'  every level, label as stored in labels.csv
    'cut<X>'  every level, present if level > X dB (e.g. 'cut40')
    """
    if rule == 'paper':
        return [dict(r, label=int(r['intensity'] <= 20.0)) for r in records
                if r['intensity'] <= 20.0 or r['intensity'] >= 70.0]
    if rule == 'stored':
        return list(records)
    if rule.startswith('cut'):
        cut = float(rule[3:])
        return [dict(r, label=int(r['intensity'] <= cut)) for r in records]
    raise ValueError(f'Unknown label rule {rule!r}')

# ==============================================================================
# TORCH DATASETS
# ==============================================================================

class ABRDataset(Dataset[tuple[Tensor, Tensor]]):
    """Preprocessed records -> (waveform (1, n), label), with an optional training transform."""

    def __init__(self, records: list[Record], transform: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        self.records = [r for r in records if 'waveform_proc' in r]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        record = self.records[idx]
        waveform = np.asarray(record['waveform_proc'], dtype=np.float32)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return (torch.from_numpy(np.expand_dims(waveform, axis=0)),
                torch.tensor(float(record['label']), dtype=torch.float32))


class MixupABRDataset(Dataset[tuple[Tensor, Tensor]]):
    """On-the-fly MixUp/CutMix of a Roenne and a Verhulst record at the same level.

    augment=False gives a fixed validation set; with noisy=True the sources' augmentation is
    applied to it as in training, drawn once per item (the same on every evaluation).
    The label, and the noise and DC shift added after mixing, are those of one source, drawn
    with probability equal to its share of the mixed waveform.

    CutMix: the base waveform, whose segment is replaced by the other source, is Roenne or
    Verhulst with equal probability, so every time point comes from each source half of the
    time. The validation cut position is drawn with the item's own generator, so it is fixed too.
    """

    def __init__(self, roenne_records: list[Record], verhulst_records: list[Record], num_output_subjects: int,
                 seed: int | None = None, augment: bool = True, dataset_augmentations: dict | None = None,
                 label_type: str = MIX_LABEL_TYPE, mode: str = 'mixup', noisy: bool = False) -> None:
        self.augment = augment
        self.noisy = noisy
        self.base_seed = seed if seed is not None else 42
        self.num_output_subjects = num_output_subjects
        self.dataset_augmentations = dataset_augmentations
        self.label_type = label_type
        self.mode = mode
        self.verhulst_by_intensity: dict[float, list[Record]] = {}
        for r in verhulst_records:
            self.verhulst_by_intensity.setdefault(r['intensity'], []).append(r)
        self.roenne_by_intensity: dict[float, list[Record]] = {}
        for r in roenne_records:
            self.roenne_by_intensity.setdefault(r['intensity'], []).append(r)
        self.intensities = sorted(self.roenne_by_intensity)
        self.num_intensities = len(self.intensities)

    def __len__(self) -> int:
        return self.num_output_subjects * self.num_intensities

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        intensity = self.intensities[idx % self.num_intensities]
        r_candidates = self.roenne_by_intensity[intensity]
        v_candidates = self.verhulst_by_intensity[intensity]

        if self.augment:
            r_rec = r_candidates[np.random.randint(0, len(r_candidates))]
            v_rec = v_candidates[np.random.randint(0, len(v_candidates))]
            alpha = np.random.uniform(0.0, 1.0)
            label_draw = np.random.rand()
            # CutMix base: Roenne or Verhulst with equal probability (MixUp draws nothing here)
            verhulst_base = np.random.rand() < 0.5 if self.mode == 'cutmix' else True
            cut_rng = None
        else:
            # Fixed selection for validation
            local_rng = np.random.default_rng(self.base_seed + idx)
            r_rec = r_candidates[local_rng.integers(0, len(r_candidates))]
            v_rec = v_candidates[local_rng.integers(0, len(v_candidates))]
            alpha = local_rng.uniform(0.0, 1.0)
            label_draw = local_rng.random()
            verhulst_base = local_rng.random() < 0.5 if self.mode == 'cutmix' else True
            cut_rng = local_rng

        wf_r = r_rec['waveform_proc']
        wf_v = v_rec['waveform_proc']
        apply_augmentation = bool(self.dataset_augmentations) and (self.augment or self.noisy)
        rng_state = None
        if apply_augmentation and not self.augment:
            rng_state = np.random.get_state()
            np.random.seed(self.base_seed + 200000 + idx)
        # Shift each source before mixing
        if apply_augmentation:
            for name in ('roenne', 'verhulst'):
                shift = self.dataset_augmentations.get(name, {}).get('shift_limit_ms')
                if shift is not None:
                    samples = int(round(shift * TARGET_RATE / 1000.0))
                    if name == 'roenne':
                        wf_r = shifted_augment(wf_r, samples)
                    else:
                        wf_v = shifted_augment(wf_v, samples)

        if self.mode == 'mixup':
            mixed, verhulst_share = mixup_waveforms(wf_v, wf_r, alpha), alpha
        elif self.mode == 'cutmix':
            base, segment = (wf_v, wf_r) if verhulst_base else (wf_r, wf_v)
            mixed, kept = cutmix_waveforms(base, segment, alpha, cut_rng)
            verhulst_share = kept if verhulst_base else 1.0 - kept
        else:
            raise ValueError(f'Unknown blend mode: {self.mode}')

        choose_verhulst_label = label_draw < verhulst_share
        if self.label_type == 'soft':
            label = verhulst_share * float(v_rec['label']) + (1.0 - verhulst_share) * float(r_rec['label'])
        else:
            label = v_rec['label'] if choose_verhulst_label else r_rec['label']

        # Noise and DC shift of the source whose label was taken
        if apply_augmentation:
            config = self.dataset_augmentations.get('verhulst' if choose_verhulst_label else 'roenne', {})
            if config.get('pink_noise_sigma') is not None:
                mixed = add_pink_noise(mixed, config['pink_noise_sigma'], float(TARGET_RATE))
            if config.get('uniform_noise_amplitude') is not None:
                mixed = add_uniform_noise(mixed, config['uniform_noise_amplitude'])
            if config.get('dc_shift_max') is not None:
                mixed = dc_shift_augment(mixed, config['dc_shift_max'])
        if rng_state is not None:
            np.random.set_state(rng_state)

        return (torch.from_numpy(np.expand_dims(mixed, axis=0)),
                torch.tensor(float(label), dtype=torch.float32))

# ==============================================================================
# SPLITS
# ==============================================================================

def noisy_copy(records: list[Record], config: dict, seed: int) -> list[Record]:
    """Records with a source's training augmentation applied once, with a fixed seed."""
    transform = make_transform(config, TARGET_RATE)
    if transform is None:
        return records
    state = np.random.get_state()
    np.random.seed(seed)
    out = [dict(r, waveform_proc=transform(r['waveform_proc'])) for r in records]
    np.random.set_state(state)
    return out


def split_sizes(n_dev: int, n_sources: int) -> tuple[int, int]:
    """Training and validation subjects per source: 75 / 25 of n_dev, divided equally."""
    n_val_total = max(1, n_dev // 4)
    return (n_dev - n_val_total) // n_sources, n_val_total // n_sources


def build_training_data(sources: tuple[str, ...], n_dev: int, cv: int, label_rule: str,
                        augmentations: dict[str, dict]) -> tuple[Dataset, Dataset, np.ndarray]:
    """Training set (augmented), validation set and the training labels (for class weighting).

    A synthetic source whose augmentation has 'noisy_validation': True also gets that
    augmentation applied once to its validation waveforms (fixed seed), so the validation
    loss used for early stopping is measured on noisy data. MixUp/CutMix: when Roenne or
    Verhulst has it, the mixed validation waveforms get the augmentation as in training.
    """
    seed = BASE_SEED + cv

    if len(sources) == 1 and sources[0] in HUMAN_SOURCES:
        name = sources[0]
        records = load_human_datasets()[name]
        if name == 'southampton':
            records = [r for r in records if r['subject'] not in HELD_OUT_SOTON_SUBJECTS]
        subjects = sorted({r['subject'] for r in records})
        if len(subjects) != n_dev:
            raise ValueError(f'{name} has {len(subjects)} development subjects, not n_dev={n_dev}')
        n_train, _ = split_sizes(n_dev, 1)
        random.Random(seed).shuffle(subjects)
        train_subj, val_subj = set(subjects[:n_train]), set(subjects[n_train:])
        train = normalise([r for r in records if r['subject'] in train_subj])
        val = normalise([r for r in records if r['subject'] in val_subj])
        train_ds = ABRDataset(train, make_transform(augmentations.get(name), TARGET_RATE))
        return train_ds, ABRDataset(val), np.array([r['label'] for r in train], dtype=np.float32)

    if len(sources) == 1 and sources[0] in MIX_SOURCES:
        n_train, n_val = split_sizes(n_dev, 2)
        recs = {m: apply_label_rule(load_synthetic(m), label_rule) for m in ('roenne', 'verhulst')}
        rng = random.Random(seed)            # one stream for both sources
        parts = {}
        for m in ('roenne', 'verhulst'):
            subjects = sorted({r['subject'] for r in recs[m]})
            rng.shuffle(subjects)
            parts[m] = (set(subjects[:n_train]), set(subjects[n_train:n_train + n_val]))
        train = {m: normalise([r for r in recs[m] if r['subject'] in parts[m][0]]) for m in recs}
        val = {m: normalise([r for r in recs[m] if r['subject'] in parts[m][1]]) for m in recs}
        mode = sources[0]
        train_ds = MixupABRDataset(train['roenne'], train['verhulst'], 2 * n_train, seed=seed, augment=True,
                                   dataset_augmentations=augmentations, mode=mode)
        noisy_val = any((augmentations.get(m) or {}).get('noisy_validation') for m in ('roenne', 'verhulst'))
        val_ds = MixupABRDataset(val['roenne'], val['verhulst'], 2 * n_val, seed=seed + 100000, augment=False,
                                 dataset_augmentations=augmentations, mode=mode, noisy=noisy_val)
        labels = np.array([r['label'] for m in ('roenne', 'verhulst') for r in train[m]], dtype=np.float32)
        return train_ds, val_ds, labels

    # One or more synthetic sources, each split with its own stream seeded the same way
    n_train, n_val = split_sizes(n_dev, len(sources))
    train_sets, val_sets, labels = [], [], []
    for m in sources:
        recs = apply_label_rule(load_synthetic(m), label_rule)
        subjects = sorted({r['subject'] for r in recs})
        if len(subjects) < n_train + n_val:
            raise ValueError(f'{m} has {len(subjects)} subjects; {n_train} + {n_val} requested')
        random.Random(seed).shuffle(subjects)
        train_subj, val_subj = set(subjects[:n_train]), set(subjects[n_train:n_train + n_val])
        train = normalise([r for r in recs if r['subject'] in train_subj])
        train_sets.append(ABRDataset(train, make_transform(augmentations.get(m), TARGET_RATE)))
        val = normalise([r for r in recs if r['subject'] in val_subj])
        if (augmentations.get(m) or {}).get('noisy_validation'):
            val = noisy_copy(val, augmentations[m], seed + 200000)
        val_sets.append(ABRDataset(val))
        labels += [r['label'] for r in train]
    return ConcatDataset(train_sets), ConcatDataset(val_sets), np.array(labels, dtype=np.float32)
