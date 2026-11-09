"""Synthetic ABR generation using the Verhulst 2018 MATLAB/Python model."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from tqdm import tqdm

from src.synthetic_shared import (
    compute_label,
    load_generated_dataset,
    save_subject_dataset,
    subject_dataset_exists,
)

try:
    import matlab.engine as matlab_engine
except ModuleNotFoundError:
    matlab_engine = None


# AMT 1.6.0 (https://amtoolbox.org) as used for the paper, which differs from the release
# (README.md). This generator needs its environments/verhulst2018/run_cochlear_model.py, which
# reads the subject number in a way that NumPy 2 accepts.
AMT_ROOT = str(REPOSITORY_ROOT / 'amtoolbox-1.6.0')
HELPER_ROOT = str(Path(__file__).resolve().parent / 'matlab')
NUM_SUBJECTS = 500
# dB peSPL, same stimulus calibration as the Roenne generator; 0-90 dB in 5 dB steps
STIM_LEVELS = list(range(0, 95, 5))
# Seed of the subject-parameter sampling; subject i always draws from np.random.seed([SEED, i])
SEED = 20261024

# The sample rate is the one verhulst2018 returns (fs_abr, subfs default = 20 kHz).

# Stimulus-family configuration.
# 'verhulst_click' preserves the original demo_verhulst2018-style
# rectangular click used by this generator.
# The other families reuse the Elberling toolbox stimuli also used by
# src/synthetic_roenne.py.
DEFAULT_STIMULUS_FAMILY = 'click'
STIMULUS_FAMILIES = (
    'verhulst_click',
    'click',
    'chirp1',
    'chirp2',
    'chirp3',
    'chirp4',
    'chirp5',
)

# Brainstem delay corrections (Verhulst et al. 2018, Fig. 6 caption):
#   The model's ascending pathway (AN → CN → IC) omits some synaptic delays
#   between each nucleus and the scalp electrode. The paper applies post-hoc
#   latency offsets to align simulations with clinical recordings:
#
#     Wave I  (cochlear nerve)     : +1.0 ms  — explicitly stated in paper
#     Wave III (cochlear nucleus)  : +2.25 ms — linearly interpolated between
#                                               W1 and W5 (not in paper); places
#                                               W3 peak at ~4.25 ms vs clinical ~3.5-4.5 ms
#     Wave V  (inferior colliculus): +3.5 ms  — explicitly stated in paper
#
#   Each correction is applied by inserting the corresponding number of zero
#   samples immediately after the click onset, keeping total array length fixed.
BRAINSTEM_DELAY_W1_MS:   float = 1.0
BRAINSTEM_DELAY_W3_MS:   float = 2.25
BRAINSTEM_DELAY_W5_MS:   float = 3.5

# Hearing profiles of Poles.mat: 'Normal' for the training data; the impairment sweeps of
# src/sensitivity_verhulst.py use 'Flat00' ... 'Flat35'.
DEFAULT_HEARING_PROFILES = ['Normal']

# Auditory-nerve fiber counts.
# numH=13 is the normal "no synaptopathy" setting.
# Reducing numH simulates cochlear synaptopathy (hidden hearing loss).
NUM_H_RANGE = (1, 13)    # random int, inclusive
NUM_M_RANGE = (1, 3)     # random int, inclusive
NUM_L_RANGE = (1, 3)     # random int, inclusive

# Seed for cochlear irregularities. Different seeds give slightly different BM
# patterns even with the same hearing profile.
SUBJECT_SEED_RANGE = (1, 10000000)  # random int, inclusive
ABSENT_LABEL_THRESHOLD = 0.09  # μV – used when label_criterion='amplitude'

# Unitary / ABR delay parameters
ABR_DELAY_MS_RANGE = (0.0, 3.0)


LEGACY_SUBJECT_DEFAULTS = {
    'hearing_profile': 'Normal',
    'subject_seed': 1,
    'num_h': 13,
    'num_m': 3,
    'num_l': 3,
    'abr_delay_ms': 0.0,
}

SUBJECT_PARAM_MODES = {
    'hearing_profile': 'default',
    'subject_seed': 'random',
    'num_h': 'random',
    'num_m': 'random',
    'num_l': 'random',
    'abr_delay_ms': 'default',
}

# Label criterion: 'db' uses intensity_db threshold; 'amplitude' uses peak-amplitude threshold.
LABEL_CRITERION = 'db'
DB_THRESHOLD = 50.0  # dB stimulus level – used when label_criterion='db'
_LABEL_COLUMNS = [
    'record_id',
    'subject_id',
    'intensity_db',
    'wave_v_present',
    'wave_v_latency_ms',
    'wave_v_amplitude_uv',
    'model',
    # Dataset-level constants (repeated per row for self-contained records)
    'native_sample_rate_hz',
    'pre_samp',
    'n_avg',
    # Per-subject generation parameters
    'hearing_profile',
    'subject_seed',
    'num_h',
    'num_m',
    'num_l',
    'abr_delay_ms',
]




def _normalize_stimulus_family(stimulus_family: str | None) -> str:
    family = DEFAULT_STIMULUS_FAMILY if stimulus_family is None else str(stimulus_family)
    family = family.strip().lower()
    if family not in STIMULUS_FAMILIES:
        raise ValueError(
            f'Unsupported stimulus_family {stimulus_family!r}. '
            f'Choose from: {list(STIMULUS_FAMILIES)}'
        )
    return family


def _normalize_subject_param_modes(subject_param_modes: dict[str, str] | None) -> dict[str, str]:
    normalized_modes = dict(SUBJECT_PARAM_MODES)
    if subject_param_modes is None:
        return normalized_modes

    unknown_params = sorted(set(subject_param_modes) -
                            set(SUBJECT_PARAM_MODES))
    if unknown_params:
        raise ValueError(
            'Unknown subject_param_modes keys: '
            + ', '.join(unknown_params)
        )

    for name, mode in subject_param_modes.items():
        if mode not in {'default', 'random'}:
            raise ValueError(
                f'Invalid subject_param_modes[{name!r}]={mode!r}. '
                'Use "default" or "random".'
            )
        normalized_modes[name] = mode

    return normalized_modes


def _delay_ms_to_samples(delay_ms: float, sample_rate: float) -> int:
    return int(round(delay_ms * 1e-3 * sample_rate))


# Module-level MATLAB engine, initialised once per process.
_eng: Any | None = None


def _init_engine() -> bool:
    """Start a MATLAB engine with AMT and the local helper path loaded."""
    global _eng
    if matlab_engine is None:
        print('[VERHULST] matlab.engine is not installed.')
        return False
    if _eng is not None:
        return True
    try:
        print('[VERHULST] Starting MATLAB engine…')
        _eng = matlab_engine.start_matlab()
        _eng.addpath(_eng.genpath(AMT_ROOT), nargout=0)
        _eng.addpath(HELPER_ROOT, nargout=0)
        _eng.amt_start(nargout=0)
        print('[VERHULST] MATLAB engine ready.')
        return True
    except Exception as exc:
        print(f'[VERHULST] Failed to start MATLAB engine: {exc}')
        _eng = None
        return False


def _apply_brainstem_delay(
    wave: np.ndarray,
    pre_samp: int,
    delay_samp: int,
    n_samples: int,
) -> np.ndarray:
    """Shift a wave by inserting delay_samp zeros after the click onset.

    The pre-click portion is kept in place; the delay zeros are inserted
    between it and the neural response. The result is truncated to n_samples
    so the array length stays fixed (tail samples beyond the window are lost).
    """
    padded = np.concatenate([
        np.zeros(delay_samp, dtype=np.float32),
        wave[:],
    ])
    return padded[:n_samples].astype(np.float32)


def simulate_subject(
    subject_index: int,
    config: dict[str, Any],
) -> tuple[str, list[np.ndarray], list[dict[str, Any]], list[np.ndarray]] | None:
    """Simulate one subject across all requested intensities.

    All intensity levels are passed to ``verhulst2018_full_abr_batch`` in a
    single MATLAB call so the cochlear model runs once per subject (not once
    per level).  The function returns all three brainstem waves (W1, W3, W5),
    each shifted by its brainstem delay before being summed into a composite ABR waveform.
    """
    global _eng
    if _eng is None:
        return None

    intensities: list[float] = config['intensities']
    hearing_profiles: list[str] = config['hearing_profiles']
    num_h_range: tuple[int, int] = config['num_h_range']
    subject_seed_range: tuple[int, int] = config['subject_seed_range']
    stimulus_family = _normalize_stimulus_family(
        config.get('stimulus_family', DEFAULT_STIMULUS_FAMILY))

    subject_id = f'verhulst_{subject_index:05d}'
    np.random.seed([config.get('seed', SEED), subject_index])  # reproducible per subject

    modes = config['subject_param_modes']

    if modes['hearing_profile'] == 'default':
        hearing_profile = str(LEGACY_SUBJECT_DEFAULTS['hearing_profile'])
    else:
        hearing_profile = str(np.random.choice(hearing_profiles))

    if modes['num_h'] == 'default':
        num_h = int(LEGACY_SUBJECT_DEFAULTS['num_h'])
    else:
        num_h = int(np.random.randint(num_h_range[0], num_h_range[1] + 1))

    if modes['num_m'] == 'default':
        num_m = int(LEGACY_SUBJECT_DEFAULTS['num_m'])
    else:
        num_m_range = config.get('num_m_range', (1, 3))
        num_m = int(np.random.randint(num_m_range[0], num_m_range[1] + 1))

    if modes['num_l'] == 'default':
        num_l = int(LEGACY_SUBJECT_DEFAULTS['num_l'])
    else:
        num_l_range = config.get('num_l_range', (1, 3))
        num_l = int(np.random.randint(num_l_range[0], num_l_range[1] + 1))

    if modes['subject_seed'] == 'default':
        subject_seed = int(LEGACY_SUBJECT_DEFAULTS['subject_seed'])
    else:
        subject_seed = int(np.random.randint(
            subject_seed_range[0], subject_seed_range[1] + 1))

    if modes['abr_delay_ms'] == 'default':
        abr_delay_ms = float(LEGACY_SUBJECT_DEFAULTS['abr_delay_ms'])
    else:
        abr_delay_ms = float(np.random.uniform(*config['abr_delay_ms_range']))

    # Fixed subject parameters (the impairment sweeps of src/sensitivity_verhulst.py) replace
    # the drawn ones; without 'fixed_params' nothing changes
    fixed = config.get('fixed_params') or {}
    hearing_profile = str(fixed.get('hearing_profile', hearing_profile))
    num_h = int(fixed.get('num_h', num_h))
    num_m = int(fixed.get('num_m', num_m))
    num_l = int(fixed.get('num_l', num_l))
    subject_seed = int(fixed.get('subject_seed', subject_seed))
    abr_delay_ms = float(fixed.get('abr_delay_ms', abr_delay_ms))
    # Further verhulst2018 parameters ('model_options': name -> value), passed on as name-value
    # pairs; without 'model_options' none are passed
    options = [v for name, value in (config.get('model_options') or {}).items() for v in (name, float(value))]

    try:
        import matlab
        levels_mat = matlab.double(intensities)
        w1_mat, w3_mat, w5_mat, fs_abr_mat, pre_samp_mat, an_mat = _eng.verhulst2018_full_abr_batch(
            levels_mat,
            stimulus_family,
            hearing_profile,
            float(subject_seed),
            float(num_h),
            float(num_m),
            float(num_l),
            *options,
            nargout=6,
        )
        # All matrices are (n_samples x n_levels) in Volts; convert to µV.
        w1_array = np.array(w1_mat) * 1e6
        w3_array = np.array(w3_mat) * 1e6
        w5_array = np.array(w5_mat) * 1e6
        an_array = np.array(an_mat)
    except Exception as exc:
        print(f'[VERHULST] MATLAB call failed for {subject_id}: {exc}')
        return None

    native_sample_rate = float(np.asarray(fs_abr_mat).squeeze())   # as returned by verhulst2018
    pre_samp = int(round(float(np.asarray(pre_samp_mat).squeeze())))
    dt_ms = 1000.0 / native_sample_rate
    brainstem_delay_w1_samp = _delay_ms_to_samples(
        BRAINSTEM_DELAY_W1_MS, native_sample_rate)
    brainstem_delay_w3_samp = _delay_ms_to_samples(
        BRAINSTEM_DELAY_W3_MS, native_sample_rate)
    brainstem_delay_w5_samp = _delay_ms_to_samples(
        BRAINSTEM_DELAY_W5_MS, native_sample_rate)
    abr_delay_samp = _delay_ms_to_samples(
        abr_delay_ms, native_sample_rate)

    # Time axis relative to the stimulus reference sample returned by the
    # MATLAB helper. For the original Verhulst click this is the click onset;
    # for Elberling click/chirp stimuli it matches the 15 ms toolbox reference.
    n_samples = w5_array.shape[0]
    t_post = (np.arange(n_samples) - pre_samp) * \
        dt_ms  # ms from click onset

    subject_waveforms: list[np.ndarray] = []
    subject_records: list[dict[str, Any]] = []
    subject_an_populations: list[np.ndarray] = []

    for intensity_index, level in enumerate(intensities):
        # Apply per-wave brainstem delay corrections and sum into full ABR.
        # Each wave is delayed by inserting zeros after the click onset;
        # see BRAINSTEM_DELAY_W*_MS constants for the rationale and values.
        delayed_w1 = _apply_brainstem_delay(
            w1_array[:, intensity_index].astype(np.float32),
            pre_samp, brainstem_delay_w1_samp, n_samples,
        )
        delayed_w3 = _apply_brainstem_delay(
            w3_array[:, intensity_index].astype(np.float32),
            pre_samp, brainstem_delay_w3_samp, n_samples,
        )
        delayed_w5 = _apply_brainstem_delay(
            w5_array[:, intensity_index].astype(np.float32),
            pre_samp, brainstem_delay_w5_samp, n_samples,
        )
        clean_abr = (delayed_w1 + delayed_w3 + delayed_w5).astype(np.float32)
        if abr_delay_samp > 0:
            clean_abr = _apply_brainstem_delay(
                clean_abr, pre_samp, abr_delay_samp, n_samples
            )

        an_population = an_array[:, intensity_index].astype(np.float32)

        # Measured wave V (largest value in 4-12 ms) at every level, whatever the label.
        _, latency_ms, amplitude_uv = compute_label(clean_abr, t_post, threshold_uv=-np.inf)

        if config.get('label_criterion', LABEL_CRITERION) == 'db':
            label = int(float(level) > config.get(
                'db_threshold', DB_THRESHOLD))
        else:
            label = int(amplitude_uv >= config.get(
                'amplitude_threshold', ABSENT_LABEL_THRESHOLD))
        record_id = subject_index * len(intensities) + intensity_index

        subject_waveforms.append(clean_abr)
        subject_an_populations.append(an_population)
        subject_records.append({
            'record_id': record_id,
            'subject_id': subject_id,
            'intensity_db': float(level),
            'wave_v_present': label,
            'wave_v_latency_ms': round(latency_ms, 3),
            'wave_v_amplitude_uv': round(amplitude_uv, 4),
            'model': 'verhulst',
            # Dataset-level constants
            'native_sample_rate_hz': native_sample_rate,
            'pre_samp': pre_samp,
            'n_avg': 1,
            # Per-subject generation parameters
            'hearing_profile': hearing_profile,
            'subject_seed': subject_seed,
            'num_h': num_h,
            'num_m': num_m,
            'num_l': num_l,
            'abr_delay_ms': round(abr_delay_ms, 4),
        })

    if not subject_records:
        return None

    return subject_id, subject_waveforms, subject_records, subject_an_populations


class VerhulstGenerator:
    """Generator of the Verhulst synthetic dataset."""

    def generate_dataset(
        self,
        n_subjects: int = NUM_SUBJECTS,
        intensities: list[float] | None = None,
        out_dir: str = 'datasets/synthetic_verhulst',
        stimulus_family: str = DEFAULT_STIMULUS_FAMILY,
        hearing_profiles: list[str] | None = None,
        num_h_range: tuple[int, int] = NUM_H_RANGE,
        subject_seed_range: tuple[int, int] = SUBJECT_SEED_RANGE,
        num_m_range: tuple[int, int] = NUM_M_RANGE,
        num_l_range: tuple[int, int] = NUM_L_RANGE,
        abr_delay_ms_range: tuple[float, float] = ABR_DELAY_MS_RANGE,
        subject_param_modes: dict[str, str] | None = None,
        n_avg: int = 1,
    ) -> None:
        del n_avg
        if matlab_engine is None:
            raise ModuleNotFoundError(
                'matlab.engine is required to generate the Verhulst synthetic dataset. '
                'Install the MATLAB Engine for Python before using model "verhulst".'
            )

        os.makedirs(out_dir, exist_ok=True)

        intensity_values = list(
            intensities) if intensities is not None else list(STIM_LEVELS)
        normalized_stimulus_family = _normalize_stimulus_family(stimulus_family)
        profile_list = list(hearing_profiles) if hearing_profiles is not None else list(
            DEFAULT_HEARING_PROFILES)
        normalized_subject_param_modes = _normalize_subject_param_modes(subject_param_modes)

        pending = [
            idx for idx in range(n_subjects)
            if not subject_dataset_exists(out_dir, f'verhulst_{idx:05d}')
        ]

        print(
            f'[VERHULST] Generating {n_subjects} subjects × {len(intensity_values)} intensities '
            f'(stimulus: {normalized_stimulus_family}, profiles: {profile_list}, numH range: {num_h_range})…'
        )

        if not pending:
            print(
                '[VERHULST] All requested subject datasets already exist; skipping generation.')
        else:
            if not _init_engine():
                raise RuntimeError(
                    'Could not start MATLAB engine for Verhulst generation.')

            worker_config = {
                'intensities': intensity_values,
                'stimulus_family': normalized_stimulus_family,
                'hearing_profiles': profile_list,
                'num_h_range': num_h_range,
                'subject_seed_range': subject_seed_range,
                'num_m_range': num_m_range,
                'num_l_range': num_l_range,
                'abr_delay_ms_range': abr_delay_ms_range,
                'subject_param_modes': normalized_subject_param_modes,
                'label_criterion': LABEL_CRITERION,
                'db_threshold': DB_THRESHOLD,
                'amplitude_threshold': ABSENT_LABEL_THRESHOLD,
                'seed': SEED,
            }

            # Process subjects sequentially: verhulst2018 already uses an
            # internal Python subprocess pool for cochlear computation, so
            # wrapping it in an outer multiprocessing pool would cause nested
            # process spawning and is avoided here.
            for subject_index in tqdm(pending, desc='Simulating subjects (Verhulst)'):
                result = simulate_subject(subject_index, worker_config)
                if result is None:
                    print(
                        f'[VERHULST] Subject {subject_index} failed; skipping.')
                    continue

                subject_id, subject_waveforms, subject_records, subject_an_populations = result
                ordered_records = [
                    {col: rec[col] for col in _LABEL_COLUMNS}
                    for rec in subject_records
                ]
                save_subject_dataset(
                    out_dir,
                    subject_id,
                    subject_waveforms,
                    ordered_records,
                    None,
                    subject_an_populations,
                )

        if not any(
            subject_dataset_exists(out_dir, f'verhulst_{idx:05d}')
            for idx in range(n_subjects)
        ):
            print('[VERHULST] No data generated.')
            return

        waveforms_np, labels_df = load_generated_dataset(out_dir)
        labels_df = labels_df[_LABEL_COLUMNS]
        present_pct = labels_df['wave_v_present'].mean() * 100
        print(f'[VERHULST] Saved {len(labels_df)} records -> {out_dir}')
        print(
            f'           Shape: {waveforms_np.shape}  dtype: {waveforms_np.dtype}')
        print(f'           Wave-V present: {present_pct:.1f}%')


def generate_dataset(
    n_subjects: int = NUM_SUBJECTS,
    intensities: list[float] | None = None,
    out_dir: str = 'datasets/synthetic_verhulst',
    stimulus_family: str = DEFAULT_STIMULUS_FAMILY,
    hearing_profiles: list[str] | None = None,
    num_h_range: tuple[int, int] = NUM_H_RANGE,
    num_m_range: tuple[int, int] = NUM_M_RANGE,
    num_l_range: tuple[int, int] = NUM_L_RANGE,
    subject_seed_range: tuple[int, int] = SUBJECT_SEED_RANGE,
    abr_delay_ms_range: tuple[float, float] = ABR_DELAY_MS_RANGE,
    subject_param_modes: dict[str, str] | None = None,
) -> None:
    VerhulstGenerator().generate_dataset(
        n_subjects=n_subjects,
        intensities=intensities,
        out_dir=out_dir,
        stimulus_family=stimulus_family,
        hearing_profiles=hearing_profiles,
        num_h_range=num_h_range,
        num_m_range=num_m_range,
        num_l_range=num_l_range,
        subject_seed_range=subject_seed_range,
        abr_delay_ms_range=abr_delay_ms_range,
        subject_param_modes=subject_param_modes,
    )

def main() -> None:
    generate_dataset()


if __name__ == '__main__':
    main()
