"""Parallel synthetic ABR generation using the Roenne 2012 MATLAB model.

SUBJECT_PARAM_MODES can also vary (all kept at their defaults here):

* stimulus-family variation across Elberling click / chirp stimuli
* cochlear-coverage variation via flow / fhigh / numCF
* single-kernel readout latency variation via delayed unitary response
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
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
# (README.md). This generator needs its models/zilany2007.m, which takes 'cohc', 'cihc' and
# 'spont' (passed by roenne2012_varied_batch.m).
AMT_ROOT = str(REPOSITORY_ROOT / 'amtoolbox-1.6.0')
HELPER_ROOT = Path(__file__).resolve().parent / 'matlab'
NUM_SUBJECTS = 500
# dB peSPL (zilany2007 calibration), 0-90 dB in 5 dB steps
STIM_LEVELS = list(range(0, 95, 5))

# Where the subject parameters come from:
#   'stored'  read subject i from STORED_PARAMS_CSV, the parameters of the paper's 500 subjects,
#             which regenerates the paper's Roenne set (README.md)
#   'sample'  draw new subjects: subject i draws the parameters set to 'random' in
#             SUBJECT_PARAM_MODES, from the ranges below, with np.random.seed([SEED, i])
SUBJECT_PARAMS = 'stored'
STORED_PARAMS_CSV = REPOSITORY_ROOT / 'datasets' / 'roenne_subject_params.csv'
# Seed of the subject-parameter sampling (SUBJECT_PARAMS = 'sample')
SEED = 20261024

SOURCE_FS = 30_000.0

STIMULUS_FAMILIES = (
    'click',
    # 'chirp1',
    # 'chirp2',
    # 'chirp3',
    # 'chirp4',
    # 'chirp5',
)
FLOW_RANGE = (80.0, 200.0)
FHIGH_RANGE = (15_000.0, 16_000.0)
NUM_CF_RANGE = (400, 500)
UR_DELAY_MS_RANGE = (0.0, 3.0)
COHC_RANGE = (0.1, 1.0)
CIHC_RANGE = (0.1, 1.0)
SPONT_RANGE = (1.0, 150.0)

# Values of the parameters set to 'default' in SUBJECT_PARAM_MODES (the stock AMT
# roenne2012/zilany2007 defaults).
LEGACY_SUBJECT_DEFAULTS = {
    'stimulus_family': 'click',
    'flow_hz': 100.0,
    'fhigh_hz': 16_000.0,
    'num_cf': 500,
    'ur_delay_ms': 0.0,
    'cohc': 1.0,
    'cihc': 1.0,
    'spont': 50.0,
}

# Edit this map to choose whether each subject-level parameter should use the
# default value or be sampled randomly from the range/list above
# (SUBJECT_PARAMS = 'sample'; 'stored' reads every parameter from STORED_PARAMS_CSV).
#
# "default" uses LEGACY_SUBJECT_DEFAULTS.
# "random" samples a new value for each synthetic subject.

# Stimulus/protocol: stimulus_family, intensity_db
# Cochlear coverage/anatomy: flow_hz, fhigh_hz, num_cf
# Cochlear physiology: cohc, cihc
# Auditory nerve neurophysiology: spont
# Brainstem/readout: ur_delay_ms
# Label/training only: DB_THRESHOLD, ABSENT_LABEL_THRESHOLD
# (No recording noise is simulated: noise is added by training-time augmentation.)

SUBJECT_PARAM_MODES = {
    'stimulus_family': 'default',
    'flow_hz': 'default',
    'fhigh_hz': 'default',
    'num_cf': 'default',
    'ur_delay_ms': 'default',
    'cohc': 'random',
    'cihc': 'random',
    'spont': 'random',
}

ABSENT_LABEL_THRESHOLD = 0.1  # μV – used when label_criterion='amplitude'

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
    # Stimulus / cochlear-coverage settings
    'stimulus_family',
    'flow_hz',
    'fhigh_hz',
    'num_cf',
    # Single-kernel readout setting
    'ur_delay_ms',
    # Per-subject generation parameters (actual values used by roenne2012.m)
    'cohc',
    'cihc',
    'spont',
]




eng: Any = None


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


def _sample_coverage_params(config: dict[str, Any]) -> tuple[float, float, int]:
    modes = config['subject_param_modes']
    flow_min, flow_max = map(float, config['flow_range'])
    fhigh_min, fhigh_max = map(float, config['fhigh_range'])
    num_cf_min, num_cf_max = map(int, config['num_cf_range'])

    if modes['flow_hz'] == 'default':
        flow_hz = float(LEGACY_SUBJECT_DEFAULTS['flow_hz'])
    else:
        fhigh_ceiling = (
            float(LEGACY_SUBJECT_DEFAULTS['fhigh_hz'])
            if modes['fhigh_hz'] == 'default'
            else fhigh_max
        )
        safe_flow_max = min(flow_max, fhigh_ceiling - 1000.0)
        if safe_flow_max < flow_min:
            raise ValueError(
                f'Invalid cochlear coverage ranges: flow_range={config["flow_range"]!r}, '
                f'fhigh_range={config["fhigh_range"]!r}. Need at least 1000 Hz separation.'
            )
        flow_hz = float(np.random.uniform(flow_min, safe_flow_max))

    if modes['fhigh_hz'] == 'default':
        fhigh_hz = float(LEGACY_SUBJECT_DEFAULTS['fhigh_hz'])
        if fhigh_hz < flow_hz + 1000.0:
            raise ValueError(
                f'Legacy/default fhigh_hz={fhigh_hz!r} is incompatible with flow_hz={flow_hz!r}. '
                'Need at least 1000 Hz separation.'
            )
    else:
        low_fhigh = max(fhigh_min, flow_hz + 1000.0)
        if low_fhigh > fhigh_max:
            raise ValueError(
                f'Invalid cochlear coverage ranges: flow_range={config["flow_range"]!r}, '
                f'fhigh_range={config["fhigh_range"]!r}. Need at least 1000 Hz separation.'
            )
        fhigh_hz = float(np.random.uniform(low_fhigh, fhigh_max))

    if modes['num_cf'] == 'default':
        num_cf = int(LEGACY_SUBJECT_DEFAULTS['num_cf'])
    else:
        num_cf = int(np.random.randint(num_cf_min, num_cf_max + 1))

    return flow_hz, fhigh_hz, num_cf


def _sample_readout_params(config: dict[str, Any]) -> float:
    modes = config['subject_param_modes']

    if modes['ur_delay_ms'] == 'default':
        return float(LEGACY_SUBJECT_DEFAULTS['ur_delay_ms'])

    return float(np.random.uniform(*config['ur_delay_ms_range']))


def _sample_subject_params(config: dict[str, Any]) -> dict[str, Any]:
    modes = config['subject_param_modes']

    stimulus_family = (
        str(LEGACY_SUBJECT_DEFAULTS['stimulus_family'])
        if modes['stimulus_family'] == 'default'
        else str(np.random.choice(config['stimulus_families']))
    )
    flow_hz, fhigh_hz, num_cf = _sample_coverage_params(config)
    ur_delay_ms = _sample_readout_params(config)

    cohc = (
        float(LEGACY_SUBJECT_DEFAULTS['cohc'])
        if modes['cohc'] == 'default'
        else float(np.random.uniform(*config['cohc_range']))
    )
    cihc = (
        float(LEGACY_SUBJECT_DEFAULTS['cihc'])
        if modes['cihc'] == 'default'
        else float(np.random.uniform(*config['cihc_range']))
    )
    spont = (
        float(LEGACY_SUBJECT_DEFAULTS['spont'])
        if modes['spont'] == 'default'
        else float(np.random.uniform(*config['spont_range']))
    )

    return {
        'stimulus_family': stimulus_family,
        'flow_hz': flow_hz,
        'fhigh_hz': fhigh_hz,
        'num_cf': num_cf,
        'ur_delay_ms': ur_delay_ms,
        'cohc': cohc,
        'cihc': cihc,
        'spont': spont,
    }


_STORED_PARAM_TYPES = {
    'stimulus_family': str,
    'flow_hz': float,
    'fhigh_hz': float,
    'num_cf': int,
    'ur_delay_ms': float,
    'cohc': float,
    'cihc': float,
    'spont': float,
}


def _load_stored_params(n_subjects: int) -> list[dict[str, Any]]:
    """Parameters of subjects 0 to n_subjects - 1 from STORED_PARAMS_CSV (one row per subject)."""
    table = pd.read_csv(STORED_PARAMS_CSV)
    if n_subjects > len(table):
        raise ValueError(
            f'{STORED_PARAMS_CSV} holds {len(table)} subjects, {n_subjects} requested. '
            "Set SUBJECT_PARAMS = 'sample' to draw new subjects."
        )
    table = table.iloc[:n_subjects]
    expected_ids = [f'roenne_{subject_index:05d}' for subject_index in range(n_subjects)]
    if table['subject_id'].tolist() != expected_ids:
        raise ValueError(f'{STORED_PARAMS_CSV} rows are not roenne_00000, roenne_00001, ... in order.')
    return [
        {name: cast(row[name]) for name, cast in _STORED_PARAM_TYPES.items()}
        for row in table.to_dict('records')
    ]


def init_worker() -> None:
    """Start a MATLAB engine and load AMT plus local helper paths once per worker."""
    global eng
    if matlab_engine is None:
        print(f'[{os.getpid()}] MATLAB engine is not installed.')
        eng = None
        return

    try:
        print(f'[{os.getpid()}] Starting MATLAB engine...')
        eng = matlab_engine.start_matlab()
        eng.addpath(eng.genpath(AMT_ROOT), nargout=0)
        eng.addpath(str(HELPER_ROOT), nargout=0)
        eng.amt_start(nargout=0)
    except Exception as exc:
        print(f'[{os.getpid()}] Failed to initialize MATLAB: {exc}')
        eng = None


def simulate_subject(task: tuple[int, dict[str, Any]]) -> tuple[str, list[np.ndarray], list[dict[str, Any]], list[np.ndarray]] | None:
    """Worker task: simulate one subject across all requested intensities."""
    global eng
    if eng is None:
        return None

    import matlab

    subject_index, config = task
    subject_id = f'roenne_{subject_index:05d}'
    if config.get('stored_params') is not None:
        subject_params = dict(config['stored_params'])
    else:
        # Reproducible per subject, whichever worker runs it
        np.random.seed([config.get('seed', SEED), subject_index])
        subject_params = _sample_subject_params(config)
    stimulus_family = subject_params['stimulus_family']
    flow_hz = subject_params['flow_hz']
    fhigh_hz = subject_params['fhigh_hz']
    num_cf = subject_params['num_cf']
    ur_delay_ms = subject_params['ur_delay_ms']
    cohc = subject_params['cohc']
    cihc = subject_params['cihc']
    spont = subject_params['spont']

    subject_waveforms: list[np.ndarray] = []
    subject_records: list[dict[str, Any]] = []
    subject_an_populations: list[np.ndarray] = []

    try:
        levels_mat = matlab.double(
            [[float(level) for level in config['intensities']]])
        simpot_matrix, fs_out, pre_samp, ansum_matrix = eng.roenne2012_varied_batch(
            levels_mat,
            stimulus_family,
            float(flow_hz),
            float(fhigh_hz),
            float(num_cf),
            float(cohc),
            float(cihc),
            float(spont),
            float(ur_delay_ms),
            nargout=4,
        )
    except Exception as exc:
        print(f'[{os.getpid()}] Error simulating {subject_id}: {exc}')
        return None

    simpot_np = np.asarray(simpot_matrix, dtype=np.float32)
    if simpot_np.ndim == 1:
        simpot_np = simpot_np[:, None]

    ansum_np = np.asarray(ansum_matrix, dtype=np.float32)
    if ansum_np.ndim == 1:
        ansum_np = ansum_np[:, None]

    native_sample_rate = float(np.asarray(fs_out).squeeze())
    pre_samp_int = int(round(float(np.asarray(pre_samp).squeeze())))
    dt_ms = 1000.0 / native_sample_rate

    for intensity_index, level in enumerate(config['intensities']):
        clean_post_source = simpot_np[:, intensity_index].astype(np.float32)
        an_population = ansum_np[:, intensity_index].astype(np.float32)
        t_post = (np.arange(len(clean_post_source), dtype=np.float32) - pre_samp_int) * dt_ms

        # Saved at native 30 kHz without noise; bandpass + resample to 20 kHz is
        # applied at load time via preprocess_synthetic_waveform().

        # Measured wave V (largest value in 4-12 ms) at every level, whatever the label.
        _, latency_ms, amplitude_uv = compute_label(clean_post_source, t_post, threshold_uv=-np.inf)

        if config.get('label_criterion', LABEL_CRITERION) == 'db':
            label = int(float(level) > config.get(
                'db_threshold', DB_THRESHOLD))
        else:
            label = int(amplitude_uv >= config.get(
                'amplitude_threshold', ABSENT_LABEL_THRESHOLD))
        record_id = subject_index * \
            len(config['intensities']) + intensity_index

        subject_waveforms.append(clean_post_source.astype(np.float32))
        subject_an_populations.append(an_population)
        subject_records.append({
            'record_id': record_id,
            'subject_id': subject_id,
            'intensity_db': float(level),
            'wave_v_present': label,
            'wave_v_latency_ms': round(latency_ms, 3),
            'wave_v_amplitude_uv': round(amplitude_uv, 4),
            'model': 'roenne',
            # Dataset-level constants (repeated per row for self-contained records)
            'native_sample_rate_hz': native_sample_rate,
            'pre_samp': pre_samp_int,
            'n_avg': 1,
            # Stimulus / cochlear-coverage settings
            'stimulus_family': stimulus_family,
            'flow_hz': round(flow_hz, 2),
            'fhigh_hz': round(fhigh_hz, 2),
            'num_cf': num_cf,
            'ur_delay_ms': round(ur_delay_ms, 4),
            # Per-subject generation parameters
            'cohc': round(cohc, 4),
            'cihc': round(cihc, 4),
            'spont': round(spont, 3),
        })

    if not subject_records:
        return None

    return subject_id, subject_waveforms, subject_records, subject_an_populations


class RoenneGenerator:
    """Generator of the Roenne synthetic dataset."""

    def __init__(
        self,
        source_fs: float = SOURCE_FS,
    ) -> None:
        self.source_fs = float(source_fs)

    def generate_dataset(
        self,
        n_subjects: int = NUM_SUBJECTS,
        intensities: list[float] | None = None,
        out_dir: str = 'datasets/synthetic_roenne',
        stimulus_families: tuple[str, ...] = STIMULUS_FAMILIES,
        flow_range: tuple[float, float] = FLOW_RANGE,
        fhigh_range: tuple[float, float] = FHIGH_RANGE,
        num_cf_range: tuple[int, int] = NUM_CF_RANGE,
        ur_delay_ms_range: tuple[float, float] = UR_DELAY_MS_RANGE,
        cohc_range: tuple[float, float] = COHC_RANGE,
        cihc_range: tuple[float, float] = CIHC_RANGE,
        spont_range: tuple[float, float] = SPONT_RANGE,
        subject_param_modes: dict[str, str] | None = None,
        subject_params: str = SUBJECT_PARAMS,
        n_avg: int = 1,
        label_criterion: str = LABEL_CRITERION,
        db_threshold: float = DB_THRESHOLD,
        amplitude_threshold: float = ABSENT_LABEL_THRESHOLD,
    ) -> None:
        del n_avg
        if matlab_engine is None:
            raise ModuleNotFoundError(
                'matlab.engine is required to generate the Roenne synthetic dataset. '
                'Install the MATLAB Engine for Python before using model "roenne".'
            )
        if subject_params not in {'stored', 'sample'}:
            raise ValueError(
                f'Invalid subject_params={subject_params!r}. Use "stored" or "sample".')
        stored_params = (
            _load_stored_params(n_subjects) if subject_params == 'stored' else None)
        os.makedirs(out_dir, exist_ok=True)

        intensity_values = list(
            float(level) for level in intensities
        ) if intensities is not None else [float(level) for level in STIM_LEVELS]
        normalized_subject_param_modes = _normalize_subject_param_modes(
            subject_param_modes)

        pending_subjects = [
            subject_index
            for subject_index in range(n_subjects)
            if not subject_dataset_exists(out_dir, f'roenne_{subject_index:05d}')
        ]

        worker_count = max(1, min(max((os.cpu_count() or 1) - 2, 1), 8))
        print(
            f'[ROENNE] Generating {n_subjects} subjects × {len(intensity_values)} intensities '
            f'with {worker_count} MATLAB workers, subject parameters: '
            + (str(STORED_PARAMS_CSV) if stored_params is not None else f'sampled (seed {SEED})')
        )

        if pending_subjects:
            worker_config = {
                'intensities': intensity_values,
                'stimulus_families': tuple(stimulus_families),
                'flow_range': flow_range,
                'fhigh_range': fhigh_range,
                'num_cf_range': num_cf_range,
                'ur_delay_ms_range': ur_delay_ms_range,
                'cohc_range': cohc_range,
                'cihc_range': cihc_range,
                'spont_range': spont_range,
                'subject_param_modes': normalized_subject_param_modes,
                'source_fs': self.source_fs,
                'label_criterion': label_criterion,
                'db_threshold': db_threshold,
                'amplitude_threshold': amplitude_threshold,
                'seed': SEED,
            }
            tasks = [
                (subject_index, worker_config if stored_params is None
                 else dict(worker_config, stored_params=stored_params[subject_index]))
                for subject_index in pending_subjects
            ]
            with multiprocessing.Pool(processes=worker_count, initializer=init_worker) as pool:
                for result in tqdm(
                    pool.imap_unordered(simulate_subject, tasks),
                    total=len(tasks),
                    desc='Simulating Subjects',
                ):
                    if result is None:
                        continue

                    subject_id, subject_waveforms, subject_records, subject_an_populations = result

                    ordered_records = [
                        {column: record[column] for column in _LABEL_COLUMNS}
                        for record in subject_records
                    ]
                    save_subject_dataset(
                        out_dir,
                        subject_id,
                        subject_waveforms,
                        ordered_records,
                        None,
                        subject_an_populations,
                    )
        else:
            print(
                '[ROENNE] All requested subject datasets already exist; skipping generation.')

        if not any(
            subject_dataset_exists(out_dir, f'roenne_{subject_index:05d}')
            for subject_index in range(n_subjects)
        ):
            print('[ROENNE] No data generated.')
            return

        waveforms_np, labels_df = load_generated_dataset(out_dir)
        labels_df = labels_df[_LABEL_COLUMNS]
        present_pct = labels_df['wave_v_present'].mean() * 100
        print(f'[ROENNE] Saved {len(labels_df)} records -> {out_dir}')
        print(
            f'         Shape: {waveforms_np.shape}  dtype: {waveforms_np.dtype}')
        print(f'         Wave-V present: {present_pct:.1f}%')


def generate_dataset(
    n_subjects: int = NUM_SUBJECTS,
    intensities: list[float] | None = None,
    out_dir: str = 'datasets/synthetic_roenne',
    stimulus_families: tuple[str, ...] = STIMULUS_FAMILIES,
    flow_range: tuple[float, float] = FLOW_RANGE,
    fhigh_range: tuple[float, float] = FHIGH_RANGE,
    num_cf_range: tuple[int, int] = NUM_CF_RANGE,
    ur_delay_ms_range: tuple[float, float] = UR_DELAY_MS_RANGE,
    cohc_range: tuple[float, float] = COHC_RANGE,
    cihc_range: tuple[float, float] = CIHC_RANGE,
    spont_range: tuple[float, float] = SPONT_RANGE,
    subject_param_modes: dict[str, str] | None = None,
    subject_params: str = SUBJECT_PARAMS,
    label_criterion: str = LABEL_CRITERION,
    db_threshold: float = DB_THRESHOLD,
    amplitude_threshold: float = ABSENT_LABEL_THRESHOLD,
) -> None:
    RoenneGenerator().generate_dataset(
        n_subjects=n_subjects,
        intensities=intensities,
        out_dir=out_dir,
        stimulus_families=stimulus_families,
        flow_range=flow_range,
        fhigh_range=fhigh_range,
        num_cf_range=num_cf_range,
        ur_delay_ms_range=ur_delay_ms_range,
        cohc_range=cohc_range,
        cihc_range=cihc_range,
        spont_range=spont_range,
        subject_param_modes=subject_param_modes,
        subject_params=subject_params,
        label_criterion=label_criterion,
        db_threshold=db_threshold,
        amplitude_threshold=amplitude_threshold,
    )


def main() -> None:
    generate_dataset()


if __name__ == '__main__':
    main()
