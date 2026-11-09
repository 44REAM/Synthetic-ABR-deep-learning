"""Training-time signal augmentation and Roenne/Verhulst waveform mixing.

All functions operate on 1-D numpy arrays and draw from numpy's global random state, which
the training code seeds once per run.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def generate_noise(n_samples: int, sigma: float, sample_rate: float) -> np.ndarray:
    """Pink (1/f) noise with standard deviation `sigma`."""
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    white = np.random.randn(len(frequencies)) + 1j * np.random.randn(len(frequencies))
    frequencies[0] = 1.0
    pink = np.fft.irfft(white / np.sqrt(frequencies), n=n_samples).real
    pink = pink / (pink.std() + 1e-12) * sigma
    return pink.astype(np.float32)


def shifted_augment(waveform: np.ndarray, max_shift: int) -> np.ndarray:
    """Shift right by a random integer in [0, max_shift] samples, padding with the first value."""
    if max_shift <= 0:
        return waveform.copy()
    shift = np.random.randint(0, max_shift + 1)
    if shift == 0:
        return waveform.copy()

    shifted = np.empty_like(waveform)
    if shift < len(waveform):
        shifted[shift:] = waveform[:-shift]
        shifted[:shift] = waveform[0]
    else:
        shifted[:] = waveform[0]
    return shifted


def add_randn_noise(waveform: np.ndarray, max_sigma: float) -> np.ndarray:
    """Gaussian noise with a standard deviation drawn uniformly from (0, max_sigma)."""
    if max_sigma <= 0:
        return waveform.copy()
    sigma = np.random.uniform(0.0, max_sigma)
    noise = np.random.randn(*waveform.shape).astype(np.float32) * sigma
    return waveform + noise


def add_pink_noise(waveform: np.ndarray, max_sigma: float, sample_rate: float) -> np.ndarray:
    """Pink noise with a standard deviation drawn uniformly from (0, max_sigma)."""
    if max_sigma <= 0:
        return waveform.copy()
    sigma = np.random.uniform(0.0, max_sigma)
    return waveform + generate_noise(len(waveform), sigma, sample_rate)


def add_uniform_noise(waveform: np.ndarray, max_amplitude: float) -> np.ndarray:
    """Uniform noise U(-a, a) with a drawn uniformly from (0, max_amplitude)."""
    if max_amplitude <= 0:
        return waveform.copy()
    limit = np.random.uniform(0.0, max_amplitude)
    noise = np.random.uniform(-limit, limit, size=waveform.shape).astype(np.float32)
    return waveform + noise


def dc_shift_augment(waveform: np.ndarray, max_shift: float) -> np.ndarray:
    """Add a DC offset drawn uniformly from (-max_shift, max_shift)."""
    if max_shift <= 0:
        return waveform.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    return (waveform + shift).astype(np.float32)


def mixup_waveforms(wf_1: np.ndarray, wf_2: np.ndarray, alpha: float) -> np.ndarray:
    """Time-domain mixup: alpha * wf_1 + (1 - alpha) * wf_2."""
    return (alpha * wf_1 + (1.0 - alpha) * wf_2).astype(np.float32)


def cutmix_waveforms(wf_1: np.ndarray, wf_2: np.ndarray, alpha: float,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Replace a random segment of wf_1, of length (1 - alpha) of the waveform, with wf_2.

    The segment lies inside the waveform, so central samples are replaced more often than the
    ends; callers that mix two sources choose wf_1 at random to balance this. rng draws the
    segment start (None: the global numpy random state). Returns the mixed waveform and the
    fraction of wf_1 kept.
    """
    length = len(wf_1)
    if length != len(wf_2):
        raise ValueError('Waveforms must have the same length.')

    width = int(round(length * (1.0 - alpha)))
    if width <= 0:
        return wf_1.copy(), 1.0
    if width >= length:
        return wf_2.copy(), 0.0

    start = int(rng.integers(0, length - width + 1)) if rng is not None else np.random.randint(0, length - width + 1)
    mixed = wf_1.copy()
    mixed[start:start + width] = wf_2[start:start + width]
    return mixed.astype(np.float32), float(1.0 - width / length)


def make_transform(config: dict | None, sample_rate: float) -> Callable[[np.ndarray], np.ndarray] | None:
    """Augmentation for one training source.

    `config` keys (any may be missing or None): 'shift_limit_ms', 'pink_noise_sigma',
    'randn_noise_sigma' (Gaussian white noise), 'uniform_noise_amplitude', 'dc_shift_max'.
    Applied in that order. Returns None when nothing is enabled.
    ('noisy_validation' is read by src.data.build_training_data, not here.)
    """
    config = config or {}
    shift_limit = config.get('shift_limit_ms')
    pink_sigma = config.get('pink_noise_sigma')
    randn_sigma = config.get('randn_noise_sigma')
    uniform_max = config.get('uniform_noise_amplitude')
    dc_max = config.get('dc_shift_max')
    if all(v is None for v in (shift_limit, pink_sigma, randn_sigma, uniform_max, dc_max)):
        return None

    def transform(wf: np.ndarray) -> np.ndarray:
        if shift_limit is not None:
            wf = shifted_augment(wf, int(round(shift_limit * sample_rate / 1000.0)))
        if pink_sigma is not None:
            wf = add_pink_noise(wf, pink_sigma, sample_rate)
        if randn_sigma is not None:
            wf = add_randn_noise(wf, randn_sigma)
        if uniform_max is not None:
            wf = add_uniform_noise(wf, uniform_max)
        if dc_max is not None:
            wf = dc_shift_augment(wf, dc_max)
        return wf

    return transform
