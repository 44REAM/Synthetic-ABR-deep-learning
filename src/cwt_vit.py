"""CWT-ViT for ABR detection, reimplemented from the paper (no code was released):

  Liu Y, Xiang L, Li Q, et al. An interpretable deep learning framework based on
  time-frequency analysis for automated auditory brainstem response detection.
  Biomed Signal Process Control 2026;112:108609. doi:10.1016/j.bspc.2025.108609

The model takes the project's input, a (batch, 1, 300) waveform (20 kHz, 15 ms after
stimulus onset, subject-level min-max normalised; PhysioNet per frequency), so it plugs into train_model() of
src/train.py like the braindecode architectures. Inside the model:

  1. keep 0-12 ms at the project's 20 kHz -> 240 samples
     DEVIATION: the paper used 30 kHz recordings (360 points); the waveforms here are not
     resampled again, so the CWT rows span 167 Hz-5 kHz instead of 250 Hz-7.5 kHz
  2. CWT with the Mexican hat wavelet, scales 1..30 -> 30 x 240 image   (paper, Section 3.2, Table 2)
  3. min-max scaling with the global min/max of the training-set CWT images (paper, Section 3.3)
  4. ViT: 10 x 10 patches (72), embedding 200, 4 heads of 50, MLP 300, 14 pre-LN encoder
     layers, dropout 0.1, [class] token + learnable positional embedding, LN, FC   (paper, Section 3.4, Table 2)

Points marked ASSUMPTION are not specified in the paper.
"""
import numpy as np
import pywt
import timm
import torch
from torch import nn

# ==============================================================================
# CONFIGURATION (paper values unless marked ASSUMPTION)
# ==============================================================================
INPUT_RATE_HZ = 20_000          # project pipeline; DEVIATION: the paper's data were at 30 kHz
CWT_WINDOW_MS = 12.0            # paper: 0-12 ms
CWT_WAVELET = 'mexh'            # paper: Mexican hat (Eq. 3)
CWT_SCALES = np.arange(1, 31)   # paper: "scale factor for CWT" = 30 and image height 30.
                                # ASSUMPTION: integer scales 1..30.
CWT_VALUE = 'coef'              # ASSUMPTION: signed coefficients ('coef'); 'abs' = magnitude.
CWT_REMOVE_MEAN = True          # DEVIATION: subtract each waveform's mean before the CWT. The project
                                # min-max normalises waveforms (baseline ~0.3-0.4), and with the CWT's
                                # zero padding that offset becomes edge artefacts. The paper's band-passed
                                # µV recordings have no such offset.

PATCH_SIZE = 10                 # paper: 10 x 10
EMBED_DIM = 200                 # paper
NUM_HEADS = 4                   # paper (head dimension 50 = 200 / 4)
MLP_HIDDEN = 300                # paper
DEPTH = 14                      # paper: 14 encoder layers
DROPOUT = 0.1                   # paper: 0.1. ASSUMPTION: applied after the positional embedding
                                # and in the MLP/projection (standard ViT), not on attention weights.

N_SAMPLES = int(round(CWT_WINDOW_MS * INPUT_RATE_HZ / 1000))   # 240


def waveform_to_cwt(waves: np.ndarray) -> np.ndarray:
    """(batch, 300) waveforms at 20 kHz -> (batch, 30, 240) CWT images (before min-max)."""
    window = waves[..., :N_SAMPLES]                                                 # 0-12 ms
    if CWT_REMOVE_MEAN:
        window = window - window.mean(axis=-1, keepdims=True)
    coefs, _ = pywt.cwt(window, CWT_SCALES, CWT_WAVELET, sampling_period=1 / INPUT_RATE_HZ, axis=-1)
    coefs = np.moveaxis(coefs, 0, -2)                                               # (batch, scales, time)
    return (np.abs(coefs) if CWT_VALUE == 'abs' else coefs).astype(np.float32)


def cwt_frequencies_hz() -> np.ndarray:
    """Characteristic frequency of each CWT row (paper Eq. 2: f = Fc * fs / a)."""
    return pywt.scale2frequency(CWT_WAVELET, CWT_SCALES) * INPUT_RATE_HZ


class CWTViT(nn.Module):
    """Single logit output (the project's BCE training). The paper uses a 2-class softmax
    with class-weighted cross-entropy, which is the same model up to parametrisation."""

    def __init__(self, num_outputs: int = 1) -> None:
        super().__init__()
        self.vit = timm.models.vision_transformer.VisionTransformer(
            img_size=(len(CWT_SCALES), N_SAMPLES), patch_size=PATCH_SIZE, in_chans=1,
            num_classes=num_outputs, embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
            mlp_ratio=MLP_HIDDEN / EMBED_DIM, qkv_bias=True, class_token=True, global_pool='token',
            pos_drop_rate=DROPOUT, proj_drop_rate=DROPOUT, attn_drop_rate=0.0, drop_rate=0.0,
        )
        # Global min-max of the training-set CWT images (saved with the checkpoint).
        self.register_buffer('cwt_min', torch.tensor(0.0))
        self.register_buffer('cwt_max', torch.tensor(1.0))

    def set_cwt_range(self, cwt_min: float, cwt_max: float) -> None:
        self.cwt_min.fill_(cwt_min)
        self.cwt_max.fill_(cwt_max)

    def images(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 1, 300) waveforms -> (batch, 1, 30, 360) scaled CWT images."""
        img = torch.from_numpy(waveform_to_cwt(x.detach().squeeze(1).cpu().numpy())).to(x.device)
        return ((img - self.cwt_min) / (self.cwt_max - self.cwt_min)).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(self.images(x))


def cwt_range(waves: np.ndarray, batch: int = 512) -> tuple[float, float]:
    """Global min and max of the CWT images of (n, 300) training waveforms."""
    lo, hi = np.inf, -np.inf
    for i in range(0, len(waves), batch):
        c = waveform_to_cwt(waves[i:i + batch])
        lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    return lo, hi
