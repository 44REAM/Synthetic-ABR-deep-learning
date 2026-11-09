"""Classifier architectures for single-channel ABR waveforms (input (batch, 1, 300), one logit).

EEGNet, ShallowConvNet, DeepConvNet, EEGConformer and ATCNet come from braindecode with their
default settings; CWT-ViT is the reimplementation in src/cwt_vit.py.

Architectures that need statistics of their training data (CWT-ViT: the min-max range of
the CWT images) get them from `fit_input_stats` before the model is built.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from braindecode.modules.convolution import CombinedConv

# Suppress expected braindecode auto-scaling warnings for 300-sample ABR inputs
warnings.filterwarnings('ignore', message='.*n_times.*is smaller than the minimum required.*',
                        category=UserWarning)


def _combined_conv_forward(self, x):
    """braindecode's CombinedConv with squeeze(2) so that a single input channel works."""
    combined_weight = (
        (self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3))
        .sum(0)
        .unsqueeze(1)
    )
    calculated_bias = None
    if self.bias_time:
        time_bias = self.conv_time.bias
        calculated_bias = (
            self.conv_spat.weight.squeeze(2).sum(-1)   # squeeze(2), not squeeze()
            .mm(time_bias.unsqueeze(-1)).squeeze()
        )
    if self.bias_spat:
        spat_bias = self.conv_spat.bias
        calculated_bias = spat_bias if calculated_bias is None else calculated_bias + spat_bias
    return F.conv2d(x, weight=combined_weight, bias=calculated_bias, stride=(1, 1))


CombinedConv.forward = _combined_conv_forward  # apply once, before building models

SAMPLE_RATE = 20_000
ARCHS = ('eegnet', 'shallow', 'deep4', 'eegconformer', 'atcnet', 'cwt_vit')
NEEDS_INPUT_STATS = {'cwt_vit'}      # architectures for which fit_input_stats returns something


def fit_input_stats(arch: str, train_waveforms: np.ndarray) -> dict | None:
    """Statistics an architecture needs from its training waveforms (n, 300); None if none."""
    if arch == 'cwt_vit':
        from src.cwt_vit import cwt_range
        cwt_min, cwt_max = cwt_range(train_waveforms.astype(np.float32))
        return {'cwt_min': cwt_min, 'cwt_max': cwt_max}
    return None


def build_model(arch: str, n_times: int = 300, input_stats: dict | None = None) -> nn.Module:
    """Untrained model for (batch, 1, n_times) input and (batch, 1) output."""
    if arch == 'eegnet':
        from braindecode.models import EEGNet
        return EEGNet(n_chans=1, n_outputs=1, n_times=n_times, sfreq=SAMPLE_RATE)
    if arch == 'shallow':
        from braindecode.models import ShallowFBCSPNet
        return ShallowFBCSPNet(n_chans=1, n_outputs=1, n_times=n_times, sfreq=SAMPLE_RATE)
    if arch == 'deep4':
        from braindecode.models import Deep4Net
        return Deep4Net(n_chans=1, n_outputs=1, n_times=n_times, sfreq=SAMPLE_RATE)
    if arch == 'eegconformer':
        from braindecode.models import EEGConformer
        # num_heads must divide n_filters_time (40 / 10 = 4)
        return EEGConformer(n_chans=1, n_outputs=1, n_times=n_times, n_filters_time=40, num_heads=10,
                            sfreq=SAMPLE_RATE)
    if arch == 'atcnet':
        from braindecode.models import ATCNet
        return ATCNet(n_chans=1, n_outputs=1, n_times=n_times, sfreq=SAMPLE_RATE)
    if arch == 'cwt_vit':
        from src.cwt_vit import CWTViT
        model = CWTViT(num_outputs=1)
        if input_stats is not None:
            model.set_cwt_range(input_stats['cwt_min'], input_stats['cwt_max'])
        return model
    raise ValueError(f'Unknown architecture {arch!r}; choose from {ARCHS}')


def bce_loss_weighted(logit: Tensor, target: Tensor, pos_weight: Tensor) -> Tensor:
    """Binary cross-entropy with logits, positive (wave V absent) class weighted by pos_weight."""
    return nn.functional.binary_cross_entropy_with_logits(logit.squeeze(-1), target, pos_weight=pos_weight)
