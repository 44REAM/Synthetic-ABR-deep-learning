"""Integrated Gradients of the networks on the real test waveforms.

The runs of experiments/multi_arch.py for ARCHS (every training source, all runs) on the test
sets each run is evaluated on (src.run.eval_set_names, without the tuning subjects), as the
networks see them (normalised). Attribution of the logit (positive class = wave V absent):
positive values push the run towards 'absent', negative towards 'present'. Captum's
IntegratedGradients, N_STEPS steps, from the baseline
  subject_mean  a flat line at the mean of all the subject's normalised recordings (PhysioNet:
                of that tone-burst frequency, the normalisation group)
Per (architecture, training source, test set, baseline): the attribution averaged over the
waveforms of each expert label and then over the runs (with its SD over runs) and the mean
waveform (curves); the attribution of each waveform averaged over the runs (attributions);
per run, the share of the summed |attribution| of the present-labelled waveforms in WINDOWS_MS
(5-11 ms is the wave V window of Berninger et al. 2014, Ear Hear 35:318-329) and the time of the largest |attribution|
(window_shares, summarised as mean ± SD over runs); and the output of each run on the
baselines themselves, P(absent) (baseline_outputs), with the convergence delta of the integration.
Writes <arch>_curves.npz, <arch>_attributions.npz, <arch>_window_shares.csv,
<arch>_baseline_outputs.csv and summary.txt into results/attribution/; an architecture whose
files exist is skipped. Figures: figures/attribution.py.

Run from the repository root (EEGConformer about 45 min on the GPU, CWT-ViT longer):
  python analysis/attribution.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path when script is run directly
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import torch

from src.cwt_vit import waveform_to_cwt
from src.data import POST_LEN, TARGET_RATE, load_eval_sets
from src.run import RunSpec, eval_set_names, load_model
from src.train import DEVICE

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NETWORK_DIR = REPOSITORY_ROOT / 'results' / 'multi_arch'
OUT_DIR = REPOSITORY_ROOT / 'results' / 'attribution'
# Architectures (display names): the paper's waveform architecture and the image-based one
ARCHS = {'eegconformer': 'EEGConformer', 'cwt_vit': 'CWT-ViT'}
SOURCES = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet',
           'roenne': 'Roenne', 'verhulst': 'Verhulst', 'roenne+verhulst': 'Roenne+Verhulst',
           'mixup': 'MixUp', 'cutmix': 'CutMix',
           'gaussian': 'Control: Gaussian'}
TEST_SETS = {'wang': 'Mendeley', 'southampton': 'Southampton', 'physionet': 'PhysioNet'}
BASELINES = ['subject_mean']
N_STEPS = 50
BATCH_SIZE = 64
INTERNAL_BATCH_SIZE = {'eegconformer': 3200, 'cwt_vit': 256}   # inputs x steps evaluated at once
WINDOWS_MS = {'0-5 ms': (0.0, 5.0), '5-11 ms': (5.0, 11.0), '11-15 ms': (11.0, 15.0)}
MAX_RUNS_PER_SOURCE: int | None = None   # for smoke tests


class DifferentiableCWTViT(torch.nn.Module):
    """CWT-ViT with its CWT computed in torch, so the logit can be differentiated with respect to
    the waveform. The model's CWT (src.cwt_vit.waveform_to_cwt: first 12 ms, mean removed,
    Mexican-hat CWT by pywt) is linear, so it is the sum of the images of the unit impulses,
    computed once with the model's own function; the scaling and the ViT are the model's own."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        basis = waveform_to_cwt(np.eye(POST_LEN, dtype=np.float32))                  # (300, scales, time)
        self.register_buffer('basis', torch.from_numpy(basis).to(model.cwt_min.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:                             # x: (batch, 1, 300)
        img = torch.einsum('bi,ist->bst', x.squeeze(1), self.basis)
        img = ((img - self.model.cwt_min) / (self.model.cwt_max - self.model.cwt_min)).unsqueeze(1)
        return self.model.vit(img)


def differentiable(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    return DifferentiableCWTViT(model).eval() if arch == 'cwt_vit' else model


def spec_of(run_json: Path) -> RunSpec:
    d = json.loads(run_json.read_text())
    return RunSpec(tuple(d['sources'].split('+')), d['n_dev'], d['arch'], d['label_rule'], d['augmentation'], d['cv'])


def baseline_of(name: str, records: list[dict], waveforms: np.ndarray) -> np.ndarray:
    """Baseline waveforms (same shape as waveforms) for the records of one evaluation set."""
    if name != 'subject_mean':
        raise ValueError(f'Unknown baseline {name!r}')
    keys =[(r['subject'], r.get('freq_khz')) for r in records]
    means = {k: waveforms[[i for i, kk in enumerate(keys) if kk == k]].mean() for k in set(keys)}
    return np.stack([np.full(waveforms.shape[1], means[k], dtype=np.float32) for k in keys])


def attribute(model: torch.nn.Module, x: np.ndarray, b: np.ndarray, internal_batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrated Gradients of the logit from baseline b, with the convergence delta and P(absent | b)."""
    ig = IntegratedGradients(model)
    attrs, deltas, p_base = [], [], []
    for i in range(0, len(x), BATCH_SIZE):
        xt = torch.from_numpy(x[i:i + BATCH_SIZE]).unsqueeze(1).to(DEVICE)
        bt = torch.from_numpy(b[i:i + BATCH_SIZE]).unsqueeze(1).to(DEVICE)
        a, d = ig.attribute(xt, baselines=bt, target=0, n_steps=N_STEPS, internal_batch_size=internal_batch,
                            return_convergence_delta=True)
        attrs.append(a.squeeze(1).detach().cpu().numpy())
        deltas.append(d.detach().cpu().numpy())
        with torch.no_grad():
            p_base.append(torch.sigmoid(model(bt)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(attrs), np.concatenate(deltas), np.concatenate(p_base)


def run_arch(arch: str, eval_sets: dict[str, list[dict]], inputs: dict[str, np.ndarray],
             labels: dict[str, np.ndarray], baselines: dict[tuple[str, str], np.ndarray]) -> None:
    runs = [p for p in sorted(NETWORK_DIR.glob('runs/*.json')) if spec_of(p).arch == arch]
    specs = {p.stem: spec_of(p) for p in runs}
    by_source: dict[str, list[str]] = {}
    for name, spec in specs.items():
        by_source.setdefault('+'.join(spec.sources), []).append(name)
    t_ms = np.arange(inputs['wang'].shape[1]) / TARGET_RATE * 1000.0
    windows = {w: (t_ms >= lo) & (t_ms < hi) for w, (lo, hi) in WINDOWS_MS.items()}

    curves: dict[str, list] = {}          # key -> per-run arrays
    attr_sum: dict[str, np.ndarray] = {}  # key -> sum over runs of per-waveform attributions
    share_rows, base_rows = [], []
    t0 = time.time()
    n_done = 0
    for source in SOURCES:
        names = by_source.get(source, [])[:MAX_RUNS_PER_SOURCE]
        for run in names:
            model = differentiable(load_model(NETWORK_DIR, run, DEVICE), arch)
            if n_done == 0 and isinstance(model, DifferentiableCWTViT):
                with torch.no_grad():
                    xt = torch.from_numpy(inputs['wang']).unsqueeze(1).to(DEVICE)
                    diff = (model(xt) - model.model(xt)).abs().max().item()
                print(f'  [{arch}] differentiable CWT against the model: max |logit difference| {diff:.2e}', flush=True)
            for eval_set in [s for s in eval_set_names(specs[run]) if s in TEST_SETS]:
                x, y = inputs[eval_set], labels[eval_set]
                for baseline in BASELINES:
                    a, delta, p_base = attribute(model, x, baselines[(eval_set, baseline)], INTERNAL_BATCH_SIZE[arch])
                    key = f'{source}|{eval_set}|{baseline}'
                    curves.setdefault(key, []).append(np.stack([a[y == 0].mean(axis=0), a[y == 1].mean(axis=0)]))
                    attr_sum[key] = attr_sum.get(key, 0) + a
                    mag = np.abs(a[y == 0]).mean(axis=0)
                    mag_absent = np.abs(a[y == 1]).mean(axis=0)
                    share_rows.append({'arch': arch, 'sources': source, 'eval_set': eval_set, 'baseline': baseline,
                                       'run': run, 'peak_ms_present': float(t_ms[mag.argmax()]),
                                       **{f'share_present {w}': float(mag[m].sum() / mag.sum()) for w, m in windows.items()},
                                       **{f'share_absent {w}': float(mag_absent[m].sum() / mag_absent.sum())
                                          for w, m in windows.items()}})
                    base_rows.append({'arch': arch, 'sources': source, 'eval_set': eval_set, 'baseline': baseline,
                                      'run': run, 'p_absent_baseline': float(p_base.mean()),
                                      'convergence_delta_abs': float(np.abs(delta).mean())})
            n_done += 1
            if n_done % 10 == 0:
                print(f'  [{arch}] {n_done} runs, {(time.time() - t0) / 60:.1f} min', flush=True)
    n_runs = {key: len(v) for key, v in curves.items()}
    np.savez(OUT_DIR / f'{arch}_curves.npz', t_ms=t_ms,
             **{f'{key}|mean': np.mean(v, axis=0) for key, v in curves.items()},
             **{f'{key}|sd': np.std(v, axis=0, ddof=1) if len(v) > 1 else np.zeros_like(v[0]) for key, v in curves.items()},
             **{f'{s}|waveform_mean': np.stack([inputs[s][labels[s] == 0].mean(axis=0), inputs[s][labels[s] == 1].mean(axis=0)])
                for s in TEST_SETS},
             **{f'{key}|n_runs': np.array(n) for key, n in n_runs.items()})
    np.savez(OUT_DIR / f'{arch}_attributions.npz', **{key: v / n_runs[key] for key, v in attr_sum.items()},
             **{f'{s}|waveform_id_first': np.array(sum(len(eval_sets[k]) for k in list(eval_sets)[:list(eval_sets).index(s)]))
                for s in TEST_SETS})
    pd.DataFrame(share_rows).to_csv(OUT_DIR / f'{arch}_window_shares.csv', index=False, float_format='%.6g')
    pd.DataFrame(base_rows).to_csv(OUT_DIR / f'{arch}_baseline_outputs.csv', index=False, float_format='%.6g')
    print(f'{arch}: {n_done} runs in {(time.time() - t0) / 60:.1f} min', flush=True)


def write_summary() -> None:
    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    for arch, arch_name in ARCHS.items():
        if not (OUT_DIR / f'{arch}_window_shares.csv').exists():
            continue
        shares = pd.read_csv(OUT_DIR / f'{arch}_window_shares.csv')
        base = pd.read_csv(OUT_DIR / f'{arch}_baseline_outputs.csv')
        out(f'\n######## {arch_name} ########')
        g = base.groupby(['sources', 'baseline'])
        out(f'\n=== P(absent) of the runs on the baseline itself (mean ± SD over runs and test sets), and the mean '
            f'|convergence delta| of the integration ===')
        t = pd.DataFrame({'P(absent | baseline)': g['p_absent_baseline'].agg(lambda v: f'{v.mean():.3f} ± {v.std():.3f}'),
                          '|delta|': g['convergence_delta_abs'].mean().map('{:.4f}'.format)}).unstack('baseline')
        out(t.reindex(list(SOURCES)).rename(index=SOURCES).to_string())
        for baseline in BASELINES:
            s = shares[shares['baseline'] == baseline]
            g = s.groupby(['sources', 'eval_set'])
            cols = {f'share_present {w}': w for w in WINDOWS_MS}
            t = pd.DataFrame({**{w: g[c].agg(lambda v: f'{v.mean():.2f} ± {v.std():.2f}') for c, w in cols.items()},
                              'peak ms': g['peak_ms_present'].agg(lambda v: f'{v.mean():.1f} ± {v.std():.1f}')})
            out(f'\n=== baseline {baseline}: share of the |attribution| of the present-labelled waveforms per window '
                f'and the time of its peak, mean ± SD over runs ===')
            t = t.unstack('eval_set').reindex(index=list(SOURCES), columns=pd.MultiIndex.from_product(
                [list(cols.values()) + ['peak ms'], list(TEST_SETS)]))
            out(t.rename(index=SOURCES).rename(columns=TEST_SETS, level=1).fillna('-').to_string())
    (OUT_DIR / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_sets = load_eval_sets()
    inputs = {s: np.stack([r['waveform_proc'] for r in eval_sets[s]]).astype(np.float32) for s in TEST_SETS}
    labels = {s: np.array([int(r['label']) for r in eval_sets[s]]) for s in TEST_SETS}
    baselines = {(s, b): baseline_of(b, eval_sets[s], inputs[s]) for s in TEST_SETS for b in BASELINES}
    for arch in ARCHS:
        if (OUT_DIR / f'{arch}_window_shares.csv').exists():
            print(f'{arch}: done already, skipped')
            continue
        run_arch(arch, eval_sets, inputs, labels, baselines)
    write_summary()


if __name__ == '__main__':
    main()
