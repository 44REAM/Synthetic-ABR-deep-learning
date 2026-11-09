"""Integrated Gradients of CWT-ViT on its own input, the CWT image.

The runs of experiments/multi_arch.py for CWT-ViT (every training source, all runs) on the test
sets each run is evaluated on, as analysis/attribution.py, which attributes the logit to the
waveform. Here the logit (positive class = wave V absent) is
attributed to the scaled 30 x 240 CWT image the ViT sees: positive values push the run towards
'absent', negative towards 'present'. Captum's IntegratedGradients, N_STEPS steps, from the
image of a flat waveform. The model removes each waveform's mean before the CWT, so a flat
waveform at any level (the subject_mean baseline of the waveform attribution included) gives
the same image, a constant at the scaled value of a zero coefficient.
Per (training source, test set): the attribution averaged over the waveforms of each expert
label and then over the runs (with its SD over runs), and the mean scaled image with its
baseline constant, both with the first run's scaling (curves); the
attribution of each waveform averaged over the runs (attributions); per run, the share of the
summed |attribution| of the present-labelled waveforms per time window and per frequency band,
and the time and frequency of its peak (shares); and P(absent) of each run on the baseline
image with the convergence delta (baseline_outputs).
Writes cwt_vit_image_curves.npz, cwt_vit_image_attributions.npz, cwt_vit_image_shares.csv,
cwt_vit_image_baseline_outputs.csv and summary_image.txt into results/attribution/; skipped
when the shares file exists. Figures: figures/attribution.py.

Run from the repository root (about an hour on the GPU):
  python analysis/attribution_image.py
"""

from __future__ import annotations

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

from analysis.attribution import (BATCH_SIZE, MAX_RUNS_PER_SOURCE, N_STEPS, NETWORK_DIR, OUT_DIR, SOURCES, TEST_SETS,
                                  spec_of)
from src.cwt_vit import INPUT_RATE_HZ, cwt_frequencies_hz, waveform_to_cwt
from src.data import load_eval_sets
from src.run import eval_set_names, load_model
from src.train import DEVICE

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ARCH = 'cwt_vit'
INTERNAL_BATCH_SIZE = 256            # images x steps evaluated at once
WINDOWS_MS = {'0-5 ms': (0.0, 5.0), '5-11 ms': (5.0, 11.0), '11-12 ms': (11.0, 12.0)}   # the image spans 0-12 ms
BANDS_HZ = {'<500 Hz': (0.0, 500.0), '500-1000 Hz': (500.0, 1000.0), '1-2 kHz': (1000.0, 2000.0),
            '>2 kHz': (2000.0, np.inf)}


def scaled_images(model: torch.nn.Module, waveforms: np.ndarray) -> np.ndarray:
    """(n, 300) waveforms -> (n, 1, 30, 240) images scaled with the run's own CWT range."""
    img = waveform_to_cwt(waveforms)
    lo, hi = model.cwt_min.item(), model.cwt_max.item()
    return ((img - lo) / (hi - lo)).astype(np.float32)[:, None]


def attribute(model: torch.nn.Module, img: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrated Gradients of the logit with respect to the image, from the baseline image, with
    the convergence delta and P(absent | baseline)."""
    ig = IntegratedGradients(model.vit)
    attrs, deltas, p_base = [], [], []
    for i in range(0, len(img), BATCH_SIZE):
        xt = torch.from_numpy(img[i:i + BATCH_SIZE]).to(DEVICE)
        bt = torch.from_numpy(base[i:i + BATCH_SIZE]).to(DEVICE)
        a, d = ig.attribute(xt, baselines=bt, target=0, n_steps=N_STEPS, internal_batch_size=INTERNAL_BATCH_SIZE,
                            return_convergence_delta=True)
        attrs.append(a.squeeze(1).detach().cpu().numpy())
        deltas.append(d.detach().cpu().numpy())
        with torch.no_grad():
            p_base.append(torch.sigmoid(model.vit(bt)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(attrs), np.concatenate(deltas), np.concatenate(p_base)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT_DIR / f'{ARCH}_image_shares.csv').exists():
        print(f'{ARCH} image attribution: done already, skipped')
        write_summary()
        return
    eval_sets = load_eval_sets()
    inputs = {s: np.stack([r['waveform_proc'] for r in eval_sets[s]]).astype(np.float32) for s in TEST_SETS}
    labels = {s: np.array([int(r['label']) for r in eval_sets[s]]) for s in TEST_SETS}
    runs = [p for p in sorted(NETWORK_DIR.glob('runs/*.json')) if spec_of(p).arch == ARCH]
    specs = {p.stem: spec_of(p) for p in runs}
    by_source: dict[str, list[str]] = {}
    for name, spec in specs.items():
        by_source.setdefault('+'.join(spec.sources), []).append(name)
    n_time = waveform_to_cwt(inputs['wang'][:1]).shape[-1]
    t_ms = np.arange(n_time) / INPUT_RATE_HZ * 1000.0
    freq_hz = cwt_frequencies_hz()
    windows = {w: (t_ms >= lo) & (t_ms < hi) for w, (lo, hi) in WINDOWS_MS.items()}
    bands = {b: (freq_hz >= lo) & (freq_hz < hi) for b, (lo, hi) in BANDS_HZ.items()}

    curves: dict[str, list] = {}          # source|eval_set -> per-run (2, scales, time)
    attr_sum: dict[str, np.ndarray] = {}  # source|eval_set -> sum over runs of per-waveform attributions
    image_mean: dict[str, np.ndarray] = {}
    baseline_value: dict[str, float] = {}
    share_rows, base_rows = [], []
    t0 = time.time()
    n_done = 0
    for source in SOURCES:
        for run in by_source.get(source, [])[:MAX_RUNS_PER_SOURCE]:
            model = load_model(NETWORK_DIR, run, DEVICE)
            for eval_set in [s for s in eval_set_names(specs[run]) if s in TEST_SETS]:
                x, y = inputs[eval_set], labels[eval_set]
                img = scaled_images(model, x)
                base = scaled_images(model, np.zeros_like(x))     # a flat waveform; any level gives this image
                if n_done == 0:
                    print(f'  baseline image constant {base.min():.4f}-{base.max():.4f}', flush=True)
                    with torch.no_grad():
                        diff = (model.vit(torch.from_numpy(img[:BATCH_SIZE]).to(DEVICE))
                                - model(torch.from_numpy(x[:BATCH_SIZE]).unsqueeze(1).to(DEVICE))).abs().max().item()
                    print(f'  images against the model: max |logit difference| {diff:.2e}', flush=True)
                a, delta, p_base = attribute(model, img, base)
                key = f'{source}|{eval_set}'
                curves.setdefault(key, []).append(np.stack([a[y == 0].mean(axis=0), a[y == 1].mean(axis=0)]))
                attr_sum[key] = attr_sum.get(key, 0) + a
                if eval_set not in image_mean:         # the first run's scaling; its baseline constant recorded too
                    image_mean[eval_set] = np.stack([img[y == 0, 0].mean(axis=0), img[y == 1, 0].mean(axis=0)])
                    baseline_value[eval_set] = float(base[0, 0, 0, 0])
                mag = np.abs(a[y == 0]).mean(axis=0)                                  # (scales, time)
                peak = np.unravel_index(mag.argmax(), mag.shape)
                share_rows.append({'arch': ARCH, 'sources': source, 'eval_set': eval_set, 'run': run,
                                   'peak_ms_present': float(t_ms[peak[1]]), 'peak_hz_present': float(freq_hz[peak[0]]),
                                   **{f'share_present {w}': float(mag[:, m].sum() / mag.sum()) for w, m in windows.items()},
                                   **{f'share_present {b}': float(mag[m].sum() / mag.sum()) for b, m in bands.items()}})
                base_rows.append({'arch': ARCH, 'sources': source, 'eval_set': eval_set, 'run': run,
                                  'p_absent_baseline': float(p_base.mean()),
                                  'convergence_delta_abs': float(np.abs(delta).mean())})
            n_done += 1
            if n_done % 10 == 0:
                print(f'  {n_done} runs, {(time.time() - t0) / 60:.1f} min', flush=True)
    n_runs = {key: len(v) for key, v in curves.items()}
    np.savez(OUT_DIR / f'{ARCH}_image_curves.npz', t_ms=t_ms, freq_hz=freq_hz,
             **{f'{key}|mean': np.mean(v, axis=0) for key, v in curves.items()},
             **{f'{key}|sd': np.std(v, axis=0, ddof=1) if len(v) > 1 else np.zeros_like(v[0]) for key, v in curves.items()},
             **{f'{s}|image_mean': m for s, m in image_mean.items()},
             **{f'{s}|baseline_value': np.array(v) for s, v in baseline_value.items()},
             **{f'{key}|n_runs': np.array(n) for key, n in n_runs.items()})
    np.savez(OUT_DIR / f'{ARCH}_image_attributions.npz',
             **{key: (v / n_runs[key]).astype(np.float16) for key, v in attr_sum.items()},
             **{f'{s}|waveform_id_first': np.array(sum(len(eval_sets[k]) for k in list(eval_sets)[:list(eval_sets).index(s)]))
                for s in TEST_SETS})
    pd.DataFrame(share_rows).to_csv(OUT_DIR / f'{ARCH}_image_shares.csv', index=False, float_format='%.6g')
    pd.DataFrame(base_rows).to_csv(OUT_DIR / f'{ARCH}_image_baseline_outputs.csv', index=False, float_format='%.6g')
    print(f'{ARCH} image attribution: {n_done} runs in {(time.time() - t0) / 60:.1f} min', flush=True)
    write_summary()


def write_summary() -> None:
    lines: list[str] = []

    def out(text: str = '') -> None:
        print(text)
        lines.append(text)

    pd.set_option('display.width', 250)
    shares = pd.read_csv(OUT_DIR / f'{ARCH}_image_shares.csv')
    base = pd.read_csv(OUT_DIR / f'{ARCH}_image_baseline_outputs.csv')
    freq_hz = cwt_frequencies_hz()
    rows = {b: int(((freq_hz >= lo) & (freq_hz < hi)).sum()) for b, (lo, hi) in BANDS_HZ.items()}
    out('######## CWT-ViT, attribution on the CWT image (baseline: image of a flat waveform) ########')
    out(f'CWT rows (scales) per frequency band: ' + ', '.join(f'{b} {n}' for b, n in rows.items())
        + f' of {len(freq_hz)}; the image spans 0-12 ms')
    g = base.groupby('sources')
    out('\n=== P(absent) of the runs on the baseline image (mean ± SD over runs and test sets), and the mean '
        '|convergence delta| of the integration ===')
    t = pd.DataFrame({'P(absent | baseline)': g['p_absent_baseline'].agg(lambda v: f'{v.mean():.3f} ± {v.std():.3f}'),
                      '|delta|': g['convergence_delta_abs'].mean().map('{:.4f}'.format)})
    out(t.reindex(list(SOURCES)).rename(index=SOURCES).to_string())
    g = shares.groupby(['sources', 'eval_set'])
    for title, cols, peak in (('per time window and the time of its peak', list(WINDOWS_MS), 'peak_ms_present'),
                              ('per frequency band and the frequency of its peak', list(BANDS_HZ), 'peak_hz_present')):
        t = pd.DataFrame({**{c: g[f'share_present {c}'].agg(lambda v: f'{v.mean():.2f} ± {v.std():.2f}') for c in cols},
                          'peak': g[peak].agg(lambda v: f'{v.mean():.0f} ± {v.std():.0f}' if peak.endswith('hz')
                                              else f'{v.mean():.1f} ± {v.std():.1f}')})
        out(f'\n=== share of the |attribution| of the present-labelled waveforms {title}, mean ± SD over runs ===')
        t = t.unstack('eval_set').reindex(index=list(SOURCES), columns=pd.MultiIndex.from_product(
            [cols + ['peak'], list(TEST_SETS)]))
        out(t.rename(index=SOURCES).rename(columns=TEST_SETS, level=1).fillna('-').to_string())
    (OUT_DIR / 'summary_image.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
