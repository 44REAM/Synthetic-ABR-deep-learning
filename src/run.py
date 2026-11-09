"""Running many training runs: what each run is, naming, resuming and saving the results.

An experiment script builds a list of RunSpec (usually with `grid`) and calls `run_grid`.
For each run: build the training data (src/data.py), train (src/train.py), save the checkpoint,
and predict every evaluation waveform (the three real test sets and the Southampton tuning
subjects). Analyses then read the predictions with `load_results` and never retrain.

Output folder layout:
  waveforms.csv               one row per evaluation waveform (waveform_id)
  runs/<run>.json             the run's spec, validation loss/AUROC, best step, augmentation used
  predictions/<run>.csv.gz    waveform_id, logit, prob_absent
  checkpoints/<run>.pt        best weights
A run whose checkpoint and predictions exist is skipped, so a grid can be extended or resumed.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import AUGDEV_SET, BASE_SEED, HUMAN_SOURCES, build_training_data, load_eval_sets, waveform_table
from src.models import build_model
from src.train import DEVICE, predict_logits, train_model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Augmentation per training source selected by the tuning experiment; used by augmentation='tuned'.
# Architectures listed in TUNED_AUGMENTATION_FILES were tuned separately; all others use the default.
TUNED_AUGMENTATION_FILE = Path(__file__).resolve().parents[1] / 'results' / 'tune_augmentation' / 'selected.json'
TUNED_AUGMENTATION_FILES = {
    'cwt_vit': Path(__file__).resolve().parents[1] / 'results' / 'tune_augmentation_cwt_vit' / 'selected.json',
}
# Default label rule (experiments/cutpoints.py)
LABEL_RULE = 'cut20'


@dataclass(frozen=True)
class RunSpec:
    sources: tuple[str, ...]          # e.g. ('roenne',), ('roenne', 'verhulst'), ('mixup',), ('wang',)
    n_dev: int                        # development subjects (train + validation, 75 / 25)
    arch: str
    label_rule: str = LABEL_RULE      # see src.data.apply_label_rule
    augmentation: str = 'tuned'       # 'tuned', 'none', or a key of run_grid's augmentation_sets
    cv: int = 0                       # Monte Carlo run: split and training seed BASE_SEED + cv

    @property
    def name(self) -> str:
        return (f'{"+".join(self.sources)}__n{self.n_dev}__{self.arch}__{self.label_rule}'
                f'__{self.augmentation}__cv{self.cv}')


def grid(sources: list[tuple[str, ...]], n_devs: list[int], archs: list[str], label_rules: list[str] = (LABEL_RULE,),
         augmentations: list[str] = ('tuned',), n_runs: int = 30) -> list[RunSpec]:
    """Every combination of the given values, with Monte Carlo runs cv = 0 .. n_runs - 1."""
    return [RunSpec(tuple(s), n, a, lr, aug, cv)
            for s, n, a, lr, aug, cv in itertools.product(sources, n_devs, archs, label_rules, augmentations,
                                                         range(n_runs))]


def resolve_augmentation(name: str, augmentation_sets: dict[str, dict] | None, arch: str | None = None) -> dict[str, dict]:
    """Per-source augmentation settings for an augmentation name ('tuned': for this architecture)."""
    if augmentation_sets and name in augmentation_sets:
        return augmentation_sets[name]
    if name == 'none':
        return {}
    if name == 'tuned':
        path = TUNED_AUGMENTATION_FILES.get(arch, TUNED_AUGMENTATION_FILE)
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: run the augmentation tuning first')
        return json.loads(path.read_text())
    raise ValueError(f'Unknown augmentation {name!r}')


def eval_set_names(spec: RunSpec) -> list[str]:
    """Evaluation sets of a run: all test sets except its own real training dataset, plus the tuning subjects."""
    own = spec.sources[0] if spec.sources[0] in HUMAN_SOURCES else None
    return [s for s in HUMAN_SOURCES if s != own] + [AUGDEV_SET]


def run_grid(specs: list[RunSpec], out_dir: str | Path | None, augmentation_sets: dict[str, dict] | None = None,
             hparams: dict | None = None, device: torch.device = DEVICE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (or skip) every run and predict the evaluation sets.

    out_dir=None keeps everything in memory (nothing is written). Returns the runs table and
    the predictions of the runs done in this call.
    """
    out = Path(out_dir) if out_dir is not None else None
    eval_sets = load_eval_sets()
    waveforms = waveform_table(eval_sets)
    first_id = waveforms.groupby('eval_set')['waveform_id'].min().to_dict()
    if out is not None:
        for sub in ('runs', 'predictions', 'checkpoints'):
            (out / sub).mkdir(parents=True, exist_ok=True)
        waveforms.to_csv(out / 'waveforms.csv', index=False)

    run_rows, pred_frames = [], []
    for i, spec in enumerate(specs, 1):
        files = None if out is None else {k: out / k / f'{spec.name}{ext}' for k, ext in
                                          (('runs', '.json'), ('predictions', '.csv.gz'), ('checkpoints', '.pt'))}
        if files is not None and all(f.exists() for f in files.values()):
            continue

        augmentation = resolve_augmentation(spec.augmentation, augmentation_sets, spec.arch)
        train_ds, val_ds, train_labels = build_training_data(spec.sources, spec.n_dev, spec.cv, spec.label_rule,
                                                             augmentation)
        model, val = train_model(train_ds, val_ds, train_labels, spec.arch, BASE_SEED + spec.cv, device, hparams)

        frames = []
        for s in eval_set_names(spec):
            logits = predict_logits(model, eval_sets[s], device)
            frames.append(pd.DataFrame({'run': spec.name, 'waveform_id': first_id[s] + np.arange(len(logits)),
                                        'logit': logits}))
        preds = pd.concat(frames, ignore_index=True)
        preds['prob_absent'] = 1.0 / (1.0 + np.exp(-preds['logit'].astype(np.float64)))
        row = {'run': spec.name, **asdict(spec), 'sources': '+'.join(spec.sources),
               'val_loss': val['loss'], 'val_auc': val['auc'], 'best_step': val['best_step'],
               'augmentation_settings': augmentation}
        if files is not None:
            torch.save(val['state'], files['checkpoints'])
            preds.to_csv(files['predictions'], index=False, float_format='%.9g', compression='gzip')
            files['runs'].write_text(json.dumps(row, indent=2, default=float))
        run_rows.append(row)
        pred_frames.append(preds)
        print(f'[{i}/{len(specs)}] {spec.name}  val loss {val["loss"]:.4f}  step {val["best_step"]}', flush=True)

    return pd.DataFrame(run_rows), (pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame())


def load_results(out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(runs, predictions, waveforms) of an output folder."""
    out = Path(out_dir)
    runs = pd.DataFrame([json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json'))])
    preds = pd.concat([pd.read_csv(p) for p in sorted((out / 'predictions').glob('*.csv.gz'))], ignore_index=True)
    return runs, preds, pd.read_csv(out / 'waveforms.csv')


def load_model(out_dir: str | Path, run: str, device: torch.device = DEVICE) -> torch.nn.Module:
    """A run's trained model in eval mode."""
    out = Path(out_dir)
    arch = json.loads((out / 'runs' / f'{run}.json').read_text())['arch']
    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(out / 'checkpoints' / f'{run}.pt', map_location=device, weights_only=True),
                          strict=True)
    return model.eval()
