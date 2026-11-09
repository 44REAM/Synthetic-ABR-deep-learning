"""Training of one classifier.

Adam with weight decay, cosine learning-rate schedule, class-weighted BCE (positive class =
wave V absent), validation loss every VAL_CHECK_INTERVAL steps, early stopping, and the
weights with the lowest validation loss kept.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data import ABRDataset
from src.models import NEEDS_INPUT_STATS, bce_loss_weighted, build_model, fit_input_stats

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HPARAMS = {
    'batch_size': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
    'max_train_steps': 5000,
    'val_check_interval': 4,         # validation loss every 4 steps
    'early_stopping_steps': 240,     # stop when the best validation loss is 240 steps (60 checks) old
}
ARCH_HPARAMS: dict[str, dict] = {}   # per-architecture overrides of HPARAMS, e.g. {'cwt_vit': {...}}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(labels: np.ndarray, device: torch.device) -> Tensor:
    """Weight of the positive class (wave V absent): n_present / n_absent, clipped to [0.5, 5]."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(float(np.clip(n_neg / n_pos, 0.5, 5.0)), dtype=torch.float32, device=device)


def hparams_for(arch: str, overrides: dict | None = None) -> dict:
    return {**HPARAMS, **ARCH_HPARAMS.get(arch, {}), **(overrides or {})}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, pos_weight: Tensor, device: torch.device) -> dict:
    """Weighted BCE loss and AUROC on a loader."""
    model.eval()
    total, logits_all, labels_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += bce_loss_weighted(logits, y, pos_weight).item() * x.size(0)
        logits_all.append(logits.squeeze(-1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    logits_arr, labels_arr = np.concatenate(logits_all), np.concatenate(labels_all).astype(int)
    auc = roc_auc_score(labels_arr, logits_arr) if len(np.unique(labels_arr)) > 1 else float('nan')
    return {'loss': total / len(loader.dataset), 'auc': auc}


def train_model(train_ds: Dataset, val_ds: Dataset, train_labels: np.ndarray, arch: str, seed: int,
                device: torch.device = DEVICE, hparams: dict | None = None) -> tuple[torch.nn.Module, dict]:
    """Train one model; returns it with its best weights loaded, and the validation metrics.

    The metrics dict also holds 'best_step' and 'state' (the best state dict).
    """
    hp = hparams_for(arch, hparams)
    # Statistics some architectures need (CWT-ViT: CWT image range), from one pass over the
    # training set as the model will see it; taken before seeding, so training is unaffected.
    input_stats = None
    if arch in NEEDS_INPUT_STATS:
        waves = np.stack([np.asarray(train_ds[i][0]).squeeze(0) for i in range(len(train_ds))])
        input_stats = fit_input_stats(arch, waves)

    set_seed(seed)
    pos_weight = compute_pos_weight(train_labels, device)
    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    if len(train_loader) == 0:
        raise ValueError(f'Training set has only {len(train_ds)} records, fewer than one batch.')

    model = build_model(arch, input_stats=input_stats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['max_train_steps'])

    best_loss, best_state, best_metrics, best_step = float('inf'), None, {}, 0
    train_iter = iter(train_loader)
    for step in range(1, hp['max_train_steps'] + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        model.train()
        x, y = x.to(device), y.to(device)
        loss = bce_loss_weighted(model(x), y, pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % hp['val_check_interval'] == 0 or step == hp['max_train_steps']:
            metrics = evaluate(model, val_loader, pos_weight, device)
            if metrics['loss'] < best_loss:
                best_loss, best_step, best_metrics = metrics['loss'], step, metrics
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if hp['early_stopping_steps'] is not None and step - best_step >= hp['early_stopping_steps']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_metrics = evaluate(model, val_loader, pos_weight, device)
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return model, {**best_metrics, 'best_step': best_step, 'state': best_state}


@torch.no_grad()
def predict_logits(model: torch.nn.Module, records: list[dict], device: torch.device = DEVICE,
                   batch_size: int = 64) -> np.ndarray:
    """Logits for already-normalised records, in order."""
    model.eval()
    loader = DataLoader(ABRDataset(records), batch_size=batch_size, shuffle=False)
    return np.concatenate([model(x.to(device)).squeeze(-1).float().cpu().numpy() for x, _ in loader])
