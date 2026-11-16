"""Shared full-batch trainer for the MLP and the graph neural networks (PyTorch)."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn

from metrics import safe_ap


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(data: dict, device=DEVICE) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}


class MLP(nn.Module):
    """Tabular MLP: input projection -> hidden layers -> logit."""

    def __init__(self, d_in: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers, d = [], d_in
        for i in range(n_layers):
            h = max(hidden // (2 ** i), 16)
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, data):  # data: dict with "x"
        return self.net(data["x"]).squeeze(-1)


def train_full_batch(model: nn.Module, data, y: torch.Tensor, train_idx: np.ndarray,
                     val_idx: np.ndarray, *, lr: float, weight_decay: float, pos_weight: float,
                     max_epochs: int, patience: int, minibatch: int | None = None,
                     seed: int = 0) -> tuple[nn.Module, dict]:
    """Train with class-weighted BCE; early-stop on validation PR-AUC; restore best weights.

    If `minibatch` is set (tabular MLP), the forward pass is on mini-batches of training rows;
    graph models use full-batch forward passes over the whole graph.
    """
    set_seed(seed)
    model = model.to(DEVICE)
    data = to_device(data)                                   # CPU or GPU, same code path
    yv = y[val_idx].numpy()
    y = y.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE))
    tr = torch.as_tensor(train_idx, device=DEVICE)
    best, best_state, bad, hist = -1.0, None, 0, []
    gen = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        model.train()
        if minibatch:
            perm = tr[torch.randperm(len(tr), generator=gen).to(DEVICE)]
            for s in range(0, len(perm), minibatch):
                b = perm[s:s + minibatch]
                opt.zero_grad()
                loss = lossf(model({"x": data["x"][b]}), y[b].float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            opt.zero_grad()
            out = model(data)
            loss = lossf(out[tr], y[tr].float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_ap = safe_ap(yv, predict(model, data)[val_idx])
        sched.step(val_ap)
        hist.append(val_ap)
        if val_ap > best + 1e-5:
            best, bad, best_state = val_ap, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_pr_auc": best, "epochs": ep + 1, "best_epoch": int(np.argmax(hist)) + 1}


@torch.no_grad()
def predict(model: nn.Module, data) -> np.ndarray:
    """Probabilities for all rows/nodes (data moved to the model's device if needed)."""
    model.eval()
    dev = next(model.parameters()).device
    return torch.sigmoid(model(to_device(data, dev))).cpu().numpy()
