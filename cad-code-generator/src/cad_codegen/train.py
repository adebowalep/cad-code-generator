"""Training and validation loops with teacher forcing.

The model is trained with cross-entropy over the next-token distribution.
Padding tokens are masked out via ``ignore_index=pad_idx``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    vocab_size: int,
    grad_clip: float | None = 1.0,
    log_every: int = 50,
) -> float:
    """Run one training epoch with teacher forcing.

    Returns the average loss over the epoch.
    """
    model.train()
    running = 0.0
    n_batches = len(loader)

    for step, (imgs, tgt_full, _meta) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        tgt_full = tgt_full.to(device, non_blocking=True)

        # Teacher forcing: decoder input is BOS..t-1, target is shifted by one.
        tgt_in, tgt_out = tgt_full[:, :-1], tgt_full[:, 1:]
        logits = model(imgs, tgt_in)
        loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item()
        if step % log_every == 0:
            print(f"  step {step:>5}/{n_batches} | loss {loss.item():.4f}")

    return running / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    vocab_size: int,
) -> float:
    """Compute average teacher-forced validation loss (with AMP on CUDA)."""
    model.eval()
    running = 0.0
    amp_ctx = (
        torch.amp.autocast(device_type="cuda")
        if device == "cuda" else contextlib.nullcontext()
    )
    with amp_ctx:
        for imgs, tgt_full, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            tgt_full = tgt_full.to(device, non_blocking=True)
            tgt_in, tgt_out = tgt_full[:, :-1], tgt_full[:, 1:]
            logits = model(imgs, tgt_in)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            running += loss.item()
    return running / len(loader)


def fit(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    n_epochs: int = 15,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    pad_idx: int,
    vocab_size: int,
    checkpoint_dir: str | Path = "checkpoints",
    device: str = "cuda",
) -> dict:
    """Full training driver. Saves a checkpoint after every epoch.

    Returns a dict with ``train_curve`` and ``val_curve`` lists for plotting.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_curve, val_curve = [], []
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        print(f"\n── Epoch {epoch:02d}/{n_epochs} ──")
        tr = train_one_epoch(model, train_dl, optimizer, criterion, device, vocab_size)
        va = validate(model, val_dl, criterion, device, vocab_size)
        train_curve.append(tr)
        val_curve.append(va)
        print(f"[{epoch:02d}] train {tr:.4f} | val {va:.4f}")

        ckpt_path = checkpoint_dir / f"ckpt_e{epoch:02d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  ✓ new best — saved to {checkpoint_dir / 'best.pt'}")

    return {"train_curve": train_curve, "val_curve": val_curve, "best_val": best_val}
