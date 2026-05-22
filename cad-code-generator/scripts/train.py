"""End-to-end training: data loading → model → fit().

Usage:
    python scripts/train.py \\
        --tokenizer-dir cadquery_tokenizer \\
        --checkpoint-dir checkpoints \\
        --epochs 15 \\
        --batch-size 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from cad_codegen.data import build_dataloaders
from cad_codegen.model import build_model
from cad_codegen.tokenizer import load_tokenizer
from cad_codegen.train import fit

SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-dir", default="cadquery_tokenizer")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--cache-dir", default="HUGGINGFACE_CACHE")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Tokenizer ──────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.tokenizer_dir}…")
    tokenizer = load_tokenizer(args.tokenizer_dir)

    # ── Dataset ────────────────────────────────────────────────────────
    print("Loading dataset…")
    ds = load_dataset(
        "CADCODER/GenCAD-Code",
        num_proc=4,
        split=["train", "test"],
        cache_dir=args.cache_dir,
    )
    ds_train, ds_test = ds
    split = ds_train.shuffle(seed=SEED).train_test_split(test_size=0.10, seed=SEED)
    hf_train, hf_val = split["train"], split["test"]
    print(f"  train={len(hf_train):,} | val={len(hf_val):,} | test={len(ds_test):,}")

    # ── Loaders ────────────────────────────────────────────────────────
    train_dl, val_dl, _ = build_dataloaders(
        hf_train, hf_val, ds_test, tokenizer,
        batch_train=args.batch_size,
        batch_eval=args.batch_size // 2,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        max_len=args.max_len,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = build_model(tokenizer).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────
    history = fit(
        model, train_dl, val_dl,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pad_idx=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    # Save loss curves
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Done. Best val loss: {history['best_val']:.4f}")


if __name__ == "__main__":
    main()
