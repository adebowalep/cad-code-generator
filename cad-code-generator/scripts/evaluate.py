"""Evaluate a trained checkpoint on the GenCAD-Code test split.

Reports Valid Syntax Rate (VSR) and mean IoU.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --tokenizer-dir cadquery_tokenizer \\
        --n-samples 200 \\
        --method greedy
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time

import torch
from datasets import load_dataset

from cad_codegen.data import build_dataloaders
from cad_codegen.decode import decode_beam, decode_greedy
from cad_codegen.eval import evaluate_codes
from cad_codegen.model import build_model
from cad_codegen.tokenizer import load_tokenizer
from cad_codegen.utils import fix_floats

SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-dir", default="cadquery_tokenizer")
    p.add_argument("--cache-dir", default="HUGGINGFACE_CACHE")
    p.add_argument("--method", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=None,
                   help="Limit evaluation to N test samples (None = full test set).")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load tokenizer + model ─────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = build_model(tokenizer).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Load test set ──────────────────────────────────────────────────
    ds = load_dataset(
        "CADCODER/GenCAD-Code",
        num_proc=4,
        split=["train", "test"],
        cache_dir=args.cache_dir,
    )
    ds_train, ds_test = ds
    split = ds_train.shuffle(seed=SEED).train_test_split(test_size=0.10, seed=SEED)

    _, _, test_dl = build_dataloaders(
        split["train"], split["test"], ds_test, tokenizer,
        batch_train=args.batch_size, batch_eval=args.batch_size,
        num_workers=2, pin_memory=(device == "cuda"),
        max_len=args.max_len,
    )

    # ── Generate predictions ───────────────────────────────────────────
    print(f"Running {args.method} decoding…")
    t0 = time.time()
    gt_codes, pred_codes = {}, {}
    processed = 0

    with torch.no_grad():
        for imgs, _, meta in test_dl:
            imgs = imgs.to(device)
            if args.method == "greedy":
                preds = decode_greedy(model, imgs, tokenizer, max_len=args.max_len)
            else:
                preds = [
                    decode_beam(model, imgs[i], tokenizer,
                                max_len=args.max_len, beam_size=args.beam_size)
                    for i in range(imgs.shape[0])
                ]
            for i, p_code in enumerate(preds):
                sid = str(meta["ids"][i])
                gt_codes[sid] = meta["code_strings"][i]
                pred_codes[sid] = fix_floats(p_code)
                processed += 1
                if args.n_samples is not None and processed >= args.n_samples:
                    break
            if args.n_samples is not None and processed >= args.n_samples:
                break

    print(f"Generated {processed} predictions in {time.time() - t0:.1f}s")

    # ── Compute metrics ────────────────────────────────────────────────
    print("Computing VSR + IoU (CadQuery execution + voxelization)…")
    with contextlib.redirect_stdout(io.StringIO()):
        results = evaluate_codes(gt_codes, pred_codes, verbose=False)

    print("\n────────── RESULTS ──────────")
    print(f"  Method        : {args.method}")
    print(f"  Samples       : {results['n_evaluated']}")
    print(f"  VSR           : {results['vsr']:.3f}")
    print(f"  Mean IoU      : {results['mean_iou']:.3f}")
    print("─────────────────────────────")


if __name__ == "__main__":
    main()
