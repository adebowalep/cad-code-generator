"""Train the BPE tokenizer on the GenCAD-Code dataset.

Usage:
    python scripts/train_tokenizer.py --save-dir cadquery_tokenizer
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from cad_codegen.tokenizer import train_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", default="cadquery_tokenizer")
    p.add_argument("--vocab-size", type=int, default=16_000)
    p.add_argument("--cache-dir", default="HUGGINGFACE_CACHE")
    args = p.parse_args()

    print("Loading dataset…")
    ds = load_dataset(
        "CADCODER/GenCAD-Code",
        num_proc=4,
        split=["train", "test"],
        cache_dir=args.cache_dir,
    )
    ds_train, ds_test = ds

    def code_iterator():
        for split in (ds_train, ds_test):
            for row in split:
                yield row["cadquery"]

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})…")
    tokenizer = train_tokenizer(
        code_iterator=code_iterator(),
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
    )
    print(f"✓ Saved tokenizer to {Path(args.save_dir).resolve()}")
    print(f"  vocab_size = {tokenizer.vocab_size}")
    print(f"  bos = {tokenizer.bos_token!r} ({tokenizer.bos_token_id})")
    print(f"  eos = {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    print(f"  pad = {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")


if __name__ == "__main__":
    main()
