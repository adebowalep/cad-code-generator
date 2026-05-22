# Changes from the original notebook

This document describes how the original `good_luck.ipynb` was refactored.

## Structural changes

| Original notebook                       | Refactored location                        |
| --------------------------------------- | ------------------------------------------ |
| Cells 14–22 (Dataset, transforms)       | `src/cad_codegen/data.py`                  |
| Cells 15–17 (Tokenizer training)        | `src/cad_codegen/tokenizer.py`             |
| Cells 26–27 (Model definition)          | `src/cad_codegen/model.py`                 |
| Cells 29–34 (Training loop)             | `src/cad_codegen/train.py`                 |
| Cells 36–37 (Decoding)                  | `src/cad_codegen/decode.py`                |
| Cells 38–39 (Utils)                     | `src/cad_codegen/utils.py`                 |
| Cells 41–42 (IoU + VSR evaluators)      | `src/cad_codegen/eval.py`                  |
| Cell 43 (Checkpoint loading)            | `src/cad_codegen/inference.py`             |
| Cells 45, 48–49 (Eval runner)           | `scripts/evaluate.py`                      |

## Functional improvements

- **`ModelConfig` dataclass**: hyperparameters are now grouped and passed
  explicitly into the model rather than relying on module-level globals.
  This makes swapping encoders/decoders later much cleaner.
- **`build_model(tokenizer, **overrides)`**: convenience factory that derives
  `vocab_size`/`pad_idx` from the tokenizer, so they can't drift.
- **`make_collate_fn(pad_idx)`**: closes over pad_idx instead of relying on a
  global, so the collate is reusable across tokenizers.
- **`fit()`**: full training driver that also saves a `best.pt` based on
  validation loss (the original notebook saved every epoch but no "best").
- **Gradient clipping**: added by default (`grad_clip=1.0`) — helps stability.
- **Step-level logging during training**: prints loss every 50 batches.
- **No more notebook globals**: every function takes explicit args.

## Bug fixes / cleanups

- Original `val_loss_only` had an issue where AMP was applied unconditionally,
  which would crash on CPU. `validate()` now uses `contextlib.nullcontext()`
  on CPU.
- Duplicate `decode_beam` definition (cells 37 and 47 in the notebook) is now
  a single canonical implementation.
- Two near-identical evaluation cells (41 and 42) consolidated into one
  `eval.py` module.
- `os.environ["CADQUERY_LOG_LEVEL"]` is set once at module load instead of in
  multiple cells.

## What was NOT changed

- The model architecture (ResNet-18 + 4-layer Transformer, embed_dim=512).
- Hyperparameters (lr=3e-4, weight_decay=1e-3, batch=128, 15 epochs).
- The evaluation methodology (VSR + best-of-4 principal-axis IoU at pitch=0.05).
- The BPE tokenizer setup (16k vocab, byte-level).

These are deliberately preserved so the refactor reproduces the original
results before we move on to the encoder/decoder upgrades (next milestone).
