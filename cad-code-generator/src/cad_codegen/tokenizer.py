"""Byte-level BPE tokenizer for CadQuery code.

We train a small BPE tokenizer over the CadQuery code strings in the dataset.
Three special tokens are used: <pad>, <bos>, <eos>.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
    code_iterator: Iterable[str],
    save_dir: str | Path,
    vocab_size: int = 16_000,
    min_frequency: int = 2,
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE tokenizer on raw CadQuery code strings.

    Args:
        code_iterator: Yields raw Python/CadQuery code strings.
        save_dir:      Where to save the tokenizer files.
        vocab_size:    Final vocabulary size (16k is plenty for code).
        min_frequency: Minimum merge frequency.

    Returns:
        A loaded HF `PreTrainedTokenizerFast` ready for use.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw = ByteLevelBPETokenizer()
    raw.train_from_iterator(
        code_iterator,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<bos>", "<eos>"],
    )
    raw.save_model(str(save_dir))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(str(save_dir))
    return tokenizer


def load_tokenizer(save_dir: str | Path) -> PreTrainedTokenizerFast:
    """Load a previously trained tokenizer from disk."""
    return PreTrainedTokenizerFast.from_pretrained(str(save_dir))
