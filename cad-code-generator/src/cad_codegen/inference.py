"""Single-image inference for trained checkpoints.

Convenient wrapper around ``decode_greedy`` / ``decode_beam`` that handles
checkpoint loading and image preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .data import build_image_transforms, read_pil_image
from .decode import decode_beam, decode_greedy
from .model import build_model
from .tokenizer import load_tokenizer
from .utils import fix_floats


def load_inference_pipeline(
    checkpoint_path: str | Path,
    tokenizer_dir: str | Path = "cadquery_tokenizer",
    device: str | None = None,
    **model_overrides,
):
    """Load tokenizer + model + checkpoint, ready for inference.

    Returns:
        (model, tokenizer, transform, device)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(tokenizer_dir)
    model = build_model(tokenizer, **model_overrides).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    transform = build_image_transforms()
    return model, tokenizer, transform, device


def predict_code(
    image: Image.Image | dict,
    model,
    tokenizer,
    transform,
    device: str,
    *,
    method: str = "greedy",
    beam_size: int = 3,
    max_len: int = 256,
    apply_fix_floats: bool = True,
) -> str:
    """Generate CadQuery code for a single image.

    ``image`` accepts a PIL.Image or a HF-style dict with raw bytes.
    """
    pil = read_pil_image(image) if isinstance(image, dict) else image.convert("RGB")
    tensor = transform(pil).to(device)

    if method == "greedy":
        code = decode_greedy(model, tensor.unsqueeze(0), tokenizer, max_len=max_len)[0]
    elif method == "beam":
        code = decode_beam(model, tensor, tokenizer, max_len=max_len, beam_size=beam_size)
    else:
        raise ValueError(f"Unknown decoding method: {method!r}")

    return fix_floats(code) if apply_fix_floats else code
