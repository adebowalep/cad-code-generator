"""Decoding strategies: batched greedy and single-image beam search.

Greedy decoding is fast and works on batches — use it for bulk evaluation.
Beam search runs one image at a time but often improves syntax validity
by exploring alternative token continuations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_greedy(
    model,
    images: torch.Tensor,
    tokenizer,
    max_len: int = 256,
) -> list[str]:
    """Batched greedy decoding.

    Args:
        model:     Trained Image2CADQuery (in eval mode).
        images:    (B, 3, H, W) float tensor on the same device as the model.
        tokenizer: HF tokenizer with bos/eos/pad ids set.
        max_len:   Hard cap on generated length.

    Returns:
        List of decoded code strings, length B.
    """
    model.eval()
    device = images.device
    B = images.shape[0]
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    tgt = torch.full((B, 1), bos_id, device=device, dtype=torch.long)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model(images, tgt)                # (B, L, V)
        next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == eos_id
        if finished.all():
            break

    pred_codes = []
    for seq in tgt:
        seq = seq[1:]  # strip BOS
        if eos_id in seq:
            eos_pos = (seq == eos_id).nonzero(as_tuple=True)[0][0]
            seq = seq[:eos_pos]
        pred_codes.append(tokenizer.decode(seq.tolist()))
    return pred_codes


@torch.no_grad()
def decode_beam(
    model,
    image: torch.Tensor,
    tokenizer,
    max_len: int = 256,
    beam_size: int = 3,
) -> str:
    """Beam search for a single image.

    Args:
        model:     Trained Image2CADQuery (in eval mode).
        image:     (3, H, W) on the same device as the model. (Single image — beam
                   search is not trivially batched and this matches the original
                   notebook's behavior.)
        tokenizer: HF tokenizer.
        max_len:   Hard cap on generated length.
        beam_size: Number of beams to keep.

    Returns:
        Best decoded code string.
    """
    model.eval()
    device = image.device
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    sequences = [(torch.tensor([[bos_id]], device=device), 0.0)]

    for _ in range(max_len):
        candidates = []
        for seq, score in sequences:
            logits = model(image.unsqueeze(0), seq)  # (1, L, V)
            probs = F.log_softmax(logits[0, -1, :], dim=-1)
            topk_probs, topk_ids = probs.topk(beam_size)
            for i in range(beam_size):
                next_id = topk_ids[i].item()
                new_seq = torch.cat(
                    [seq, torch.tensor([[next_id]], device=device)], dim=1
                )
                candidates.append((new_seq, score + topk_probs[i].item()))

        sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[0, -1].item() == eos_id for seq, _ in sequences):
            break

    best_seq = sequences[0][0]
    # Skip BOS; trim at EOS if present
    tokens = best_seq[0, 1:].tolist()
    if eos_id in tokens:
        tokens = tokens[: tokens.index(eos_id)]
    return tokenizer.decode(tokens)
