"""Image-to-CadQuery model: vision encoder + Transformer decoder.

The baseline uses a ResNet-18 encoder and a small Transformer decoder.
The architecture is parameterized so that swapping in a stronger vision
backbone (DINOv2, SigLIP) or a different decoder is straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet18


@dataclass
class ModelConfig:
    """All hyperparameters needed to instantiate Image2CADQuery."""
    vocab_size: int
    pad_idx: int
    embed_dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_position: int = 1024


class Image2CADQuery(nn.Module):
    """ResNet-18 encoder + Transformer decoder for image-to-code generation.

    Forward signature:
        images:     (B, 3, H, W)
        tgt_tokens: (B, L)  shifted target tokens (input to decoder)
    Returns:
        logits:     (B, L, vocab_size)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Vision encoder — drop classifier head, project to embed_dim
        resnet = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(resnet.fc.in_features, cfg.embed_dim)

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_idx)
        self.pos_emb = nn.Embedding(cfg.max_position, cfg.embed_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.n_layers)

        # Output projection
        self.fc_out = nn.Linear(cfg.embed_dim, cfg.vocab_size)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return memory tensor of shape (B, 1, embed_dim) for the decoder."""
        B = images.shape[0]
        feats = self.backbone(images)             # (B, 512, 1, 1)
        feats = feats.view(B, -1)                 # (B, 512)
        mem = self.img_proj(feats).unsqueeze(1)   # (B, 1, embed_dim)
        return mem

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        B, L = tgt_tokens.shape
        mem = self.encode_image(images)

        pos = torch.arange(L, device=images.device).unsqueeze(0)
        tgt = self.tok_emb(tgt_tokens) + self.pos_emb(pos)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(images.device)
        dec_out = self.decoder(tgt, mem, tgt_mask=tgt_mask)
        return self.fc_out(dec_out)


def build_model(tokenizer, **overrides) -> Image2CADQuery:
    """Convenience factory that derives vocab_size/pad_idx from a tokenizer."""
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_token_id,
        **overrides,
    )
    return Image2CADQuery(cfg)
