"""Dataset, image transforms, and batch collation for the CAD code generator.

The dataset wraps the Hugging Face `CADCODER/GenCAD-Code` dataset and exposes
images as normalized tensors plus tokenized CadQuery code with BOS/EOS markers.
"""

from __future__ import annotations

import io
from typing import Any, Callable

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet statistics — appropriate for ResNet/DINO-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard eval-style transform pipeline.

    We currently use the same transform for train/val/test for reproducibility.
    Augmentations like RandomResizedCrop can hurt vision-to-code tasks because
    the geometry of the part is the signal — augment with care.
    """
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=Image.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_pil_image(img_field: Any) -> Image.Image:
    """Coerce a HF image field (dict with 'bytes') or a PIL.Image into RGB PIL.Image."""
    if isinstance(img_field, dict):
        return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    raise TypeError(f"Unknown image type: {type(img_field)}")


class CADCodeDataset(Dataset):
    """Wraps a Hugging Face split for image-to-CadQuery training.

    Each item is a dict:
        {
          "image":       FloatTensor (3, H, W),
          "tokens":      LongTensor  (L,)  -- includes BOS and EOS,
          "code_string": str,
          "id":          str,
        }
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        code_col: str = "cadquery",
        id_col: str = "deepcad_id",
        max_len: int = 256,
        img_transform: Callable | None = None,
    ):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.code_col = code_col
        self.id_col = id_col
        self.max_len = max_len
        self.img_transform = img_transform

        # Cache special token ids
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]

        pil_img = read_pil_image(row["image"])
        img = self.img_transform(pil_img) if self.img_transform is not None else pil_img

        # Truncate to leave room for BOS + EOS
        ids = self.tokenizer(
            row[self.code_col],
            add_special_tokens=False,
        ).input_ids[: self.max_len - 2]
        ids = [self.bos_id] + ids + [self.eos_id]

        return {
            "image": img,
            "tokens": torch.tensor(ids, dtype=torch.long),
            "code_string": row[self.code_col],
            "id": row[self.id_col],
        }


def make_collate_fn(pad_idx: int) -> Callable:
    """Build a collate_fn closed over the tokenizer's pad_idx.

    Returns (imgs, padded_tokens, meta) where meta carries the raw strings + ids
    needed for evaluation.
    """
    def collate(samples: list[dict]):
        imgs = torch.stack([s["image"] for s in samples])
        seqs = [s["tokens"] for s in samples]
        tgt_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
        meta = {
            "code_strings": [s["code_string"] for s in samples],
            "ids": [s["id"] for s in samples],
        }
        return imgs, tgt_padded, meta

    return collate


def build_dataloaders(
    hf_train,
    hf_val,
    hf_test,
    tokenizer,
    batch_train: int = 128,
    batch_eval: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    max_len: int = 256,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience builder for the three standard dataloaders."""
    tfm = build_image_transforms(image_size)
    collate = make_collate_fn(tokenizer.pad_token_id)

    train_ds = CADCodeDataset(hf_train, tokenizer, max_len=max_len, img_transform=tfm)
    val_ds = CADCodeDataset(hf_val, tokenizer, max_len=max_len, img_transform=tfm)
    test_ds = CADCodeDataset(hf_test, tokenizer, max_len=max_len, img_transform=tfm)

    train_dl = DataLoader(
        train_ds, batch_size=batch_train, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_eval, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_eval, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    return train_dl, val_dl, test_dl
