# CAD Code Generator

> Generate parametric CadQuery code from a single rendered image of a 3D part.

A vision-to-code model that takes a rendered image of a CAD part and produces
the corresponding [CadQuery](https://github.com/CadQuery/cadquery) Python
script. Trained on the
[CADCODER/GenCAD-Code](https://huggingface.co/datasets/CADCODER/GenCAD-Code)
dataset (≈147k image / code pairs derived from DeepCAD).

| Metric                       | Baseline (ResNet-18 + 4-layer Transformer) |
| ---------------------------- | ------------------------------------------ |
| Valid Syntax Rate (VSR)      | **~50 %**                                  |
| Mean IoU (on valid outputs)  | **~0.78**                                  |
| Trainable params             | ~22 M                                      |
| Training time (T4, 15 ep.)   | ~10 h                                      |

## Why this is interesting

CAD code generation is a niche but technically rich problem:
- **Multimodal**: image input → code output.
- **Structurally constrained**: predictions must parse as Python *and* evaluate
  to valid CadQuery geometry.
- **Two-tier evaluation**: syntax validity (VSR) + 3D geometric similarity
  (voxel IoU after principal-axis alignment).

## Architecture

```
┌──────────────┐   ┌────────────────┐   ┌──────────────────────┐   ┌────────────┐
│  CAD image   │ → │   ResNet-18    │ → │  Transformer decoder │ → │  CadQuery  │
│  224×224×3   │   │  (ImageNet)    │   │  4 layers, 8 heads   │   │   code     │
└──────────────┘   └────────────────┘   └──────────────────────┘   └────────────┘
                          ↓                       ↑
                  512-d image feature      BPE tokens (16k vocab)
```

Training uses teacher forcing with cross-entropy and `ignore_index=pad`.
Inference supports both greedy and beam search.

## Repo layout

```
cad-code-generator/
├── src/cad_codegen/
│   ├── data.py          # Dataset, transforms, collate
│   ├── tokenizer.py     # BPE training + loading
│   ├── model.py         # Image2CADQuery (ResNet + Transformer)
│   ├── train.py         # Training / validation loops
│   ├── decode.py        # Greedy & beam search
│   ├── eval.py          # VSR + IoU computation
│   ├── inference.py     # Single-image inference helper
│   └── utils.py         # fix_floats, formatting helpers
├── scripts/
│   ├── train_tokenizer.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── 01_explore_data.ipynb
├── pyproject.toml
└── requirements.txt
```

## Quickstart

### 1. Install

```bash
git clone https://github.com/<your-user>/cad-code-generator.git
cd cad-code-generator

# Linux: CadQuery needs libgl
sudo apt-get install -y libgl1

pip install -e .
```

### 2. Train the tokenizer

```bash
python scripts/train_tokenizer.py --save-dir cadquery_tokenizer
```

### 3. Train the model

```bash
python scripts/train.py \
    --tokenizer-dir cadquery_tokenizer \
    --checkpoint-dir checkpoints \
    --epochs 15 \
    --batch-size 128
```

Checkpoints are saved per-epoch as `ckpt_eNN.pt`, plus a `best.pt` symlinked to
the lowest validation-loss epoch.

### 4. Evaluate on the test set

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer-dir cadquery_tokenizer \
    --method greedy \
    --n-samples 200
```

Reports VSR and mean IoU.

### 5. Single-image inference (Python)

```python
from PIL import Image
from cad_codegen.inference import load_inference_pipeline, predict_code

model, tokenizer, transform, device = load_inference_pipeline(
    checkpoint_path="checkpoints/best.pt",
    tokenizer_dir="cadquery_tokenizer",
)
img = Image.open("my_part.png")
code = predict_code(img, model, tokenizer, transform, device, method="beam")
print(code)
```

## Evaluation methodology

- **VSR (Valid Syntax Rate)**: fraction of predictions that execute and produce
  a CadQuery `Workplane` / `Solid` / `Compound` object. Implemented in
  `eval.evaluate_syntax_rate`.
- **IoU (best)**: predicted and ground-truth solids are converted to meshes,
  centered at their centroid, isotropically scaled by the radius of gyration,
  then aligned along principal axes (best of 4 sign flips) before voxelization
  at `pitch=0.05`. Standard `intersection / union` on the resulting boolean
  voxel grids. Implemented in `eval.iou_best`.

## Roadmap

This repo is the baseline. Planned improvements, in order of expected impact:

- [ ] **Stronger vision encoder** (DINOv2 / SigLIP) — current ResNet-18 leaves
      VSR on the table.
- [ ] **Pretrained code decoder** (Qwen2.5-Coder 0.5B) — bootstraps Python
      syntax knowledge.
- [ ] **Execution-feedback decoding loop** — on syntax failure, re-decode
      conditioned on the error message.
- [ ] **Gradio demo on Hugging Face Spaces** for live image → code → 3D viewer.

## Dataset

[`CADCODER/GenCAD-Code`](https://huggingface.co/datasets/CADCODER/GenCAD-Code):
~147k pairs of `(rendered_image, cadquery_code)` derived from the DeepCAD
dataset.

## Acknowledgments

- [CadQuery](https://github.com/CadQuery/cadquery) for the parametric CAD library.
- The [GenCAD-Code](https://huggingface.co/datasets/CADCODER/GenCAD-Code)
  authors for the dataset.
- IoU evaluation method follows the official GenCAD-Code task spec.

## License

MIT.
