"""Evaluation metrics for image-to-CadQuery models.

Two metrics:

1. **VSR (Valid Syntax Rate)**: fraction of generated scripts that execute and
   produce a CadQuery object (Workplane / Solid / Compound).
2. **IoU (best)**: 3D voxel intersection-over-union between the predicted and
   ground-truth meshes after centroid + gyration-radius normalization and
   principal-axis alignment (best of 4 sign flips).

The IoU pipeline mirrors the reference evaluator from the GenCAD-Code task.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Union

import numpy as np
import trimesh

import cadquery as cq
from cadquery import exporters

os.environ.setdefault("CADQUERY_LOG_LEVEL", "ERROR")

SolidLike = Union[cq.Solid, cq.Compound]


# ─── Code execution ────────────────────────────────────────────────────────

def load_solid_from_code(code: str, script_id: str = "unknown") -> SolidLike:
    """Execute CadQuery code and return the resulting solid/compound.

    The execution namespace pre-imports ``cq``, ``cadquery``, ``np``, ``numpy``
    so generated scripts can omit imports. We search the resulting locals for
    common variable names (``solid``, ``result``, ``shape``, …) and fall back
    to "first CadQuery object found".
    """
    cleaned = textwrap.dedent(code).strip()
    ns = {"cq": cq, "cadquery": cq, "np": np, "numpy": np, "__builtins__": __builtins__}

    try:
        exec(cleaned, ns)
    except Exception as e:
        raise ValueError(f"Error executing script {script_id}: {e}")

    cad_objs = [
        (name, val) for name, val in ns.items()
        if isinstance(val, (cq.Workplane, cq.Solid, cq.Compound))
    ]
    if not cad_objs:
        raise ValueError(f"No CadQuery objects found in script {script_id}")

    if len(cad_objs) > 1:
        for preferred in ("solid", "result", "shape", "part", "object", "obj", "res"):
            match = [t for t in cad_objs if t[0] == preferred]
            if match:
                cad_objs = match
                break

    _, solid_obj = cad_objs[0]
    if isinstance(solid_obj, cq.Workplane):
        solid_obj = solid_obj.val()

    if hasattr(solid_obj, "Solids") and callable(solid_obj.Solids):
        solids = solid_obj.Solids()
        if len(solids) == 1:
            solid_obj = solids[0]
        elif not solids:
            raise ValueError(f"No solids in compound for script {script_id}")

    if not isinstance(solid_obj, (cq.Solid, cq.Compound)):
        raise ValueError(f"Object is not Solid/Compound in script {script_id}")
    return solid_obj


# ─── Mesh normalization & IoU ──────────────────────────────────────────────

def _root_gyration(solid: SolidLike) -> float:
    vol = solid.Volume()
    inertia = np.array(cq.Shape.matrixOfInertia(solid)).reshape(3, 3)
    return np.sqrt(np.trace(inertia) / (2.0 * vol))


def _normalized_mesh(solid: SolidLike) -> trimesh.Trimesh:
    """Center at origin and scale isotropically by the radius of gyration."""
    r_g = _root_gyration(solid)
    c = solid.Center()
    centroid = np.array([c.x, c.y, c.z])

    with tempfile.TemporaryDirectory() as tmp:
        stl_path = Path(tmp) / "part.stl"
        exporters.export(solid, str(stl_path))
        mesh = trimesh.load(str(stl_path), force="mesh")
    mesh.apply_translation(-centroid)
    mesh.apply_scale(1.0 / r_g)
    return mesh


def _principal_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    _, vecs = np.linalg.eigh(mesh.moment_inertia)
    return vecs


def _apply_rotation(mesh: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    T = np.eye(4)
    T[:3, :3] = R
    m = mesh.copy()
    m.apply_transform(T)
    return m


def _voxel_bool_unified(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    pitch: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize both meshes into a common grid so logical_and/or are valid."""
    v1, v2 = mesh1.voxelized(pitch), mesh2.voxelized(pitch)
    b1, b2 = v1.bounds, v2.bounds
    lo = np.minimum(b1[0], b2[0])
    hi = np.maximum(b1[1], b2[1])
    grid = np.ceil((hi - lo) / pitch).astype(int)

    a = np.zeros(grid, dtype=bool)
    b = np.zeros(grid, dtype=bool)
    o1 = np.round((b1[0] - lo) / pitch).astype(int)
    o2 = np.round((b2[0] - lo) / pitch).astype(int)
    e1, e2 = o1 + v1.matrix.shape, o2 + v2.matrix.shape

    if np.all(o1 >= 0) and np.all(e1 <= grid):
        a[o1[0]:e1[0], o1[1]:e1[1], o1[2]:e1[2]] = v1.matrix
    if np.all(o2 >= 0) and np.all(e2 <= grid):
        b[o2[0]:e2[0], o2[1]:e2[1], o2[2]:e2[2]] = v2.matrix
    return a, b


def iou_best(
    mesh_gt: trimesh.Trimesh,
    mesh_pred: trimesh.Trimesh,
    pitch: float = 0.05,
) -> float:
    """Best IoU across 4 valid principal-axis sign flips."""
    axes_gt = _principal_axes(mesh_gt)
    axes_pr = _principal_axes(mesh_pred)

    best = 0.0
    for signs in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]:
        R = axes_gt @ (axes_pr @ np.diag(signs)).T
        aligned = _apply_rotation(mesh_pred, R)
        vox_gt, vox_pr = _voxel_bool_unified(mesh_gt, aligned, pitch)
        inter = np.logical_and(vox_gt, vox_pr).sum()
        union = np.logical_or(vox_gt, vox_pr).sum()
        if union > 0:
            best = max(best, inter / union)
    return best


def get_iou_best(code_gt: str, code_pred: str, pitch: float = 0.05) -> float:
    """End-to-end: two code strings → IoU."""
    s1 = load_solid_from_code(code_gt, "gt")
    s2 = load_solid_from_code(code_pred, "pred")
    return iou_best(_normalized_mesh(s1), _normalized_mesh(s2), pitch)


# ─── Full evaluation runner ────────────────────────────────────────────────

def evaluate_codes(
    gt_codes: dict[str, str],
    pred_codes: dict[str, str],
    pitch: float = 0.05,
    verbose: bool = False,
) -> dict:
    """Compute VSR + mean IoU over a {id: code} dict pair.

    A sample counts toward VSR only if **both** GT and prediction execute.
    IoU is averaged over samples where both execute.
    """
    ids = sorted(gt_codes.keys())
    if not ids:
        raise ValueError("no ground-truth scripts provided")

    vsr_success = 0
    ious: list[float] = []

    for _id in ids:
        if _id not in pred_codes:
            if verbose:
                print(f"missing prediction for {_id}, skipping")
            continue
        try:
            s_gt = load_solid_from_code(gt_codes[_id], f"gt_{_id}")
            s_pr = load_solid_from_code(pred_codes[_id], f"pred_{_id}")
            vsr_success += 1
        except Exception as exc:
            if verbose:
                print(f"{_id}: error -> {exc}")
            continue

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m_gt = _normalized_mesh(s_gt)
                m_pr = _normalized_mesh(s_pr)
                ious.append(iou_best(m_gt, m_pr, pitch))
        except Exception as exc:
            if verbose:
                print(f"{_id}: IoU error -> {exc}")

    n_total = len(ids)
    vsr = vsr_success / n_total if n_total else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"vsr": vsr, "mean_iou": mean_iou, "n_evaluated": n_total}


def evaluate_syntax_rate(codes: dict[str, str], verbose: bool = False) -> dict:
    """Compute VSR alone — useful when you don't have GT meshes."""
    if not codes:
        return {"vsr": 0.0, "successful": 0, "total": 0, "failed_ids": []}

    ids = sorted(codes.keys())
    successful = 0
    failed: list[str] = []
    for sid in ids:
        try:
            load_solid_from_code(codes[sid], sid)
            successful += 1
        except Exception as exc:
            failed.append(sid)
            if verbose:
                print(f"✗ {sid}: {exc}")

    return {
        "vsr": successful / len(ids),
        "successful": successful,
        "total": len(ids),
        "failed_ids": failed,
    }
