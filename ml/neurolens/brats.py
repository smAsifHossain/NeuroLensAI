"""Reproducible, zero-training BraTS feasibility pipeline.

The pretrained network is NVIDIA/MONAI's ``brats_mri_segmentation`` bundle.
This module deliberately keeps measurements deterministic and separate from
future VLM-generated prose. It is for academic research, not clinical use.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import resource
except ImportError:  # Windows has no resource module; cloud runs use Linux.
    resource = None  # type: ignore[assignment]

BUNDLE_NAME = "brats_mri_segmentation"
BUNDLE_VERSION = "0.5.4"
BUNDLE_REPOSITORY = f"MONAI/{BUNDLE_NAME}"
BUNDLE_REVISION = "370f7f9d062745fbac445e7fe6d6616d35df04ec"
MODEL_ID = f"{BUNDLE_REPOSITORY}:{BUNDLE_VERSION}@{BUNDLE_REVISION[:7]}"
MODEL_DISPLAY_NAME = "SegResNet"
MODALITY_ORDER = ("t1c", "t1", "t2", "flair")
REGION_DEFINITIONS = {
    "whole_tumor": {1, 2, 4},
    "tumor_core": {1, 4},
    "enhancing_tumor": {4},
    "flair_abnormality_outside_core": {2},
}

REGION_COLORS = {
    1: (245, 166, 89),
    2: (101, 113, 232),
    4: (92, 229, 182),
}


@dataclass(frozen=True)
class CasePaths:
    """The four aligned MRI modalities required by the MONAI bundle."""

    case_id: str
    t1c: Path
    t1: Path
    t2: Path
    flair: Path
    ground_truth: Path | None = None

    def ordered_images(self) -> list[Path]:
        return [self.t1c, self.t1, self.t2, self.flair]

    def serializable(self) -> dict[str, str | None]:
        return {
            "case_id": self.case_id,
            "t1c": str(self.t1c),
            "t1": str(self.t1),
            "t2": str(self.t2),
            "flair": str(self.flair),
            "ground_truth": str(self.ground_truth) if self.ground_truth else None,
        }


@dataclass(frozen=True)
class InferenceRun:
    mask_path: Path
    elapsed_seconds: float
    peak_gpu_memory_mb: int | None
    child_peak_rss_mb: float | None
    model_sha256: str
    command: list[str]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nifti_stem(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".nii.gz"):
        return lowered[:-7]
    if lowered.endswith(".nii"):
        return lowered[:-4]
    return lowered


def modality_from_name(name: str) -> str | None:
    """Classify common legacy and modern BraTS filename suffixes."""

    stem = _nifti_stem(Path(name).name)
    token = stem.replace("-", "_").split("_")[-1]
    if token in {"seg", "label", "labels"}:
        return "ground_truth"
    if token in {"t1ce", "t1c"}:
        return "t1c"
    if token in {"flair", "t2f"}:
        return "flair"
    if token in {"t2", "t2w"}:
        return "t2"
    if token in {"t1", "t1n"}:
        return "t1"
    return None


def discover_cases(root: str | Path) -> list[CasePaths]:
    """Find folders containing a complete BraTS-style four-modality case."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Data root does not exist: {root_path}")

    grouped: dict[Path, dict[str, Path]] = {}
    candidates = [
        path
        for path in list(root_path.rglob("*.nii")) + list(root_path.rglob("*.nii.gz"))
        if path.is_file()
    ]
    for path in sorted(set(candidates)):
        modality = modality_from_name(path.name)
        if modality is None:
            continue
        grouped.setdefault(path.parent, {})[modality] = path.resolve()

    # Kaggle expands each ``.nii.gz`` upload into a directory named ``*.nii``
    # when the gzip header carries an original filename. Recover the single
    # NIfTI member while retaining the BraTS modality encoded by the directory.
    for container in sorted(path for path in root_path.rglob("*.nii") if path.is_dir()):
        modality = modality_from_name(container.name)
        nested = sorted(path for path in container.rglob("*.nii") if path.is_file())
        if modality is not None and len(nested) == 1:
            grouped.setdefault(container.parent, {})[modality] = nested[0].resolve()

    cases: list[CasePaths] = []
    for folder, modalities in sorted(grouped.items(), key=lambda item: str(item[0])):
        if not all(modality in modalities for modality in MODALITY_ORDER):
            continue
        cases.append(
            CasePaths(
                case_id=folder.name,
                t1c=modalities["t1c"],
                t1=modalities["t1"],
                t2=modalities["t2"],
                flair=modalities["flair"],
                ground_truth=modalities.get("ground_truth"),
            )
        )
    return cases


def _require_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "nibabel is required for NIfTI validation. Install ml/requirements-cloud.txt."
        ) from error
    return nib


def validate_nifti_case(case: CasePaths) -> dict[str, Any]:
    """Validate existence, dimensionality, alignment, affine, and spacing."""

    nib = _require_nibabel()
    images: dict[str, Any] = {}
    warnings: list[str] = []
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None

    for modality in MODALITY_ORDER:
        path = getattr(case, modality)
        if not path.exists():
            raise FileNotFoundError(f"Missing {modality} image: {path}")
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"{modality} must be a 3D volume, got shape {image.shape}")
        if reference_shape is None:
            reference_shape = tuple(int(value) for value in image.shape)
            reference_affine = np.asarray(image.affine)
        else:
            if tuple(image.shape) != reference_shape:
                raise ValueError(
                    f"All modalities must share one shape; {modality} is {image.shape}, "
                    f"expected {reference_shape}."
                )
            if not np.allclose(image.affine, reference_affine, rtol=0, atol=1e-3):
                raise ValueError(f"{modality} affine does not match the other modalities.")
        images[modality] = image

    assert reference_shape is not None
    assert reference_affine is not None
    spacing = tuple(float(value) for value in images["flair"].header.get_zooms()[:3])
    if not np.allclose(spacing, (1.0, 1.0, 1.0), rtol=0, atol=0.05):
        warnings.append(
            f"Expected approximately 1 mm isotropic spacing; received {spacing}. "
            "Use registered BraTS data."
        )

    if case.ground_truth is not None:
        truth = nib.load(str(case.ground_truth))
        if len(truth.shape) != 3 or tuple(truth.shape) != reference_shape:
            raise ValueError(
                f"Ground-truth shape {truth.shape} does not match modalities {reference_shape}."
            )
        if not np.allclose(truth.affine, reference_affine, rtol=0, atol=1e-3):
            raise ValueError("Ground-truth affine does not match the MRI modalities.")

    return {
        "case_id": case.case_id,
        "shape": list(reference_shape),
        "spacing_mm": list(spacing),
        "orientation": list(nib.aff2axcodes(reference_affine)),
        "affines_match": True,
        "warnings": warnings,
        "modalities": case.serializable(),
    }


def write_case_datalist(case: CasePaths, destination: str | Path) -> Path:
    """Write the MONAI Decathlon-style testing list in required channel order."""

    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "NeuroLens zero-training feasibility case",
        "tensorImageSize": "4D",
        "modality": {"0": "T1c", "1": "T1", "2": "T2", "3": "FLAIR"},
        "testing": [{"image": [str(path.resolve()) for path in case.ordered_images()]}],
    }
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def download_monai_bundle(bundle_dir: str | Path, python: str = sys.executable) -> Path:
    """Download the exact pretrained MONAI bundle version used by the project."""

    bundle_root = Path(bundle_dir).expanduser().resolve()
    expected = bundle_root / BUNDLE_NAME
    if (expected / "models" / "model.pt").exists():
        return expected

    bundle_root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to download the pinned MONAI bundle."
        ) from error

    snapshot_download(
        repo_id=BUNDLE_REPOSITORY,
        revision=BUNDLE_REVISION,
        local_dir=expected,
    )
    if not (expected / "models" / "model.pt").exists():
        raise FileNotFoundError(f"Downloaded bundle is missing model.pt: {expected}")
    return expected


class _GpuMemorySampler:
    def __init__(self, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mb: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if values:
                    current = max(values)
                    self.peak_mb = current if self.peak_mb is None else max(self.peak_mb, current)
            except (FileNotFoundError, ValueError, subprocess.SubprocessError):
                return
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "_GpuMemorySampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def run_monai_inference(
    case: CasePaths,
    bundle_root: str | Path,
    run_dir: str | Path,
    python: str = sys.executable,
    require_cuda: bool = True,
) -> InferenceRun:
    """Run official bundle inference and return the produced NIfTI label map."""

    bundle_path = Path(bundle_root).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    output_path = run_path / "model_output"
    datalist_path = write_case_datalist(case, run_path / "datalist.json")
    output_path.mkdir(parents=True, exist_ok=True)

    if require_cuda:
        check = subprocess.run(
            [python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
            check=False,
        )
        if check.returncode != 0:
            raise RuntimeError(
                "CUDA GPU not available. Use a Kaggle/Colab GPU runtime for this milestone."
            )

    config_path = bundle_path / "configs" / "inference.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Bundle inference config not found: {config_path}")
    model_path = bundle_path / "models" / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Bundle checkpoint not found: {model_path}")

    command = [
        python,
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        str(config_path),
        "--bundle_root",
        str(bundle_path),
        "--dataset_dir",
        str(case.t1c.parent.resolve()),
        "--data_list_file_path",
        str(datalist_path),
        "--output_dir",
        str(output_path),
        "--dataloader#shuffle",
        "false",
        "--dataloader#num_workers",
        "2",
    ]

    before_rss = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if resource is not None else None
    )
    started = time.perf_counter()
    with _GpuMemorySampler() as sampler:
        subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    after_rss = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if resource is not None else None
    )

    masks = sorted(output_path.rglob("*_seg.nii.gz"))
    if not masks:
        raise FileNotFoundError(f"MONAI completed but produced no *_seg.nii.gz under {output_path}")

    rss_mb: float | None = None
    if before_rss is not None and after_rss is not None:
        rss_delta = max(0.0, float(after_rss - before_rss))
        divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
        rss_mb = rss_delta / divisor or None
    return InferenceRun(
        mask_path=masks[0],
        elapsed_seconds=elapsed,
        peak_gpu_memory_mb=sampler.peak_mb,
        child_peak_rss_mb=rss_mb,
        model_sha256=_sha256_file(model_path),
        command=command,
    )


def load_nifti_case(
    case: CasePaths, mask_path: str | Path
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[float, float, float]]:
    """Load modalities and mask in closest-canonical RAS orientation."""

    nib = _require_nibabel()
    arrays: dict[str, np.ndarray] = {}
    canonical_shape: tuple[int, ...] | None = None
    canonical_affine: np.ndarray | None = None
    spacing: tuple[float, float, float] | None = None

    for modality in MODALITY_ORDER:
        image = nib.as_closest_canonical(nib.load(str(getattr(case, modality))))
        data = np.asarray(image.dataobj, dtype=np.float32)
        if canonical_shape is None:
            canonical_shape = data.shape
            canonical_affine = np.asarray(image.affine)
            spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        elif data.shape != canonical_shape:
            raise ValueError("Canonical modality shapes do not match.")
        arrays[modality] = data

    mask_image = nib.as_closest_canonical(nib.load(str(mask_path)))
    mask = normalize_brats_labels(np.asarray(mask_image.dataobj))
    if canonical_shape is None or mask.shape != canonical_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match images {canonical_shape}.")
    if canonical_affine is None or not np.allclose(
        mask_image.affine, canonical_affine, rtol=0, atol=1e-3
    ):
        raise ValueError("Mask affine does not match the canonical MRI modalities.")
    assert spacing is not None
    return arrays, mask, spacing


def _mask_for_region(mask: np.ndarray, values: Iterable[int]) -> np.ndarray:
    return np.isin(mask, list(values))


def normalize_brats_labels(mask: np.ndarray) -> np.ndarray:
    """Convert modern BraTS ET label 3 to the bundle's canonical ET label 4."""

    array = np.asarray(mask)
    if not np.isfinite(array).all():
        raise ValueError("BraTS label maps cannot contain NaN or infinite values.")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError("BraTS label maps must contain integer-valued labels.")
    labels = {int(value) for value in np.unique(rounded)}
    unknown = labels - {0, 1, 2, 3, 4}
    if unknown:
        raise ValueError(f"Unexpected BraTS labels: {sorted(unknown)}")
    if 3 in labels and 4 in labels:
        raise ValueError("Label map mixes modern ET label 3 and legacy ET label 4.")

    canonical = rounded.astype(np.uint8)
    if 3 in labels:
        canonical[canonical == 3] = 4
    return canonical


def calculate_region_metrics(
    mask: np.ndarray,
    spacing_mm: Sequence[float],
    brain_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate nested BraTS volumes and approximate canonical laterality."""

    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D, got {mask.ndim} dimensions.")
    mask = normalize_brats_labels(mask)
    if len(spacing_mm) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0 for value in spacing_mm
    ):
        raise ValueError("spacing_mm must contain three positive finite values.")

    voxel_volume_ml = float(np.prod(np.asarray(spacing_mm, dtype=np.float64))) / 1000.0
    result: dict[str, Any] = {}
    for name, values in REGION_DEFINITIONS.items():
        region = _mask_for_region(mask, values)
        voxel_count = int(region.sum())
        result[name] = {
            "voxel_count": voxel_count,
            "volume_ml": round(voxel_count * voxel_volume_ml, 3),
        }

    whole = _mask_for_region(mask, REGION_DEFINITIONS["whole_tumor"])
    coordinates = np.argwhere(whole)
    if coordinates.size:
        centroid = coordinates.mean(axis=0)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        if brain_mask is not None and brain_mask.shape == mask.shape and brain_mask.any():
            brain_coordinates = np.argwhere(brain_mask)
            midline = float(brain_coordinates[:, 0].min() + brain_coordinates[:, 0].max()) / 2.0
        else:
            midline = (mask.shape[0] - 1) / 2.0
        x_coordinates = np.arange(mask.shape[0], dtype=np.float32)[:, None, None]
        left_count = int(np.logical_and(whole, x_coordinates < midline).sum())
        right_count = int(np.logical_and(whole, x_coordinates > midline).sum())
        total_lr = max(1, left_count + right_count)
        left_fraction = left_count / total_lr
        right_fraction = right_count / total_lr
        if left_fraction >= 0.8:
            laterality = "predominantly left"
        elif right_fraction >= 0.8:
            laterality = "predominantly right"
        else:
            laterality = "crosses or approaches midline"
        spatial = {
            "centroid_voxel_ras": [round(float(value), 2) for value in centroid],
            "bounding_box_voxel_ras": {
                "min": [int(value) for value in minimum],
                "max": [int(value) for value in maximum],
            },
            "laterality": laterality,
            "left_fraction": round(left_fraction, 4),
            "right_fraction": round(right_fraction, 4),
        }
    else:
        spatial = {
            "centroid_voxel_ras": None,
            "bounding_box_voxel_ras": None,
            "laterality": "no predicted tumor voxels",
            "left_fraction": 0.0,
            "right_fraction": 0.0,
        }

    return {
        "canonical_label_encoding": {
            "background": 0,
            "necrotic_or_non_enhancing_core": 1,
            "edema": 2,
            "enhancing_tumor": 4,
        },
        "voxel_spacing_mm": [float(value) for value in spacing_mm],
        "voxel_volume_ml": round(voxel_volume_ml, 6),
        "regions": result,
        "spatial_summary": spatial,
    }


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_count = int(prediction.sum())
    target_count = int(target.sum())
    if pred_count + target_count == 0:
        return 1.0
    intersection = int(np.logical_and(prediction, target).sum())
    return 2.0 * intersection / (pred_count + target_count)


def compute_dice_scores(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have identical shapes.")
    prediction = normalize_brats_labels(prediction)
    ground_truth = normalize_brats_labels(ground_truth)
    return {
        name: round(
            _dice(_mask_for_region(prediction, values), _mask_for_region(ground_truth, values)),
            5,
        )
        for name, values in REGION_DEFINITIONS.items()
        if name != "flair_abnormality_outside_core"
    }


def _normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    finite = np.asarray(slice_data, dtype=np.float32)
    nonzero = finite[np.isfinite(finite) & (finite != 0)]
    if not nonzero.size:
        return np.zeros(finite.shape, dtype=np.uint8)
    low, high = np.percentile(nonzero, [1, 99])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((finite - low) / (high - low), 0, 1)
    normalized[~np.isfinite(normalized)] = 0
    return np.rint(normalized * 255).astype(np.uint8)


def _overlay_image(base: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    gray = _normalize_slice(base)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for label, color in REGION_COLORS.items():
        selected = mask == label
        if selected.any():
            rgb[selected] = rgb[selected] * (1 - alpha) + np.asarray(color) * alpha
    rgb = np.rot90(np.clip(rgb, 0, 255).astype(np.uint8))
    return Image.fromarray(rgb, mode="RGB")


def _plain_image(base: np.ndarray) -> Image.Image:
    gray = np.rot90(_normalize_slice(base))
    return Image.fromarray(gray, mode="L").convert("RGB")


def _add_label(image: Image.Image, label: str) -> Image.Image:
    banner_height = 38
    labeled = Image.new("RGB", (image.width, image.height + banner_height), (20, 27, 56))
    labeled.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    draw.text((12, 12), label, fill=(241, 244, 250), font=font)
    return labeled


def render_representative_overlays(
    modalities: Mapping[str, np.ndarray],
    mask: np.ndarray,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render the axial slice with the largest predicted whole-tumor area."""

    missing = [modality for modality in MODALITY_ORDER if modality not in modalities]
    if missing:
        raise KeyError(f"Missing MRI modalities for overlays: {', '.join(missing)}")
    if any(modalities[modality].shape != mask.shape for modality in MODALITY_ORDER):
        raise ValueError("Overlay modalities and mask must have identical shapes.")
    mask = normalize_brats_labels(mask)

    region = _mask_for_region(mask, REGION_DEFINITIONS["whole_tumor"])
    areas = region.sum(axis=(0, 1))
    slice_index = int(np.argmax(areas)) if areas.size else mask.shape[2] // 2
    if int(areas[slice_index]) == 0:
        slice_index = mask.shape[2] // 2

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    images = {
        "flair_original": _add_label(
            _plain_image(modalities["flair"][:, :, slice_index]), "FLAIR - original scan"
        ),
        "t1c_overlay": _add_label(
            _overlay_image(modalities["t1c"][:, :, slice_index], mask[:, :, slice_index]),
            "T1c - Predicted Mask",
        ),
        "t1_overlay": _add_label(
            _overlay_image(modalities["t1"][:, :, slice_index], mask[:, :, slice_index]),
            "T1 - Predicted Mask",
        ),
        "t2_overlay": _add_label(
            _overlay_image(modalities["t2"][:, :, slice_index], mask[:, :, slice_index]),
            "T2 - Predicted Mask",
        ),
        "flair_overlay": _add_label(
            _overlay_image(modalities["flair"][:, :, slice_index], mask[:, :, slice_index]),
            "FLAIR - Predicted Mask",
        ),
    }
    paths: dict[str, str] = {}
    for key, image in images.items():
        path = destination / f"{key}.png"
        image.save(path)
        paths[key] = str(path)

    panels = list(images.values())
    panel_width = panels[0].width
    panel_height = panels[0].height
    comparison = Image.new("RGB", (panel_width * 3, panel_height * 2), (8, 12, 27))
    positions = [
        (0, 0),
        (panel_width, 0),
        (panel_width * 2, 0),
        (panel_width // 2, panel_height),
        (panel_width + panel_width // 2, panel_height),
    ]
    for image, position in zip(panels, positions):
        comparison.paste(image, position)
    comparison_path = destination / "comparison.png"
    comparison.save(comparison_path)
    paths["comparison"] = str(comparison_path)

    return {"representative_axial_slice": slice_index, "artifacts": paths}


def deterministic_summary(case_id: str, metrics: Mapping[str, Any]) -> str:
    """Create a measurement-grounded summary without an LLM."""

    del case_id  # The scan identifier is presented separately from the findings.
    regions = metrics["regions"]
    whole = float(regions["whole_tumor"]["volume_ml"])
    core = float(regions["tumor_core"]["volume_ml"])
    enhancing = float(regions["enhancing_tumor"]["volume_ml"])
    edema = float(regions["flair_abnormality_outside_core"]["volume_ml"])
    laterality = metrics["spatial_summary"]["laterality"]
    if whole == 0:
        return (
            "Automated findings: The predicted mask contains no tumor-region voxels, "
            "so all reported region volumes are 0.000 mL. These measurements describe "
            "the model output for academic review and are not a diagnosis."
        )

    if laterality == "predominantly left":
        location = "The predicted regions are predominantly left-sided."
    elif laterality == "predominantly right":
        location = "The predicted regions are predominantly right-sided."
    else:
        location = "The predicted regions cross or approach the brain midline."

    if core > 0:
        enhancing_share = enhancing / core * 100
        core_detail = (
            f"The predicted tumor core measures {core:.3f} mL and includes "
            f"{enhancing:.3f} mL of enhancing tumor ({enhancing_share:.1f}% of the core)."
        )
    else:
        core_detail = "No tumor-core or enhancing-tumor voxels are present in the predicted mask."

    return (
        f"Automated findings: The {MODEL_DISPLAY_NAME} mask contains a total predicted "
        f"tumor-region volume of {whole:.3f} mL. {core_detail} Predicted peritumoral "
        f"edema measures {edema:.3f} mL. {location} These measurements describe the "
        "model output for academic review and are not a diagnosis."
    )


def collect_environment() -> dict[str, Any]:
    """Record non-identifying runtime provenance for reproducibility."""

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_id": MODEL_ID,
        "research_only": True,
    }
    try:
        import torch  # type: ignore

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_version"] = torch.version.cuda
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        result.update({"torch": None, "cuda_available": False, "cuda_version": None, "gpu": None})
    try:
        import monai  # type: ignore

        result["monai"] = monai.__version__
    except ImportError:
        result["monai"] = None
    return result


def save_feasibility_result(
    destination: str | Path,
    case: CasePaths,
    validation: Mapping[str, Any],
    run: InferenceRun,
    metrics: Mapping[str, Any],
    overlays: Mapping[str, Any],
    dice_scores: Mapping[str, float] | None = None,
) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    relative_artifacts: dict[str, str] = {}
    for name, artifact in overlays.get("artifacts", {}).items():
        try:
            relative_artifacts[name] = str(Path(artifact).resolve().relative_to(path.parent))
        except ValueError:
            relative_artifacts[name] = str(artifact)
    try:
        relative_mask = str(run.mask_path.resolve().relative_to(path.parent))
    except ValueError:
        relative_mask = str(run.mask_path)

    payload = {
        "schema_version": "1.1",
        "environment": collect_environment(),
        "case": case.serializable(),
        "validation": dict(validation),
        "inference": {
            "mask_path": str(run.mask_path),
            "mask_relative_path": relative_mask,
            "mask_sha256": _sha256_file(run.mask_path),
            "model_sha256": run.model_sha256,
            "elapsed_seconds": round(run.elapsed_seconds, 3),
            "peak_gpu_memory_mb": run.peak_gpu_memory_mb,
            "child_peak_rss_mb": round(run.child_peak_rss_mb, 2) if run.child_peak_rss_mb else None,
            "command": run.command,
        },
        "input_sha256": {
            modality: _sha256_file(getattr(case, modality)) for modality in MODALITY_ORDER
        },
        "measurements": dict(metrics),
        "evaluation": {"dice": dict(dice_scores)} if dice_scores else None,
        "overlays": {**dict(overlays), "artifact_relative_paths": relative_artifacts},
        "summary": deterministic_summary(case.case_id, metrics),
        "limitations": [
            "SegResNet was not adapted to this specific uploaded scan.",
            "No clinical validation was performed.",
            "The result must not be used for diagnosis, treatment, or patient care.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
