"""Command-line entry point for the NeuroLens feasibility run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .brats import (
    calculate_region_metrics,
    compute_dice_scores,
    discover_cases,
    download_monai_bundle,
    load_nifti_case,
    render_representative_overlays,
    run_monai_inference,
    save_feasibility_result,
    validate_nifti_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-training MONAI BraTS inference on one case.")
    parser.add_argument("--data-root", required=True, type=Path, help="Folder containing BraTS NIfTI cases.")
    parser.add_argument("--work-dir", default=Path("artifacts/feasibility"), type=Path)
    parser.add_argument("--bundle-dir", default=Path("bundles"), type=Path)
    parser.add_argument("--case-index", default=0, type=int)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow slow CPU inference; not recommended.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = discover_cases(args.data_root)
    if not cases:
        raise SystemExit(f"No complete BraTS cases found under {args.data_root}")
    if not 0 <= args.case_index < len(cases):
        raise SystemExit(f"case-index {args.case_index} is outside 0..{len(cases) - 1}")

    case = cases[args.case_index]
    run_dir = args.work_dir.resolve() / case.case_id
    validation = validate_nifti_case(case)
    bundle = download_monai_bundle(args.bundle_dir)
    run = run_monai_inference(case, bundle, run_dir, require_cuda=not args.allow_cpu)
    modalities, prediction, spacing = load_nifti_case(case, run.mask_path)
    brain_mask = modalities["flair"] != 0
    metrics = calculate_region_metrics(prediction, spacing, brain_mask)
    overlays = render_representative_overlays(modalities, prediction, run_dir / "overlays")

    dice_scores = None
    if case.ground_truth:
        import nibabel as nib

        truth = np.rint(
            np.asarray(nib.as_closest_canonical(nib.load(str(case.ground_truth))).dataobj)
        ).astype(np.uint8)
        dice_scores = compute_dice_scores(prediction, truth)

    result_path = save_feasibility_result(
        run_dir / "feasibility_result.json",
        case,
        validation,
        run,
        metrics,
        overlays,
        dice_scores,
    )
    print(json.dumps({"case_id": case.case_id, "result": str(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
