from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from neurolens.brats import (
    CasePaths,
    InferenceRun,
    calculate_region_metrics,
    compute_dice_scores,
    deterministic_summary,
    discover_cases,
    modality_from_name,
    normalize_brats_labels,
    render_representative_overlays,
    save_feasibility_result,
    write_case_datalist,
)
from neurolens.worker import _create_analysis_pdf, safe_extract_brats_zip


class FilenameDiscoveryTests(unittest.TestCase):
    def test_legacy_and_modern_suffixes(self) -> None:
        expected = {
            "BraTS_001_t1ce.nii.gz": "t1c",
            "BraTS_001_t1c.nii.gz": "t1c",
            "BraTS_001_t1.nii.gz": "t1",
            "BraTS_001_t1n.nii.gz": "t1",
            "BraTS_001_t2.nii.gz": "t2",
            "BraTS_001_t2w.nii.gz": "t2",
            "BraTS_001_flair.nii.gz": "flair",
            "BraTS_001_t2f.nii.gz": "flair",
            "BraTS_001_seg.nii.gz": "ground_truth",
        }
        self.assertEqual({name: modality_from_name(name) for name in expected}, expected)

    def test_discovers_only_complete_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "BraTS-GLI-001"
            incomplete = root / "BraTS-GLI-002"
            complete.mkdir()
            incomplete.mkdir()
            for suffix in ("t1c", "t1n", "t2w", "t2f", "seg"):
                (complete / f"BraTS-GLI-001-{suffix}.nii.gz").touch()
            (incomplete / "BraTS-GLI-002-t1c.nii.gz").touch()

            cases = discover_cases(root)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "BraTS-GLI-001")
            self.assertIsNotNone(cases[0].ground_truth)

    def test_discovers_kaggle_auto_extracted_gzip_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case_root = Path(directory) / "BraTS-GLI-001"
            case_root.mkdir()
            nested_names = {
                "t1c": "brain_t1ce.nii",
                "t1n": "brain_t1.nii",
                "t2w": "brain_t2.nii",
                "t2f": "brain_flair.nii",
            }
            for modern_suffix, nested_name in nested_names.items():
                container = case_root / f"BraTS-GLI-001-{modern_suffix}.nii"
                container.mkdir()
                (container / nested_name).touch()
            (case_root / "BraTS-GLI-001-seg.nii").touch()

            cases = discover_cases(Path(directory))

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "BraTS-GLI-001")
            self.assertEqual(cases[0].t1c.name, "brain_t1ce.nii")
            self.assertIsNotNone(cases[0].ground_truth)


class MeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((10, 8, 6), dtype=np.uint8)
        self.mask[1, 1, 2] = 4
        self.mask[2:4, 1, 2] = 1
        self.mask[1:4, 2, 2] = 2

    def test_nested_region_volumes_and_laterality(self) -> None:
        brain = np.ones_like(self.mask, dtype=bool)
        result = calculate_region_metrics(self.mask, (2.0, 2.0, 2.0), brain)

        self.assertEqual(result["regions"]["whole_tumor"]["volume_ml"], 0.048)
        self.assertEqual(result["regions"]["tumor_core"]["volume_ml"], 0.024)
        self.assertEqual(result["regions"]["enhancing_tumor"]["volume_ml"], 0.008)
        self.assertEqual(
            result["regions"]["flair_abnormality_outside_core"]["volume_ml"], 0.024
        )
        self.assertEqual(result["spatial_summary"]["laterality"], "predominantly left")

    def test_dice_identity_and_empty_region(self) -> None:
        scores = compute_dice_scores(self.mask, self.mask.copy())
        self.assertEqual(scores, {"whole_tumor": 1.0, "tumor_core": 1.0, "enhancing_tumor": 1.0})
        empty = np.zeros_like(self.mask)
        self.assertEqual(compute_dice_scores(empty, empty)["whole_tumor"], 1.0)

    def test_modern_enhancing_label_is_normalized(self) -> None:
        modern = self.mask.copy()
        modern[modern == 4] = 3
        normalized = normalize_brats_labels(modern)
        self.assertNotIn(3, np.unique(normalized))
        self.assertEqual(compute_dice_scores(self.mask, modern)["enhancing_tumor"], 1.0)

    def test_findings_are_natural_and_measurement_grounded(self) -> None:
        metrics = calculate_region_metrics(self.mask, (2.0, 2.0, 2.0), np.ones_like(self.mask))
        findings = deterministic_summary("BraTS-GLI-00002-000", metrics)

        self.assertIn("Automated findings:", findings)
        self.assertIn("predicted tumor-region volume of 0.048 mL", findings)
        self.assertIn("predominantly left-sided", findings)
        self.assertIn("not a diagnosis", findings)
        self.assertNotIn("BraTS-GLI-00002-000", findings)
        self.assertNotIn("MONAI", findings)

    def test_overlay_outputs(self) -> None:
        modalities = {
            "flair": np.arange(self.mask.size, dtype=np.float32).reshape(self.mask.shape),
            "t1c": np.ones(self.mask.shape, dtype=np.float32),
            "t1": np.full(self.mask.shape, 2, dtype=np.float32),
            "t2": np.full(self.mask.shape, 3, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            result = render_representative_overlays(modalities, self.mask, directory)
            self.assertEqual(result["representative_axial_slice"], 2)
            self.assertEqual(
                set(result["artifacts"]),
                {
                    "flair_original",
                    "t1c_overlay",
                    "t1_overlay",
                    "t2_overlay",
                    "flair_overlay",
                    "comparison",
                },
            )
            for artifact in result["artifacts"].values():
                self.assertTrue(Path(artifact).is_file())

    def test_pdf_segmentation_summary(self) -> None:
        modalities = {
            "flair": np.arange(self.mask.size, dtype=np.float32).reshape(self.mask.shape),
            "t1c": np.ones(self.mask.shape, dtype=np.float32),
            "t1": np.full(self.mask.shape, 2, dtype=np.float32),
            "t2": np.full(self.mask.shape, 3, dtype=np.float32),
        }
        metrics = calculate_region_metrics(self.mask, (1, 1, 1))
        manifest = {
            "case_id": "synthetic-case",
            "summary": "Automated findings from a SegResNet test segmentation.",
            "inference": {
                "model_id": "MONAI/brats_mri_segmentation:0.5.4@370f7f9",
                "elapsed_seconds": 3.25,
                "hardware": "Test GPU",
                "mask_sha256": "a" * 64,
            },
            "measurements": metrics,
        }
        with tempfile.TemporaryDirectory() as directory:
            overlays = render_representative_overlays(modalities, self.mask, directory)
            destination = Path(directory) / "summary.pdf"
            result = _create_analysis_pdf(
                manifest, Path(overlays["artifacts"]["comparison"]), destination
            )
            self.assertEqual(result, destination)
            self.assertTrue(result.read_bytes().startswith(b"%PDF-"))


class DatalistTests(unittest.TestCase):
    def test_writes_required_channel_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / name
                for name in ("t1c.nii.gz", "t1.nii.gz", "t2.nii.gz", "flair.nii.gz")
            ]
            case = CasePaths("case", *paths)
            destination = write_case_datalist(case, root / "datalist.json")
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["testing"][0]["image"], [str(path.resolve()) for path in paths]
            )


class ResultManifestTests(unittest.TestCase):
    def test_manifest_contains_hashes_and_portable_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [
                root / name
                for name in ("t1c.nii.gz", "t1.nii.gz", "t2.nii.gz", "flair.nii.gz")
            ]
            for index, path in enumerate(inputs):
                path.write_bytes(f"input-{index}".encode())
            case = CasePaths("case", *inputs)
            run_dir = root / "case"
            mask_path = run_dir / "model_output" / "case_seg.nii.gz"
            comparison = run_dir / "overlays" / "comparison.png"
            mask_path.parent.mkdir(parents=True)
            comparison.parent.mkdir(parents=True)
            mask_path.write_bytes(b"mask")
            comparison.write_bytes(b"png")
            run = InferenceRun(mask_path, 1.25, 1024, 512.0, "model-hash", ["python"])
            metrics = calculate_region_metrics(np.zeros((2, 2, 2), dtype=np.uint8), (1, 1, 1))

            result_path = save_feasibility_result(
                run_dir / "feasibility_result.json",
                case,
                {"warnings": []},
                run,
                metrics,
                {"representative_axial_slice": 0, "artifacts": {"comparison": str(comparison)}},
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(result["schema_version"], "1.1")
            self.assertEqual(
                result["inference"]["mask_relative_path"],
                str(Path("model_output") / "case_seg.nii.gz"),
            )
            self.assertEqual(result["inference"]["model_sha256"], "model-hash")
            self.assertEqual(len(result["inference"]["mask_sha256"]), 64)
            self.assertEqual(len(result["input_sha256"]), 4)


class WorkerArchiveTests(unittest.TestCase):
    def test_extracts_a_bounded_case_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "case.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for suffix in ("t1c", "t1n", "t2w", "t2f"):
                    zipped.writestr(f"BraTS-001/BraTS-001-{suffix}.nii.gz", suffix)

            destination = safe_extract_brats_zip(archive, root / "extracted")

            self.assertTrue((destination / "BraTS-001" / "BraTS-001-t1c.nii.gz").is_file())

    def test_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.nii.gz", b"unsafe")

            with self.assertRaisesRegex(ValueError, "unsafe file path"):
                safe_extract_brats_zip(archive, root / "extracted")


if __name__ == "__main__":
    unittest.main()
