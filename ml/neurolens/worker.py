"""Gradio API worker for real, upload-driven NeuroLens inference.

Run this module in a CUDA-enabled Kaggle, Colab, or Hugging Face environment.
The web client calls the named ``/analyze`` endpoint with one BraTS ZIP. The
worker returns only artifacts derived from that upload; it has no demo result.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import stat
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .brats import (
    BUNDLE_VERSION,
    MODEL_DISPLAY_NAME,
    MODEL_ID,
    calculate_region_metrics,
    collect_environment,
    compute_dice_scores,
    discover_cases,
    download_monai_bundle,
    load_nifti_case,
    render_representative_overlays,
    run_monai_inference,
    save_feasibility_result,
    validate_nifti_case,
)

MAX_ARCHIVE_MEMBERS = 64
MAX_UNCOMPRESSED_BYTES = int(1.5 * 1024 * 1024 * 1024)


def _case_id_from_archive(path: Path) -> str:
    stem = path.name[:-4] if path.name.lower().endswith(".zip") else path.stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return (cleaned or "uploaded-case")[:80]


def safe_extract_brats_zip(archive: str | Path, destination: str | Path) -> Path:
    """Extract a bounded ZIP while rejecting traversal, links, and zip bombs."""

    archive_path = Path(archive).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if archive_path.suffix.lower() != ".zip":
        raise ValueError("Upload one .zip archive containing a complete MRI scan set.")

    destination_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zipped:
        members = [member for member in zipped.infolist() if not member.is_dir()]
        if not members:
            raise ValueError("The uploaded ZIP is empty.")
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"The ZIP contains more than {MAX_ARCHIVE_MEMBERS} files.")
        if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("The uncompressed case is larger than the 1.5 GB research limit.")

        for member in members:
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Symbolic links are not allowed inside the ZIP.")
            target = (destination_path / member.filename).resolve()
            if destination_path != target and destination_path not in target.parents:
                raise ValueError("The ZIP contains an unsafe file path.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    return destination_path


def _create_analysis_pdf(
    manifest: dict[str, Any], comparison_path: Path, destination: Path
) -> Path:
    """Write a readable, measurement-grounded PDF segmentation summary."""

    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as ReportImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "NeuroLensTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#15130e"),
        alignment=TA_LEFT,
        spaceAfter=6 * mm,
    )
    section_style = ParagraphStyle(
        "NeuroLensSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#c7401e"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "NeuroLensBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2b2821"),
    )
    note_style = ParagraphStyle(
        "NeuroLensNote",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#982d13"),
        borderColor=colors.HexColor("#c7401e"),
        borderWidth=0.6,
        borderPadding=7,
        backColor=colors.HexColor("#faeee8"),
    )

    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="NeuroLens AI Brain Tumor Segmentation Summary",
        author="S M Asif Hossain",
        subject="Automated brain MRI segmentation summary",
    )
    regions = manifest["measurements"]["regions"]
    story: list[Any] = [
        Paragraph("NeuroLens AI", title_style),
        Paragraph("Brain Tumor Segmentation Summary", section_style),
        Paragraph(
            f"<b>Scan ID:</b> {html.escape(str(manifest['case_id']))}<br/>"
            f"<b>Model:</b> {MODEL_DISPLAY_NAME}<br/>"
            f"<b>Bundle version:</b> {BUNDLE_VERSION}",
            body_style,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "For academic and educational use only. This automated segmentation is not a diagnosis or a clinical radiology report.",
            note_style,
        ),
        Paragraph("Segmentation views", section_style),
    ]

    with PILImage.open(comparison_path) as comparison_image:
        width, height = comparison_image.size
    report_width = 178 * mm
    report_height = 105 * mm
    image_scale = min(report_width / width, report_height / height)
    story.append(
        ReportImage(
            str(comparison_path), width=width * image_scale, height=height * image_scale
        )
    )
    story.extend(
        [
            Paragraph("Automated findings", section_style),
            Paragraph(html.escape(str(manifest["summary"])), body_style),
            Spacer(1, 2 * mm),
            Paragraph(
                "The findings above are generated deterministically from the segmentation mask measurements. No vision-language model is used.",
                body_style,
            ),
            Paragraph("Measured tumor regions", section_style),
        ]
    )
    measurement_data = [
        ["Region", "Volume (mL)", "Voxel count"],
        ["Whole tumor", f"{regions['whole_tumor']['volume_ml']:.3f}", f"{regions['whole_tumor']['voxel_count']:,}"],
        ["Tumor core", f"{regions['tumor_core']['volume_ml']:.3f}", f"{regions['tumor_core']['voxel_count']:,}"],
        ["Enhancing tumor", f"{regions['enhancing_tumor']['volume_ml']:.3f}", f"{regions['enhancing_tumor']['voxel_count']:,}"],
        ["Peritumoral edema", f"{regions['flair_abnormality_outside_core']['volume_ml']:.3f}", f"{regions['flair_abnormality_outside_core']['voxel_count']:,}"],
    ]
    table = Table(measurement_data, colWidths=[86 * mm, 42 * mm, 42 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#15130e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdb6a8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2efe6")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            table,
            Paragraph("Processing details", section_style),
            Paragraph(
                "The four NIfTI sequences were checked for matching geometry. "
                "Non-zero intensities were normalized independently for each sequence, "
                "then the SegResNet model ran sliding-window inference on the GPU.",
                body_style,
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                f"<b>Runtime:</b> {manifest['inference']['elapsed_seconds']:.3f} seconds &nbsp;&nbsp; "
                f"<b>Hardware:</b> {html.escape(str(manifest['inference']['hardware'] or 'GPU worker'))}<br/>"
                f"<b>Mask SHA-256:</b> {html.escape(str(manifest['inference']['mask_sha256']))}",
                body_style,
            ),
            Paragraph("Developed by S M Asif Hossain", section_style),
            Paragraph("NeuroLens AI is released under the MIT License.", body_style),
        ]
    )

    def add_footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#716c61"))
        canvas.drawString(16 * mm, 8 * mm, "NeuroLens AI - Academic use only")
        canvas.drawRightString(194 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return destination


def analyze_case_archive(
    archive: str | Path,
    bundle_root: str | Path,
    work_root: str | Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Run real inference for one uploaded ZIP and return public result files."""

    archive_path = Path(archive).expanduser().resolve()
    case_id = _case_id_from_archive(archive_path)
    run_root = Path(work_root).expanduser().resolve() / case_id
    input_root = safe_extract_brats_zip(archive_path, run_root / "input")
    cases = discover_cases(input_root)
    if not cases:
        raise ValueError("No complete MRI scan set was found. T1c, T1, T2, and FLAIR are required.")
    if len(cases) > 1:
        raise ValueError("The ZIP contains more than one complete scan set. Upload one at a time.")

    case = replace(cases[0], case_id=case_id)
    validation = validate_nifti_case(case)
    run = run_monai_inference(case, bundle_root, run_root, require_cuda=True)
    modalities, prediction, spacing = load_nifti_case(case, run.mask_path)
    measurements = calculate_region_metrics(prediction, spacing, modalities["flair"] != 0)
    overlays = render_representative_overlays(modalities, prediction, run_root / "overlays")

    dice_scores = None
    if case.ground_truth:
        import nibabel as nib

        truth = np.rint(
            np.asarray(nib.as_closest_canonical(nib.load(str(case.ground_truth))).dataobj)
        ).astype(np.uint8)
        dice_scores = compute_dice_scores(prediction, truth)

    result_path = save_feasibility_result(
        run_root / "feasibility_result.json",
        case,
        validation,
        run,
        measurements,
        overlays,
        dice_scores,
    )
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    environment = collect_environment()
    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "case_id": case_id,
        "research_only": True,
        "validation": {
            "shape": validation["shape"],
            "spacing_mm": validation["spacing_mm"],
            "orientation": validation["orientation"],
            "warnings": validation["warnings"],
        },
        "inference": {
            "model_id": MODEL_ID,
            "elapsed_seconds": stored["inference"]["elapsed_seconds"],
            "peak_gpu_memory_mb": stored["inference"]["peak_gpu_memory_mb"],
            "model_sha256": stored["inference"]["model_sha256"],
            "mask_sha256": stored["inference"]["mask_sha256"],
            "hardware": environment.get("gpu"),
        },
        "measurements": measurements,
        "evaluation": {"dice": dice_scores} if dice_scores else None,
        "summary": stored["summary"],
    }
    comparison_path = Path(overlays["artifacts"]["comparison"]).resolve()
    report_path = _create_analysis_pdf(
        manifest, comparison_path, run_root / "neurolens_ai_segmentation_summary.pdf"
    )
    return manifest, comparison_path, run.mask_path.resolve(), report_path.resolve()


def build_gradio_app(bundle_root: str | Path, work_root: str | Path):
    """Create the single-case Gradio API consumed by the NeuroLens website."""

    import gradio as gr

    bundle_path = Path(bundle_root).expanduser().resolve()
    work_path = Path(work_root).expanduser().resolve()

    def analyze(case_zip: str):
        if not case_zip:
            raise gr.Error("Choose one MRI scan ZIP.")
        try:
            manifest, comparison, mask, report = analyze_case_archive(
                case_zip, bundle_path, work_path
            )
            # Gradio 6 validates file component outputs as strings rather than
            # accepting pathlib.Path instances as earlier releases did.
            return manifest, str(comparison), str(mask), str(report)
        except Exception as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title="NeuroLens AI GPU Worker") as app:
        gr.Markdown("# NeuroLens AI GPU Worker\nAcademic use only. Upload one de-identified MRI scan ZIP.")
        case_zip = gr.File(label="MRI scan ZIP", file_types=[".zip"], type="filepath")
        run_button = gr.Button("Run segmentation", variant="primary")
        manifest = gr.JSON(label="Structured result")
        comparison = gr.Image(label="Segmentation comparison", type="filepath")
        mask = gr.File(label="NIfTI segmentation mask")
        report = gr.File(label="PDF segmentation summary")
        run_button.click(
            fn=analyze,
            inputs=[case_zip],
            outputs=[manifest, comparison, mask, report],
            api_name="analyze",
            concurrency_limit=1,
        )
    return app.queue(default_concurrency_limit=1, max_size=4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the NeuroLens Gradio GPU worker.")
    parser.add_argument("--bundle-dir", default=Path("bundles"), type=Path)
    parser.add_argument("--work-root", default=Path("artifacts/worker"), type=Path)
    parser.add_argument("--share", action="store_true", help="Create a temporary public gradio.live URL.")
    args = parser.parse_args()
    bundle_root = download_monai_bundle(args.bundle_dir)
    app = build_gradio_app(bundle_root, args.work_root)
    app.launch(server_name="0.0.0.0", share=args.share, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
