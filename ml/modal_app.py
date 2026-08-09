"""Permanent Modal deployment for the NeuroLens BraTS GPU worker.

Deploy from the repository root with::

    modal deploy ml/modal_app.py

The MONAI bundle is pinned and cached in the container image during the build,
so every deployed container starts with the same pretrained checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

APP_NAME = "neurolens-brats-worker"
BUNDLE_ROOT = Path("/opt/neurolens/bundles/brats_mri_segmentation")
RUN_ROOT = Path("/tmp/neurolens-runs")


def _cache_model_bundle() -> None:
    """Bake the exact pretrained MONAI bundle into the immutable image."""

    sys.path.insert(0, "/root")
    from neurolens.brats import download_monai_bundle

    download_monai_bundle(BUNDLE_ROOT.parent)


runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.7.1",
        "monai==1.5.2",
        "nibabel==5.2.1",
        "pytorch-ignite==0.4.11",
        "fire>=0.5,<1",
        "huggingface_hub>=0.23,<2",
        "pillow>=10,<13",
        "reportlab>=4,<5",
        "gradio>=5,<7",
        "fastapi>=0.115,<1",
    )
    .add_local_dir("ml/neurolens", "/root/neurolens", copy=True)
    .run_function(_cache_model_bundle, timeout=1800)
)

app = modal.App(APP_NAME)


@app.function(
    image=runtime_image,
    gpu="L4",
    timeout=1800,
    startup_timeout=1800,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
)
@modal.asgi_app()
def web():
    """Serve the existing named Gradio API at a stable Modal URL."""

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import gradio as gr

    sys.path.insert(0, "/root")
    from neurolens.worker import build_gradio_app

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    api = FastAPI(title="NeuroLens BraTS Worker", docs_url=None, redoc_url=None)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://neurolens-lab.covalent-con-7153.chatgpt.site",
            "https://smasifhossain.github.io",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        # Gradio's browser queue uses credentialed requests for its session
        # stream, so an explicit origin allowlist must opt into credentials.
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.get("/healthz")
    def health() -> dict[str, object]:
        import torch

        return {
            "service": APP_NAME,
            "ready": bool(torch.cuda.is_available() and BUNDLE_ROOT.exists()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    gradio_app = build_gradio_app(BUNDLE_ROOT, RUN_ROOT)
    return gr.mount_gradio_app(api, gradio_app, path="/")
