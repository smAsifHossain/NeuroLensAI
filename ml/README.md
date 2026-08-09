# NeuroLens model feasibility package

This package runs NVIDIA/MONAI's pretrained `brats_mri_segmentation` bundle without training. It validates a four-modality BraTS case, performs inference, writes a NIfTI label map, calculates nested tumor-region volumes, renders representative overlays, and records reproducibility information.

It is an academic research pipeline and must not be used for diagnosis or patient care.

## Cloud execution

The supported path is the notebook at `notebooks/neurolens_brats_feasibility.ipynb`. Use a Kaggle or Colab GPU runtime and attach a registered, de-identified BraTS dataset.

The bundle expects aligned, approximately 1 mm isotropic volumes in this order:

1. T1 contrast-enhanced (`t1ce` or `t1c`)
2. T1 native (`t1` or `t1n`)
3. T2 (`t2` or `t2w`)
4. FLAIR (`flair` or `t2f`)

The filename discovery supports both legacy and modern BraTS suffixes.

## Command-line form

After installing `requirements-cloud.txt` in a CUDA-enabled environment:

```bash
PYTHONPATH=ml python -m neurolens.cli \
  --data-root /path/to/brats \
  --work-dir artifacts/feasibility \
  --bundle-dir bundles
```

The command intentionally refuses CPU inference unless `--allow-cpu` is supplied.

## Permanent Modal GPU worker

The production site has no preloaded result and connects to a stable Modal Web
Function. Authenticate the Modal CLI once, then deploy from the repository root:

```bash
modal deploy ml/modal_app.py
```

`ml/modal_app.py` pins the L4 runtime, bakes the exact MONAI checkpoint into the
container image, limits concurrency to one container, and scales to zero after
one idle minute. The URL remains stable while an idle container is asleep and
wakes automatically on the next request.

Set the site runtime variable `NEUROLENS_WORKER_URL` to the URL printed by Modal.
Users do not configure or paste worker URLs. Each request must contain one
de-identified BraTS ZIP; the worker extracts it safely, runs genuine GPU
inference, and returns that case's mask, measurements, overlay, and report.
The browser queue allows only the production NeuroLens origin and localhost;
add a new trusted origin in `ml/modal_app.py` before moving to a custom domain.

For a non-medical deployment smoke test:

```bash
python scripts/create_synthetic_brats.py artifacts/synthetic-brats-smoke.zip
pnpm verify:worker -- https://your-worker.modal.run artifacts/synthetic-brats-smoke.zip
```
