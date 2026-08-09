<div align="center">

# NeuroLens AI

**GPU-powered brain tumor segmentation from a multimodal MRI scan ZIP.**

NeuroLens AI validates four MRI sequences, runs the SegResNet model on a managed
NVIDIA L4 GPU, and returns color-coded tumor overlays, volume measurements, a
NIfTI segmentation mask, and a downloadable PDF summary.

<p>
  <a href="https://smasifhossain.github.io/NeuroLensAI/"><img alt="NeuroLens AI live application" src="https://img.shields.io/badge/Live%20app-Open-c7401e?style=flat&amp;logo=googlechrome&amp;logoColor=white&amp;labelColor=15130e"></a>
  <a href="https://huggingface.co/MONAI/brats_mri_segmentation"><img alt="SegResNet model" src="https://img.shields.io/badge/Model-SegResNet-8664b5?style=flat&amp;logo=pytorch&amp;logoColor=white&amp;labelColor=15130e"></a>
  <a href="ml/modal_app.py"><img alt="Modal NVIDIA L4 worker" src="https://img.shields.io/badge/GPU-Modal%20L4-55b98e?style=flat&amp;logo=nvidia&amp;logoColor=white&amp;labelColor=15130e"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-16a34a?style=flat&amp;logo=opensourceinitiative&amp;logoColor=white&amp;labelColor=15130e"></a>
</p>

[Live application](https://smasifhossain.github.io/NeuroLensAI/) | [Model benchmark](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md#performance) | [Feasibility run](docs/FEASIBILITY_RUN.md) | [GPU worker](ml/README.md)

If NeuroLens AI is useful to you, consider [starring the repository](https://github.com/smAsifHossain/NeuroLensAI). It helps others discover and support the project.

</div>

---

## What NeuroLens AI does

NeuroLens AI is an academic brain tumor segmentation application for four-sequence
brain MRI studies. A user uploads one ZIP containing T1 contrast-enhanced, native
T1, T2, and FLAIR NIfTI volumes. The browser transfers the archive directly to a
managed GPU service, where the files are validated, normalized, and processed by
the SegResNet model.

The result includes predicted-mask overlays for every MRI sequence, tumor-region
volumes, model and runtime provenance, a downloadable NIfTI label map, and a
measurement-grounded PDF summary. The PDF is generated deterministically from the
mask measurements; no vision-language model is used.

NeuroLens AI is an academic prototype, not diagnostic software, a medical device,
or a clinical radiology reporting system.

## Main features

- One clear upload-to-segmentation workflow with no demonstration result.
- Direct browser transfer to a stable Modal endpoint backed by an NVIDIA L4 GPU.
- Safe ZIP extraction with traversal, symbolic-link, member-count, and size checks.
- T1c, T1, T2, and FLAIR discovery across common NIfTI naming conventions.
- Shape, voxel-spacing, orientation, and affine-geometry validation.
- Channel-wise normalization of non-zero MRI intensities.
- Sliding-window SegResNet inference using the pinned model checkpoint.
- Predicted-mask views for the original scan and all four MRI sequences.
- Whole-tumor, tumor-core, enhancing-tumor, and peritumoral-edema measurements.
- Downloadable NIfTI mask and PDF segmentation summary.
- Model version, checkpoint hash, mask hash, runtime, GPU, and memory provenance.
- Interactive model architecture and published benchmark sections.
- Responsive, keyboard-accessible interface with GitHub Pages deployment.

## How an MRI study is analyzed

1. **Validate the archive.** The worker accepts one ZIP, rejects unsafe paths and
   oversized content, and looks for one complete four-sequence MRI study.
2. **Check the MRI volumes.** T1c, T1, T2, and FLAIR files must have matching
   dimensions, voxel spacing, orientation, and affine geometry.
3. **Normalize each sequence.** Non-zero intensities are normalized independently
   for each MRI channel.
4. **Run SegResNet.** The model receives the four volumes as a 3D, four-channel
   input and performs GPU sliding-window inference.
5. **Create the label map.** Tumor-core, whole-tumor, and enhancing-tumor
   probability channels are thresholded and converted into one NIfTI mask.
6. **Build the results.** NeuroLens AI renders the overlays, calculates regional
   volumes, records provenance, and creates the PDF summary.

## Model

The deployed model is SegResNet from the
[`brats_mri_segmentation` bundle](https://huggingface.co/MONAI/brats_mri_segmentation),
version `0.5.4`, pinned to revision
`370f7f9d062745fbac445e7fe6d6616d35df04ec`. The checkpoint was trained for
adult glioma segmentation using BraTS 2018 data.

The bundle authors publish the following validation Dice scores for the checkpoint
used by NeuroLens AI:

| Region | Dice score |
| --- | ---: |
| Average across three target regions | 0.8518 |
| Whole tumor | 0.9026 |
| Tumor core | 0.8559 |
| Enhancing tumor | 0.7905 |

The published split contains 285 studies: 200 for training, 42 for validation, and
43 for held-out testing. These are model-author benchmark results, not a score for
an individual uploaded scan. The complete benchmark context is available in the
[bundle documentation](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md#performance).

## Supported MRI study

Upload one ZIP containing one complete four-sequence NIfTI study. Files may be in
the archive root or a case folder.

| MRI sequence | Common accepted suffixes | Purpose |
| --- | --- | --- |
| T1 contrast-enhanced | `t1c`, `t1ce` | Highlights contrast-enhancing tissue |
| T1 native | `t1`, `t1n` | Provides anatomical T1 information |
| T2 | `t2`, `t2w` | Shows fluid-sensitive tissue contrast |
| FLAIR | `flair`, `t2f` | Highlights edema and other fluid-related abnormality |

Each file must be a `.nii` or `.nii.gz` volume. The uncompressed archive is
limited to 1.5 GB and 64 files. Upload only de-identified data that you are
authorized to process.

## Architecture

```text
Browser
  |
  | GitHub Pages
  v
React frontend
  |
  | direct HTTPS upload
  v
Modal web endpoint
  |
  | wakes on demand
  v
NVIDIA L4 container
  |
  v
ZIP validation -> MRI normalization -> SegResNet inference
  |
  v
Overlays -> volume measurements -> NIfTI mask -> PDF summary
```

GitHub Pages hosts only the static interface. MRI archives are sent directly to
the Modal service and do not pass through GitHub. Modal keeps at most one GPU
container active and scales it to zero after one idle minute; the public endpoint
remains stable and wakes on the next request.

## Run locally

Requires Node.js 22.13 or newer and pnpm 11.

```powershell
git clone https://github.com/smAsifHossain/NeuroLensAI.git
cd NeuroLensAI
pnpm install
Copy-Item .env.example .env.local
```

Set the stable Modal worker endpoint in `.env.local`:

```env
NEUROLENS_WORKER_URL=https://your-worker.modal.run
```

Start the local application:

```powershell
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The local worker configuration
route supplies the endpoint to the browser automatically.

## Test with an MRI study

Use a de-identified four-sequence MRI study that you are authorized to analyze.
Place the T1c, T1, T2, and FLAIR NIfTI files in one ZIP, open the application,
choose the archive, and select **Run brain tumor segmentation**.

A non-medical synthetic archive for connection testing can be created with:

```powershell
python scripts/create_synthetic_brats.py artifacts/synthetic-brats-smoke.zip
```

Verify the deployed worker with:

```powershell
pnpm verify:worker -- https://your-worker.modal.run artifacts/synthetic-brats-smoke.zip
```

Synthetic data confirms the upload and service contract only. It does not provide
a meaningful segmentation-performance result.

## Deploy the frontend from GitHub

The workflow in `.github/workflows/deploy-pages.yml` builds and publishes the
frontend whenever `main` is updated. It uses GitHub Pages and does not require a
separate frontend server.

```text
Push to main
  -> install the dedicated static-frontend dependencies
  -> build the static NeuroLens AI interface
  -> upload the Pages artifact
  -> publish the GitHub Pages site
```

The public worker URL is stored in `public/worker-config.json`. It is not a secret:
the browser must know the endpoint to upload a study. Access is restricted by the
worker's CORS allowlist.

## Deploy the GPU worker

Authenticate the Modal CLI once, then deploy from the repository root:

```powershell
modal deploy ml/modal_app.py
```

The deployment builds an immutable Python 3.11 image with the pinned model bundle,
PyTorch, MONAI, NIfTI processing, Gradio, and PDF generation dependencies. The
worker uses an NVIDIA L4, permits one active container, and has a 30-minute request
timeout for large 3D studies.

If the Modal URL changes, update `public/worker-config.json` and
`NEUROLENS_WORKER_URL` in the local or hosted environment.

## Tests

Run the frontend checks:

```powershell
pnpm lint
pnpm test
pnpm build:pages
```

Run the Python model and worker checks from a compatible Python environment with
the package dependencies installed:

```powershell
$env:PYTHONPATH=(Resolve-Path 'ml').Path
python -m unittest discover -s tests_ml -v
```

The test suites cover server rendering, upload-driven result visibility, managed
worker configuration, ZIP safety, modality discovery, geometry validation, tumor
measurements, overlays, PDF output, and inference manifest behavior.

## Repository layout

```text
.github/workflows/    GitHub Pages deployment
.openai/              Existing Sites hosting metadata
app/                  NeuroLens AI interface and local worker-config route
docs/                 Feasibility evidence and research notes
github-pages/         Static GitHub Pages entry point
ml/                   SegResNet pipeline, Gradio API, PDF output, and Modal app
notebooks/            Reproducible Kaggle and Colab feasibility notebook
public/               Icons, social preview, and public worker configuration
scripts/              Worker verification and synthetic-data utilities
tests/                Frontend and server-route checks
tests_ml/             MRI pipeline and worker tests
package.json          Frontend scripts and dependency definitions
package.pages.json    Minimal dependency manifest for GitHub Pages
vite.github.config.ts GitHub Pages build configuration
```

Generated MRI files, model bundles, local environments, build output, and secrets
are excluded from Git.

## Privacy and academic scope

The application has no user accounts, scan history, advertising, or analytics.
Uploads are processed inside an ephemeral Modal container. The public interface
asks users to upload only de-identified MRI data that they have permission to use.

Every result is derived from the uploaded scan and the pinned model checkpoint.
NeuroLens AI does not provide a diagnosis, treatment recommendation, prognosis, or
clinical radiology report. Results require review by appropriately qualified
medical and research professionals.

## Known limitations

- The model was trained for the BraTS adult-glioma task and is not validated for
  every tumor type, scanner, acquisition protocol, institution, or patient group.
- Four compatible MRI sequences are required; a single image or screenshot is not
  a valid input.
- Model-author benchmark scores do not establish clinical performance on new data.
- Tumor volumes depend on the predicted mask and input voxel spacing.
- GPU cold starts, upload bandwidth, and large NIfTI files can increase total wait
  time.
- The generated PDF is a structured academic summary, not a diagnostic report.
- No external clinical validation or regulatory review has been completed.

## Data and model references

- [SegResNet brain tumor segmentation bundle](https://huggingface.co/MONAI/brats_mri_segmentation)
- [MONAI model-zoo source](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation)
- [BraTS 2023 adult-glioma challenge workspace](https://www.synapse.org/Synapse:syn51156910)
- [Feasibility run and reproduced case evidence](docs/FEASIBILITY_RUN.md)

## Author

Developed by [S M Asif Hossain](https://www.linkedin.com/in/smasifhossain).

## License

NeuroLens AI is available under the [MIT License](LICENSE).
