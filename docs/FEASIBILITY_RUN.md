# NeuroLens BraTS feasibility checkpoint

This checkpoint answers one question: can the project run a real, pretrained brain-tumor segmentation model on a genuine BraTS case, without training, on a free cloud GPU?

The implementation uses NVIDIA/MONAI's `brats_mri_segmentation` bundle v0.5.4, pinned to Hugging Face revision `370f7f9d062745fbac445e7fe6d6616d35df04ec`. The model was trained on BraTS 2018 adult glioma data and expects four aligned 3D NIfTI volumes in the fixed order T1 contrast-enhanced, T1 native, T2, and FLAIR. It predicts the BraTS label map used for whole tumor, tumor core, and enhancing tumor measurements. Modern adult-glioma ground-truth ET label 3 is normalized to the model's legacy ET label 4 before evaluation.

## Verified Kaggle checkpoint

Private Kaggle Version 1, `BraTS case 00002 feasibility`, completed successfully on August 7, 2026 using a Tesla T4. The input was the official `BraTS-GLI-00002-000` case with shape `240 × 240 × 155`, 1 mm isotropic spacing, matched affines, and all four required modalities.

- End-to-end inference wrapper time: 22.589 seconds
- Peak observed GPU memory: 11,865 MB
- Whole-tumor prediction: 108.551 mL
- Tumor-core prediction: 25.829 mL
- Enhancing-tumor prediction: 12.827 mL
- Dice: whole tumor 0.71055, tumor core 0.84264, enhancing tumor 0.66561

These values document technical feasibility on one source-domain case. They are not clinical performance claims.

## Run it

1. Open `notebooks/neurolens_brats_feasibility.ipynb` in Kaggle or Colab.
2. Select a compatible CUDA GPU runtime and enable Internet access. Kaggle's current PyTorch image supports T4 but not P100.
3. Attach a legally obtained, de-identified BraTS adult-glioma dataset and this repository.
4. If automatic discovery does not find them, set `NEUROLENS_PROJECT_ROOT` and `NEUROLENS_DATA_ROOT` in the notebook environment.
5. Run cells from top to bottom once.
6. Download the generated case ZIP and preserve it as experiment evidence.

BraTS access is intentionally not automated by this repository. Dataset registration, license acceptance, and de-identification remain the researcher's responsibility.

## Evidence produced

Each run creates:

- the model's `*_seg.nii.gz` label map;
- original/overlay PNGs and a side-by-side comparison;
- whole-tumor, tumor-core, enhancing-tumor, and label-2 volumes in mL;
- an approximate canonical-RAS laterality summary;
- elapsed inference time, observed peak GPU memory, and child-process memory;
- software/model/GPU provenance and SHA-256 hashes for the model, inputs, and output mask;
- regional Dice scores when a reference segmentation exists;
- a deterministic research summary grounded only in calculated measurements.

## Acceptance gate

The first milestone is complete only after a cloud execution on one genuine case produces a readable, non-empty mask, a visible comparison image, and a populated `feasibility_result.json`. If a ground-truth label is available, preserve the Dice values as well. The notebook must then be stopped for review before adding MedGemma or connecting results to the public interface.

## Scope and limitations

- No training, fine-tuning, diagnosis, or clinical validation is performed.
- The model's source-domain results do not establish performance on a new dataset.
- The laptop is suitable for development but not practical 3D inference; use a Kaggle/Colab CUDA runtime.
- The deterministic summary is not a radiology report. A future VLM layer must remain measurement-grounded and clearly labeled as research-only.

## Primary references

- [MONAI model-zoo bundle configuration](https://github.com/Project-MONAI/model-zoo/tree/dev/models/brats_mri_segmentation)
- [MONAI Bundle command-line documentation](https://docs.monai.io/en/stable/bundle.html)
- [BraTS 2023 adult-glioma challenge workspace](https://www.synapse.org/Synapse:syn51156910)
