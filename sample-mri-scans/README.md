# Sample multimodal MRI scans

These five de-identified academic cases let users run the complete NeuroLens AI
workflow without preparing their own data. Download one ZIP, return to the
[NeuroLens AI application](https://smasifhossain.github.io/NeuroLensAI/), choose
the downloaded archive, and run the brain tumor segmentation.

| MRI case | Size | Download |
| --- | ---: | --- |
| BraTS-GLI-00002-000 | 8.7 MB | [Download ZIP](./BraTS-GLI-00002-000.zip?raw=1) |
| BraTS-GLI-00003-000 | 10.0 MB | [Download ZIP](./BraTS-GLI-00003-000.zip?raw=1) |
| BraTS-GLI-00009-000 | 8.1 MB | [Download ZIP](./BraTS-GLI-00009-000.zip?raw=1) |
| BraTS-GLI-00012-000 | 10.1 MB | [Download ZIP](./BraTS-GLI-00012-000.zip?raw=1) |
| BraTS-GLI-00018-000 | 9.3 MB | [Download ZIP](./BraTS-GLI-00018-000.zip?raw=1) |

Each ZIP contains T1 contrast-enhanced (`t1c`), native T1 (`t1n`), T2 (`t2w`),
and FLAIR (`t2f`) NIfTI volumes. A reference segmentation (`seg`) is also present
for dataset completeness. NeuroLens AI does not use that reference mask as a
model input; SegResNet produces a new predicted mask from the four MRI volumes.

## Data terms and attribution

The MRI archives are BraTS 2023 Adult Glioma data obtained through Synapse ID
[`syn51156910`](https://www.synapse.org/Synapse:syn51156910/wiki/627000). They
are provided for non-commercial academic and educational use under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
They are separate dataset materials and are not covered by the repository's MIT
software license.

Required attribution statement:

> Data used in this project were obtained as part of the Brain Tumor Segmentation
> (BraTS) Challenge project through Synapse ID: syn51156910.

Users of these data should follow the official BraTS post-challenge terms and
cite the manuscripts listed on the [BraTS data-access page](https://www.synapse.org/Synapse:syn51156910/wiki/627000),
including:

1. U. Baid et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
   Segmentation and Radiogenomic Classification,” arXiv:2107.02314, 2021.
2. B. H. Menze et al., “The Multimodal Brain Tumor Image Segmentation Benchmark
   (BRATS),” *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024, 2015.
3. S. Bakas et al., “Advancing The Cancer Genome Atlas glioma MRI collections
   with expert segmentation labels and radiomic features,” *Scientific Data*,
   4:170117, 2017.

Do not attempt to re-identify any individual represented in the data.

## SHA-256 checksums

```text
6D70B196640D136F43B7C2A3C3B4739986A8BC619CB49E4DB8B791BC96050AA1  BraTS-GLI-00002-000.zip
5169C65FFFABA1BFD0A0DDDDC225742BD8F0BE51E624B210178C2E5E97058E9A  BraTS-GLI-00003-000.zip
F9BB506DCFF76AECD6B404C485A34E3F318CA763D0BBEF8DC5F9329BF9A14E0E  BraTS-GLI-00009-000.zip
DD47083661E70C5DA53207F0E88CA800FE9D0CD3C124528C229ACA526140EBC4  BraTS-GLI-00012-000.zip
C21C56D84B2B498CBE3BDB81B8A65A7D23B0ACD20BE587536E9D3CEC73E2300F  BraTS-GLI-00018-000.zip
```
