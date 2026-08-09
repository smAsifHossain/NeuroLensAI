"""Create a non-medical BraTS-shaped ZIP for deployment smoke tests.

The volumes contain only zeros. They verify file transfer, NIfTI parsing, CUDA
execution, and artifact delivery without uploading human imaging data.
"""

from __future__ import annotations

import argparse
import gzip
import struct
import zipfile
from pathlib import Path

SHAPE = (240, 240, 155)
MODALITIES = ("t1c", "t1", "t2", "flair")


def write_zero_nifti(path: Path) -> None:
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *SHAPE, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)  # float32
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 1, 1, 1, 1, 1, 1, 1, 1)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<h", header, 254, 1)  # sform code
    struct.pack_into("<4f", header, 280, 1, 0, 0, 0)
    struct.pack_into("<4f", header, 296, 0, 1, 0, 0)
    struct.pack_into("<4f", header, 312, 0, 0, 1, 0)
    header[344:348] = b"n+1\0"

    zero_chunk = bytes(1024 * 1024)
    remaining = SHAPE[0] * SHAPE[1] * SHAPE[2] * 4
    with gzip.open(path, "wb", compresslevel=6) as stream:
        stream.write(header)
        stream.write(bytes(4))
        while remaining:
            chunk_size = min(remaining, len(zero_chunk))
            stream.write(zero_chunk[:chunk_size])
            remaining -= chunk_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    source = destination.parent / "synthetic-brats-smoke"
    source.mkdir(parents=True, exist_ok=True)

    for modality in MODALITIES:
        write_zero_nifti(source / f"synthetic-smoke-{modality}.nii.gz")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in source.iterdir():
            archive.write(path, arcname=f"synthetic-smoke/{path.name}")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
