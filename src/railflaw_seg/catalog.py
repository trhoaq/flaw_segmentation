from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    dataset: str
    key: str
    image: Path
    mask: Path
    encoding: str
    group: str
    synthetic: bool = False
    combined: bool = True


def indexed(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    output = {}
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in suffixes or p.name.startswith("._"):
            continue
        key = p.stem.casefold()
        if key in output:
            raise ValueError(f"Duplicate case-insensitive ID: {p}")
        output[key] = p
    return output


def split_for_group(group: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Ratios must leave nonempty train, validation and test fractions")
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
    return "train" if value < train_ratio else "val" if value < train_ratio + val_ratio else "test"


def rsdds_group(kind: int, image_id: str) -> str:
    number = re.fullmatch(r"rail_(\d+)", image_id, re.IGNORECASE)
    if number is None:
        raise ValueError(f"Unrecognized RSDDs frame ID: {image_id}")
    return f"rsdds:{kind}:rail_{int(number[1])}"


def cgan_group(image_id: str) -> str:
    match = re.match(r"type_([12])_cropped_rail_(\d+)(?:_|$)", image_id, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot map CGAN derivative to source frame: {image_id}")
    return f"rsdds:{int(match[1])}:rail_{int(match[2])}"


def nrsd_group(image_id: str) -> str:
    match = re.fullmatch(r"(.+)_\d+_\d+", image_id)
    return "nrsd:" + (match[1] if match else image_id)


def discover(root: Path, convert_nrsd_original: bool = True, include_coarse_rail5k: bool = False) -> tuple[list[Sample], list[dict]]:
    root = root.resolve()
    samples, omitted = [], []

    def pairs(dataset, image_dir, mask_dir, encoding, make_group, *, synthetic=False, combined=True, mask_key=None):
        images = indexed(root / image_dir, IMAGE_SUFFIXES)
        masks = indexed(root / mask_dir, IMAGE_SUFFIXES)
        if mask_key:
            masks = {mask_key(key): path for key, path in masks.items()}
        for key, image in images.items():
            if key not in masks:
                omitted.append({"dataset": dataset, "image": str(image), "reason": "missing_segmentation_mask"})
                continue
            samples.append(Sample(dataset, image.stem, image, masks[key], encoding, make_group(key), synthetic, combined))
        extras = masks.keys() - images.keys()
        if extras:
            omitted.append({"dataset": dataset, "reason": "mask_without_source_image", "count": len(extras)})

    for kind, name in [(1, "Type-I RSDDs dataset"), (2, "Type-II RSDDs dataset")]:
        pairs(f"RSDDs_Type_{kind}", f"RSDDs/{name}/Rail surface images", f"RSDDs/{name}/GroundTruth", "binary_255",
              lambda key, k=kind: rsdds_group(k, key))
    for variant in (["NRSD-relabel", "NRSD"] if convert_nrsd_original else ["NRSD-relabel"]):
        pairs("NRSD_relabel" if variant == "NRSD-relabel" else "NRSD_original",
              f"NRSD-MN/data/{variant}/JPEGImages", f"NRSD-MN/data/{variant}/SegmentationClass", "indexed", nrsd_group,
              combined=variant == "NRSD-relabel")
    pairs("NEU_RSDDS_113", "NEU_RSDDS_113/data/INPUT_IMG", "NEU_RSDDS_113/data/Ground Truth", "binary", lambda k: f"neu:{k}")
    pairs("CGAN", "CGAN_Rail_Surface_Defects/data/images", "CGAN_Rail_Surface_Defects/data/labels", "red_green", cgan_group, synthetic=True)
    pairs("Rail5k_sample", "Rail-5k/sample/Rail-5k_sample/JPEGImages", "Rail-5k/sample/Rail-5k_sample/SegmentationClass",
          "binary_255", lambda k: f"rail5k:{k}", combined=include_coarse_rail5k, mask_key=lambda k: k.removesuffix("_lw"))
    raw = root / "Rail-5k/sample/Rail-5k_sample/RawImages"
    if raw.exists():
        omitted.append({"dataset": "Rail5k_sample", "reason": "unlabeled_raw_images", "count": len(indexed(raw, IMAGE_SUFFIXES))})
    if len({(s.dataset, s.key.casefold()) for s in samples}) != len(samples):
        raise ValueError("Duplicate output image IDs")
    return samples, omitted
