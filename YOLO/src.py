"""Shared paths + dataset check. The split already exists on disk in Data/kvasir-seg-split/{images,labels}/{train,val,test}.

python YOLO/src.py [--config Configs/default.yml]   # verify images/labels pair up and write data.yaml
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "Configs/default.yml"
SPLITS = ("train", "val", "test")


def resolve(path):
    """Config paths are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def data_yaml_path(cfg):
    """The dataset description file ultralytics reads: where the images are, which splits exist, class names."""
    return resolve(cfg["paths"]["dataset"]) / "data.yaml"


def prepare(cfg):
    """Check every image has a label file, then write data.yaml (no `path:` key, so ultralytics uses its folder)."""
    root = resolve(cfg["paths"]["dataset"])
    for split in SPLITS:
        images = {p.stem for p in (root / "images" / split).glob("*.jpg")}
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        if not images:
            raise SystemExit(f"No images in {root / 'images' / split}")
        if images != labels:
            raise SystemExit(f"{split}: {len(images - labels)} images without labels, "
                             f"{len(labels - images)} labels without images")
        print(f"[data] {split:5s}: {len(images)} images")
    data_yaml_path(cfg).write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: polyp\n")
    print("[data] ->", data_yaml_path(cfg))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare(yaml.safe_load(p.parse_args().config.read_text()))
