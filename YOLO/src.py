"""Kvasir-SEG -> YOLO dataset. Training lives in train.py. Paths come from Configs/default.yml.

python YOLO/src.py [--config Configs/default.yml]
"""
import argparse
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "Configs/default.yml"


def resolve(path):
    """Config paths are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def data_yaml_path(cfg):
    """The dataset description file ultralytics reads: where the images are, which splits exist, class names."""
    return resolve(cfg["paths"]["dataset"]) / "data.yaml"


def to_yolo(entry):
    w, h = entry["width"], entry["height"]
    lines = []
    for b in entry["bbox"]:
        x0, y0, x1, y1 = max(b["xmin"], 0), max(b["ymin"], 0), min(b["xmax"], w), min(b["ymax"], h)
        if x1 > x0 and y1 > y0:
            lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return "\n".join(lines)


def prepare(cfg):
    images, out = resolve(cfg["paths"]["images"]), resolve(cfg["paths"]["dataset"])
    ann = json.loads(resolve(cfg["paths"]["labels"]).read_text())
    ids = sorted(ann)
    random.Random(cfg["split"]["seed"]).shuffle(ids)
    n_train = int(cfg["split"]["train"] * len(ids))
    n_val = int(cfg["split"]["val"] * len(ids))
    splits = {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in names:
            link = out / "images" / split / f"{i}.jpg"
            if not link.exists():
                link.symlink_to(images / f"{i}.jpg")
            (out / "labels" / split / f"{i}.txt").write_text(to_yolo(ann[i]))
    data_yaml_path(cfg).write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: polyp\n")
    print({k: len(v) for k, v in splits.items()}, "->", data_yaml_path(cfg))


if __name__ == "__main__":
    assert to_yolo({"width": 100, "height": 50, "bbox": [{"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50}]}) == "0 0.250000 0.500000 0.500000 1.000000"
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare(yaml.safe_load(p.parse_args().config.read_text()))
