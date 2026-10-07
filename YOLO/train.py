"""Train YOLO26 for polyp detection on Kvasir-SEG.

Examples:
    python YOLO/train.py                                   # use Configs/default.yml
    python YOLO/train.py --model yolo26s.pt --epochs 200
    python YOLO/train.py --set lr0=0.005 mosaic=0.5 cache=false
    python YOLO/train.py --resume runs/yolo26_kvasir/weights/last.pt
    python YOLO/train.py --dry-run                         # print resolved config and exit
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import DEFAULT_CONFIG, data_yaml_path, prepare, resolve  # noqa: E402

TRAIN_SECTIONS = ("train", "augment", "output")


# ---------------------------------------------------------------- config

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    p.add_argument("--model", help="override model (e.g. yolo26s.pt)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", help="auto | cpu | mps | 0 | 0,1")
    p.add_argument("--name", help="run name under output.project")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any train/augment/output key, e.g. --set lr0=0.005 mosaic=0")
    p.add_argument("--resume", type=Path, help="resume from a last.pt checkpoint")
    p.add_argument("--no-eval", action="store_true", help="skip test-split evaluation")
    p.add_argument("--dry-run", action="store_true", help="print resolved config and exit")
    return p.parse_args()


def load_config(args):
    cfg = yaml.safe_load(args.config.read_text())
    if args.model:
        cfg["model"] = args.model
    for key in ("epochs", "batch", "imgsz", "device"):
        if getattr(args, key) is not None:
            cfg["train"][key] = getattr(args, key)
    if args.name:
        cfg["output"]["name"] = args.name
    for item in args.set:
        key, _, raw = item.partition("=")
        section = next((s for s in TRAIN_SECTIONS if key in cfg[s]), "train")
        cfg[section][key] = yaml.safe_load(raw)  # "0.005" -> float, "false" -> bool, "null" -> None
    return cfg


def train_kwargs(cfg):
    """Flatten train/augment/output sections into ultralytics kwargs and validate key names."""
    kwargs = {k: v for s in TRAIN_SECTIONS for k, v in cfg[s].items()}
    unknown = sorted(set(kwargs) - set(DEFAULT_CFG_DICT))
    if unknown:
        raise SystemExit(f"Unknown ultralytics training args in config: {unknown}")
    kwargs["data"] = str(data_yaml_path(cfg))
    kwargs["project"] = str(resolve(kwargs["project"]))
    kwargs["device"] = resolve_device(kwargs.get("device", "auto"))
    return kwargs


def resolve_device(device):
    if device not in (None, "auto"):
        return device
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- setup

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dataset(cfg):
    data_yaml = data_yaml_path(cfg)
    if not data_yaml.exists():
        print(f"[data] {data_yaml} missing, preparing dataset...")
        prepare(cfg)
    data = yaml.safe_load(data_yaml.read_text())
    for split in ("train", "val", "test"):
        n = len(list((Path(data["path"]) / data[split]).glob("*.jpg")))
        print(f"[data] {split:5s}: {n} images")
        if split != "test" and n == 0:
            raise SystemExit(f"No images found for split '{split}' — re-run `python YOLO/src.py`")


def print_config(cfg, kwargs):
    print("[config] model:", cfg["model"])
    width = max(map(len, kwargs))
    for k, v in kwargs.items():
        print(f"  {k:<{width}} : {v}")


# ---------------------------------------------------------------- run

def train(cfg, kwargs, resume=None):
    if resume:
        print(f"[train] resuming from {resume}")
        model = YOLO(str(resume))
        model.train(resume=True)
    else:
        model = YOLO(cfg["model"])
        model.train(**kwargs)
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "config_resolved.yaml").write_text(yaml.safe_dump({**cfg, "resolved_train_kwargs": kwargs}, sort_keys=False))
    return save_dir, Path(model.trainer.best)


def evaluate(cfg, kwargs, weights, save_dir):
    ev = cfg.get("evaluate") or {}
    if not ev.get("split"):
        return None
    print(f"[eval] {weights} on '{ev['split']}' split")
    m = YOLO(str(weights)).val(data=kwargs["data"], split=ev["split"], imgsz=kwargs["imgsz"],
                               batch=kwargs["batch"] if kwargs["batch"] > 0 else 16,
                               conf=ev.get("conf", 0.001), iou=ev.get("iou", 0.6), device=kwargs["device"],
                               project=str(save_dir), name=f"eval_{ev['split']}", plots=True)
    metrics = {
        "split": ev["split"],
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "fitness": float(m.fitness),
    }
    (save_dir / f"metrics_{ev['split']}.json").write_text(json.dumps(metrics, indent=2))
    print("[eval]", json.dumps(metrics, indent=2))
    return metrics


def export(cfg, weights):
    fmt = (cfg.get("export") or {}).get("format")
    if fmt:
        print(f"[export] {weights} -> {fmt}")
        YOLO(str(weights)).export(format=fmt)


def main():
    args = parse_args()
    cfg = load_config(args)
    kwargs = train_kwargs(cfg)
    print_config(cfg, kwargs)
    if args.dry_run:
        return

    seed_everything(kwargs.get("seed", 0))
    ensure_dataset(cfg)
    save_dir, best = train(cfg, kwargs, resume=args.resume)
    print(f"[train] done, best weights: {best}")

    if not args.no_eval:
        evaluate(cfg, kwargs, best, save_dir)
    export(cfg, best)


if __name__ == "__main__":
    main()
