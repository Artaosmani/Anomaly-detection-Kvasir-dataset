"""Sample lists and tf.data pipelines for the polyp classifier.

python Classifier/data.py [--config Configs/classifier.yml]   # print split sizes without training

A sample is (path, label, box): label 1 = polyp, box = crop (y0, x0, y1, x1) as fractions, FULL = whole image.
Kvasir-SEG positives keep their train/val/test assignment; extra positives and negatives get a seeded split with
whole videos per split. An image (by hash) is used once at most.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
from PIL import Image

from common import DEFAULT_CONFIG, FULL, load_config, load_image, normalise, resolve

SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}

NEGATIVES_HELP = """No negative (no-polyp) frames found in any negatives.sources dir. Download them yourself and unpack:
  HyperKvasir, labelled images, lower GI normals only (cecum, ileum, bbps-0-1, bbps-2-3)
      https://datasets.simula.no/hyper-kvasir/     -> Data/negatives/hyperkvasir/<class>/*.jpg
  Kvasir-Capsule, labelled images, "Normal clean mucosa" (keep the <video>_<frame>.jpg filenames)
      https://datasets.simula.no/kvasir-capsule/   -> Data/negatives/kvasir-capsule/*.jpg
Do not use upper-GI classes (z-line, pylorus) as the main negatives."""


# ---------------------------------------------------------------- sample lists

def positives(cfg):
    root = resolve(cfg["paths"]["positives"])
    out = {s: [(str(p), 1, FULL) for p in sorted((root / s).glob("*.jpg"))] for s in SPLITS}
    for s in SPLITS:
        if not out[s]:
            raise SystemExit(f"No positives in {root / s}")
    return out


def image_hash(path, n=16):
    """16×16 average hash: catches the same frame exported twice (Kvasir v1 vs HyperKvasir, repeated capsule frames)."""
    g = np.asarray(Image.open(path).convert("L").resize((n, n), Image.BILINEAR), np.float32)
    return np.packbits(g > g.mean()).tobytes()


def thin_video(files, regex, stride):
    """Group frames by video and keep one per `stride` frame numbers. Returns {path: group}."""
    rx, frames, groups = re.compile(regex), defaultdict(list), {}
    for p in files:
        m = rx.match(p.stem)
        if m:
            frames[m["video"]].append((int(m["frame"]), p))
        else:
            groups[p] = str(p)  # unparsable name: its own group
    for video, items in frames.items():
        last = None
        for frame, p in sorted(items):
            if last is None or frame - last >= stride:
                groups[p], last = video, frame
    if files and not frames:
        print(f"[data] warning: video_regex matched no filenames in {files[0].parent}")
    return groups


def scan(cfg, sources, label, seen, name):
    """Scan folders, thin videos, drop images already in `seen` (hashes), split by group (seeded).

    Returns ({split: samples}, capsule paths). `seen` is updated, so later calls never reuse an image.
    """
    groups, capsule = defaultdict(list), set()
    for src in sources:
        d = resolve(src["dir"])
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT) if d.is_dir() else []
        if not files:
            print(f"[data] {name}: {src['dir']} missing or empty, skipped")
            continue
        kept = thin_video(files, src["video_regex"], src.get("stride", 1)) if src.get("video_regex") \
            else {p: str(p) for p in files}
        n = 0
        for p, g in kept.items():
            h = image_hash(p)
            if h in seen:
                continue
            seen.add(h)
            groups[f"{d}:{g}"].append(str(p))
            n += 1
            if src.get("capsule"):
                capsule.add(str(p))
        print(f"[data] {name}: {src['dir']}: {len(files)} files -> {n} kept")

    keys = sorted(groups)
    random.Random(cfg["train"]["seed"]).shuffle(keys)
    split = cfg["negatives"]["split"]
    total, ends = sum(map(len, groups.values())), np.cumsum(split) / np.sum(split)
    out, count = {s: [] for s in SPLITS}, 0
    for k in keys:
        s = SPLITS[min(int(np.searchsorted(ends, count / total, side="right")), 2)]
        out[s] += [(p, label, FULL) for p in groups[k]]
        count += len(groups[k])
    return out, capsule


def hard_negatives(cfg, train_pos):
    """Random square crops of Kvasir-SEG train images that overlap no polyp box."""
    neg = cfg["negatives"]
    boxes = json.loads(resolve(cfg["paths"]["boxes"]).read_text())
    rng, lo, hi, out = random.Random(cfg["train"]["seed"]), *neg["hard_side"], []
    for path, _, _ in train_pos:
        info = boxes.get(Path(path).stem)
        if not info:
            continue
        H, W = info["height"], info["width"]
        for _ in range(neg["hard_per_image"]):
            for _try in range(50):
                s = rng.uniform(lo, hi) * min(H, W)
                x0, y0 = rng.uniform(0, W - s), rng.uniform(0, H - s)
                if not any(x0 < b["xmax"] and x0 + s > b["xmin"] and y0 < b["ymax"] and y0 + s > b["ymin"]
                           for b in info["bbox"]):
                    out.append((path, 0, (y0 / H, x0 / W, (y0 + s) / H, (x0 + s) / W)))
                    break
    return out


def build_samples(cfg):
    """{split: samples} plus the Kvasir-Capsule eval set (empty if those frames aren't there)."""
    pos = positives(cfg)
    seen = {image_hash(p) for split in pos.values() for p, _, _ in split}  # extra images must not repeat Kvasir-SEG
    extra, _ = scan(cfg, (cfg.get("extra_positives") or {}).get("sources") or [], 1, seen, "extra positives")
    neg, capsule = scan(cfg, cfg["negatives"]["sources"], 0, seen, "negatives")
    if not any(neg.values()):
        raise SystemExit(NEGATIVES_HELP)
    samples = {s: pos[s] + extra[s] + neg[s] for s in SPLITS}
    if cfg["negatives"]["hard_negatives"]:
        hard = hard_negatives(cfg, pos["train"])
        samples["train"] += hard
        print(f"[data] hard negatives: {len(hard)} crops added to train")
    for s in SPLITS:
        n_pos = sum(y for _, y, _ in samples[s])
        print(f"[data] {s:5s}: {n_pos} polyp + {len(samples[s]) - n_pos} no-polyp")

    capsule_set = []
    cp = resolve(cfg["paths"]["capsule_polyps"])
    capsule_neg = [x for x in neg["test"] if x[0] in capsule]
    if cp.is_dir() and capsule_neg:
        capsule_set = [(str(p), 1, FULL) for p in sorted(cp.rglob("*")) if p.suffix.lower() in IMG_EXT] + capsule_neg
        print(f"[data] capsule eval: {len(capsule_set) - len(capsule_neg)} polyp + {len(capsule_neg)} no-polyp")
    return samples, capsule_set


def class_weights(samples):
    """n / (2 * n_class), so both classes weigh the same in the loss."""
    y = np.array([s[1] for s in samples])
    return [len(y) / (2 * max(np.sum(y == c), 1)) for c in (0, 1)]


# ---------------------------------------------------------------- tf.data

def augment(x, a, rotate):
    """x: size×size×3 float in 0..1. Same colour/geometry ranges as the YOLO config, no mosaic."""
    size = x.shape[0]
    if a["inset_mask"]:  # before flips/rotation: the inset sits at a fixed spot in the raw frame
        x0, y0, x1, y1 = a["inset_box"]
        m = np.ones((size, size, 1), np.float32)
        m[int(y0 * size):int(y1 * size), int(x0 * size):int(x1 * size)] = 0
        x = tf.where(tf.random.uniform([]) < a["inset_mask"], x * m, x)
    x = tf.where(tf.random.uniform([]) < a["fliplr"], tf.image.flip_left_right(x), x)
    x = tf.where(tf.random.uniform([]) < a["flipud"], tf.image.flip_up_down(x), x)
    if a["degrees"]:
        x = rotate(x[None], training=True)[0]
    if a["hsv_h"]:
        x = tf.image.random_hue(x, a["hsv_h"])
    if a["hsv_s"]:
        x = tf.image.random_saturation(x, 1 - a["hsv_s"], 1 + a["hsv_s"])
    if a["hsv_v"]:
        x = x * tf.random.uniform([], 1 - a["hsv_v"], 1 + a["hsv_v"])
    return tf.clip_by_value(x, 0.0, 1.0)


def dataset(samples, cfg, train=False, weights=None, batch=None):
    """Batches of (normalised image, label shaped like the (1,1,1) logit[, sample weight])."""
    paths, labels, boxes = zip(*samples)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.float32(labels), np.float32(boxes)))
    if train:
        ds = ds.shuffle(len(samples), seed=cfg["train"]["seed"], reshuffle_each_iteration=True)
    size, a = cfg["model"]["imgsz"], cfg["augment"]
    rotate = keras.layers.RandomRotation(a["degrees"] / 360, fill_mode="constant", seed=cfg["train"]["seed"])
    w = tf.constant(weights or [1.0, 1.0], tf.float32)

    def load(path, y, box):
        x = load_image(path, box, size)
        if train:
            x = augment(x / 255.0, a, rotate) * 255.0
        out = (normalise(x), tf.reshape(y, (1, 1, 1)))
        return out + (tf.gather(w, tf.cast(y, tf.int32)),) if weights else out

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch or cfg["train"]["batch"]).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    build_samples(load_config(args.config, args.set))
