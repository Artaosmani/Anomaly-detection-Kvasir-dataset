"""Shared bits for the polyp classifier: config, image loading (the same for training, calibration and
run_csv.py), the INT8 .tflite runner and the metrics."""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # QAT (tensorflow_model_optimization) needs Keras 2 = tf_keras
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from YOLO.src import ROOT, resolve  # noqa: E402,F401

DEFAULT_CONFIG = ROOT / "Configs/classifier.yml"
FULL = (0.0, 0.0, 1.0, 1.0)  # crop box (y0, x0, y1, x1) as fractions: the whole image


# ---------------------------------------------------------------- config

def load_config(path, sets=()):
    """Read the YAML config and apply `--set key=value` (key = a section key like imgsz, or a dotted path like stage2.lr)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in sets:
        key, _, raw = item.partition("=")
        *parents, leaf = key.split(".")
        if parents and parents[0] not in cfg:  # stage2.lr -> train.stage2.lr
            parents = [s for s, v in cfg.items() if isinstance(v, dict) and parents[0] in v][:1] + parents
        if parents:
            node = cfg
            for p in parents:
                node = node.get(p) if isinstance(node, dict) else None
        else:
            node = next((s for s in cfg.values() if isinstance(s, dict) and leaf in s), None)
        if not isinstance(node, dict) or leaf not in node:
            raise SystemExit(f"Unknown config key: {key}")
        node[leaf] = yaml.safe_load(raw)  # "0.005" -> float, "false" -> bool, "null" -> None
    return cfg


# ---------------------------------------------------------------- images

def load_image(path, box, size):
    """Decode, crop `box`, squash to size×size with an area (box-filter) resize. Returns float32 RGB in 0..255.

    Board frames are already square (336×336); Kvasir images are not, so they get squashed rather than cropped
    to keep the polyp in view. 336 -> 84 is an exact 4×4 block mean, which a 1/4-scale JPEG decode approximates.
    """
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    h, w = tf.cast(tf.shape(img)[0], tf.float32), tf.cast(tf.shape(img)[1], tf.float32)
    y0, x0 = tf.cast(box[0] * h, tf.int32), tf.cast(box[1] * w, tf.int32)
    y1, x1 = tf.cast(box[2] * h, tf.int32), tf.cast(box[3] * w, tf.int32)
    return tf.image.resize(tf.cast(img[y0:y1, x0:x1], tf.float32), (size, size), method="area")


def normalise(x):
    """0..255 -> -1..1, the range the MobileNet ImageNet weights expect."""
    return x / 127.5 - 1.0


# ---------------------------------------------------------------- INT8 model

class Int8Model:
    """Runs model_int8.tflite one frame at a time, the way the board will."""

    def __init__(self, path, threads=1):
        self.it = tf.lite.Interpreter(model_path=str(path), num_threads=threads,  # reference int8 kernels, no XNNPACK
                                      experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]
        self.size = int(self.inp["shape"][1])

    def quantise(self, x):
        """Normalised float (-1..1) -> int8 input."""
        return np.clip(np.round(x / self.in_scale + self.in_zp), -128, 127).astype(np.int8)

    def __call__(self, x):
        """One normalised H×W×3 image -> raw int8 output (the logit, quantised)."""
        self.it.set_tensor(self.inp["index"], self.quantise(np.asarray(x))[None])
        self.it.invoke()
        return int(self.it.get_tensor(self.out["index"]).ravel()[0])

    def logit(self, raw):
        return (np.asarray(raw, np.float32) - self.out_zp) * self.out_scale


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


# ---------------------------------------------------------------- metrics

def pick_threshold(y, score, target_recall):
    """Threshold with recall >= target on (y, score), then lowered as far as it goes without adding a false positive
    (a missed polyp costs more than an extra YOLO run). Works for float logits and raw int8 alike."""
    y, score = np.asarray(y), np.asarray(score)
    pos = np.sort(score[y == 1])
    if not len(pos):
        raise SystemExit("No positives to pick a threshold on")
    t = pos[int(np.floor((1 - target_recall) * len(pos)))]  # at most (1-target) of positives fall below
    below = score[(y == 0) & (score < t)]
    if len(below):  # halfway between the highest negative under t and the next positive
        t = (below.max() + pos[pos > below.max()].min()) / 2
    else:
        t = pos[0]
    return int(np.ceil(t)) if np.issubdtype(score.dtype, np.integer) else float(t)


def metrics(y, score, thr):
    """Recall, precision, F1, specificity, ROC-AUC and confusion matrix for `score >= thr` = polyp."""
    y, score = np.asarray(y), np.asarray(score)
    pred = score >= thr
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, tn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "n": int(len(y)), "positives": int(np.sum(y == 1)), "negatives": int(np.sum(y == 0)),
        "threshold": thr,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity": tn / max(tn + fp, 1),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < np.sum(y) < len(y) else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
