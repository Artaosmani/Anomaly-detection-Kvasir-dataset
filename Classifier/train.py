"""Train the binary polyp classifier (MobileNetV1 α=0.25 -> one logit) and export it as INT8 for the nRF54LM20 Axon NPU.

Examples:
    python Classifier/train.py                                  # use Configs/classifier.yml
    python Classifier/train.py --set imgsz=84 dropout=0.3
    python Classifier/train.py --set stage1.epochs=1 stage2.epochs=1 --name smoke
    python Classifier/train.py --weights runs/classifier/best.weights.h5   # skip training: quantise, evaluate, export
    python Classifier/train.py --dry-run                        # print resolved config + split sizes and exit

Writes runs/<name>/: best.weights.h5, threshold.json, metrics_test.json, model_int8.tflite, model_int8.h,
preprocess.md, config_resolved.yaml.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
import yaml
from tf_keras import layers

from common import DEFAULT_CONFIG, Int8Model, load_config, metrics, pick_threshold, resolve, sigmoid
from data import build_samples, class_weights, dataset

# MobileNetV1 depthwise-separable blocks: (pointwise filters before α, stride)
BLOCKS = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)] + [(512, 1)] * 5 + [(1024, 2), (1024, 1)]
# TFLite builtins the Axon NPU accelerates; anything else runs on the M33 or isn't supported
AXON_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D", "MAX_POOL_2D",
            "RELU", "RELU6", "LEAKY_RELU"}


# ---------------------------------------------------------------- model

def build_model(cfg):
    """MobileNetV1 with padding='same' everywhere, AveragePool over the whole map, Dropout, 1×1 Conv -> one logit.

    Keras' own MobileNet pads with ZeroPadding2D (a PAD op) and GlobalAveragePooling2D becomes MEAN; neither is an
    Axon op, so the layers are rebuilt here with the same names and the ImageNet weights copied over by name.
    Output shape is (B,1,1,1): no Flatten/Reshape, no sigmoid.
    """
    m = cfg["model"]
    size, alpha = m["imgsz"], m["alpha"]

    def conv_bn(x, layer):
        x = layer(x)
        x = layers.BatchNormalization(name=f"{layer.name}_bn")(x)
        return layers.ReLU(6.0, name=f"{layer.name}_relu")(x)

    inp = layers.Input((size, size, 3))
    x = conv_bn(inp, layers.Conv2D(int(32 * alpha), 3, strides=2, padding="same", use_bias=False, name="conv1"))
    for i, (filters, stride) in enumerate(BLOCKS, 1):
        x = conv_bn(x, layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"conv_dw_{i}"))
        x = conv_bn(x, layers.Conv2D(int(filters * alpha), 1, use_bias=False, name=f"conv_pw_{i}"))
    x = layers.AveragePooling2D(x.shape[1:3], name="pool")(x)
    x = layers.Dropout(m["dropout"], name="dropout")(x)
    model = keras.Model(inp, layers.Conv2D(1, 1, name="logit")(x), name="polyp_mobilenet")

    if m["weights"] == "imagenet":
        ref = keras.applications.MobileNet(input_shape=(size, size, 3), alpha=alpha, include_top=False,
                                           weights="imagenet")  # warns for < 128 px and uses the 224 weights
        names = {l.name for l in model.layers}
        copied = [l.name for l in ref.layers if l.weights and l.name in names]
        for name in copied:
            model.get_layer(name).set_weights(ref.get_layer(name).get_weights())
        expected = sum(1 for l in model.layers if l.weights and l.name.startswith("conv"))
        if len(copied) != expected:
            raise SystemExit(f"Copied ImageNet weights into {len(copied)}/{expected} layers")
    return model


def set_backbone_trainable(model, trainable, freeze_bn):
    for l in model.layers:
        if l.name.startswith("conv"):
            l.trainable = trainable and not (freeze_bn and isinstance(l, layers.BatchNormalization))


def count_macs(model):
    macs = 0
    for l in model.layers:
        if isinstance(l, layers.Conv2D):  # DepthwiseConv2D is a Conv2D subclass
            h, w = l.output.shape[1:3]
            per_pixel = int(np.prod(l.kernel_size)) * l.input.shape[-1]
            macs += h * w * per_pixel * (1 if isinstance(l, layers.DepthwiseConv2D) else l.filters)
    return int(macs)


# ---------------------------------------------------------------- training

def fit(model, cfg, train_ds, val_ds, epochs, lr, ckpt, best=None):
    """One stage: BCE on logits, early stop on val AUC, keep the best weights in `ckpt`. Returns best val AUC."""
    model.compile(optimizer=keras.optimizers.legacy.Adam(lr),  # the v2 Adam is slow on Apple silicon
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  weighted_metrics=[], metrics=[keras.metrics.AUC(from_logits=True, name="auc")])
    save = keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_auc", mode="max", save_best_only=True,
                                           save_weights_only=True, initial_value_threshold=best)
    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=cfg["train"]["patience"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[save, stop], verbose=2)
    if ckpt.exists():
        model.load_weights(str(ckpt))
    return max([best or 0.0] + hist.history["val_auc"])


def train(model, cfg, samples, out):
    t = cfg["train"]
    w = class_weights(samples["train"])
    print(f"[train] class weights: no-polyp {w[0]:.3f}, polyp {w[1]:.3f}")
    train_ds, val_ds = dataset(samples["train"], cfg, train=True, weights=w), dataset(samples["val"], cfg)
    ckpt = out / "best.weights.h5"

    print("[train] stage 1: backbone frozen")
    set_backbone_trainable(model, False, t["freeze_bn"])
    best = fit(model, cfg, train_ds, val_ds, t["stage1"]["epochs"], t["stage1"]["lr"], ckpt)
    print("[train] stage 2: backbone unfrozen" + (" (BatchNorm frozen)" if t["freeze_bn"] else ""))
    set_backbone_trainable(model, True, t["freeze_bn"])
    best = fit(model, cfg, train_ds, val_ds, t["stage2"]["epochs"], t["stage2"]["lr"], ckpt, best)
    print(f"[train] best val AUC {best:.4f} -> {ckpt}")
    return train_ds, val_ds


def qat(model, cfg, train_ds, val_ds, out):
    """Quantisation-aware fine-tuning, used only when post-training INT8 loses too much recall."""
    import tensorflow_model_optimization as tfmot
    q = cfg["quantize"]
    qmodel = tfmot.quantization.keras.quantize_model(model)
    fit(qmodel, cfg, train_ds, val_ds, q["qat_epochs"], q["qat_lr"], out / "qat.weights.h5")
    return qmodel


# ---------------------------------------------------------------- INT8

def to_int8(model, cfg, samples, path):
    """Full-integer conversion (int8 in/out), calibrated on un-augmented train images."""
    n = cfg["quantize"]["representative"]
    rng = np.random.default_rng(cfg["train"]["seed"])
    pick = [samples["train"][i] for i in rng.permutation(len(samples["train"]))[:n]]
    rep = dataset(pick, cfg, batch=1)

    def representative():
        for x, _ in rep:
            yield [x]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    path.write_bytes(conv.convert())
    return Int8Model(path)


def tflite_report(path):
    """Ops used and peak activation memory (largest input+output activations of any single op).

    MobileNetV1 has no skip connections, so that is a fair estimate of the tensor arena; TFLM adds its own overhead.
    """
    it = Int8Model(path).it
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    activations = {o for op in ops for o in op["outputs"]} | {d["index"] for d in it.get_input_details()}

    def nbytes(i):
        return int(np.prod(tensors[i]["shape"])) * np.dtype(tensors[i]["dtype"]).itemsize

    peak = max(sum(nbytes(i) for i in list(op["inputs"]) + list(op["outputs"]) if i in activations) for op in ops)
    return sorted({op["op_name"] for op in ops}), peak


def time_pc(m, runs):
    """Median/p95 ms of one invoke() on this PC. NOT the board latency."""
    x = np.zeros((m.size, m.size, 3), np.float32)
    m(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "runs": runs, "threads": 1}


# ---------------------------------------------------------------- scoring

def scores(samples, cfg, model=None, int8=None):
    """Labels and scores: float logits from `model`, or raw int8 outputs from `int8`."""
    ds = dataset(samples, cfg, batch=64)
    y = np.array([s[1] for s in samples])
    if model is not None:
        return y, model.predict(ds, verbose=0).ravel()
    return y, np.array([int8(x) for xb, _ in ds for x in xb.numpy()])


def evaluate(samples, cfg, model, int8, thr):
    y, f = scores(samples, cfg, model=model)
    _, q = scores(samples, cfg, int8=int8)
    return {"float": metrics(y, f, thr["float"]["logit"]), "int8": metrics(y, q, thr["int8"]["raw"])}


# ---------------------------------------------------------------- export

def write_header(path, tflite, m, thr_raw):
    """C array + the constants the firmware needs: input LUT (uint8 pixel -> int8), quant params, threshold."""
    lut = [int(v) for v in m.quantise(np.arange(256, dtype=np.float32) / 127.5 - 1.0)]
    body = ",\n  ".join(", ".join(f"0x{b:02x}" for b in tflite[i:i + 16]) for i in range(0, len(tflite), 16))
    path.write_text(f"""/* Generated by Classifier/train.py: polyp yes/no, MobileNetV1 a=0.25, full INT8. Do not edit. */
#pragma once
#include <stdint.h>

#define POLYP_INPUT_SIZE        {m.size}
#define POLYP_INPUT_SCALE       {m.in_scale:.9g}f
#define POLYP_INPUT_ZERO_POINT  {m.in_zp}
#define POLYP_OUTPUT_SCALE      {m.out_scale:.9g}f
#define POLYP_OUTPUT_ZERO_POINT {m.out_zp}
#define POLYP_THRESHOLD_INT8    {thr_raw}  /* detected = (int8 output >= POLYP_THRESHOLD_INT8) */

/* uint8 RGB pixel (after resize) -> int8 model input */
static const int8_t polyp_input_lut[256] = {{
{", ".join(map(str, lut))}
}};

static const unsigned int polyp_model_int8_len = {len(tflite)};
__attribute__((aligned(16))) static const uint8_t polyp_model_int8[] = {{
  {body}
}};
""")


def write_preprocess(path, m, cfg, thr, ops):
    path.write_text(f"""# Board preprocessing for model_int8.tflite

Generated by `Classifier/train.py`. The same numbers are in `model_int8.h`.

| | |
|---|---|
| Input | `{m.size}×{m.size}×3`, int8, NHWC, RGB order |
| Input scale / zero-point | `{m.in_scale:.9g}` / `{m.in_zp}` |
| Output | one int8 logit, shape `1×1×1×1` (no sigmoid in the model) |
| Output scale / zero-point | `{m.out_scale:.9g}` / `{m.out_zp}` |
| Threshold | `detected = output_int8 >= {thr['int8']['raw']}` (p = {thr['int8']['prob']:.4f}, val recall >= {thr['target_recall']:.0%}) |
| Ops | {', '.join(f'`{o}`' for o in ops)} |

## Steps

1. Take the 336×336 RGB frame.
2. Resize to {m.size}×{m.size} with an **area (box-filter) resize**: every output pixel is the mean of the input
   pixels it covers. At 84 this is an exact 4×4 block mean, so a 1/4-scale JPEG decode (libjpeg `scale_denom=4`)
   gives the same image without a separate resize. At 96 (3.5× down) do a real area resize.
3. Map every uint8 channel value through `polyp_input_lut[256]`. That is
   `int8 = clamp(round((pixel / 127.5 - 1) / {m.in_scale:.9g} + {m.in_zp}), -128, 127)`.
4. Run the model, read the single int8 output and compare it with `POLYP_THRESHOLD_INT8`.
   A probability for the CSV, if wanted, is `1 / (1 + exp(-(out - {m.out_zp}) * {m.out_scale:.9g}))` on the CPU.

Training squashed non-square Kvasir images to a square without cropping; board frames are already square.
""")


# ---------------------------------------------------------------- run

def output_dir(cfg):
    o = cfg["output"]
    base = resolve(o["project"]) / o["name"]
    out, i = base, 2
    while out.exists() and not o["exist_ok"]:
        out, i = base.with_name(f"{base.name}-{i}"), i + 1
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["output"]["name"] = args.name
    print(yaml.safe_dump(cfg, sort_keys=False))
    keras.utils.set_random_seed(cfg["train"]["seed"])
    samples, capsule = build_samples(cfg)
    if args.dry_run:
        return
    out = output_dir(cfg)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = build_model(cfg)
    stats = {"params": int(model.count_params()), "macs": count_macs(model), "imgsz": cfg["model"]["imgsz"]}
    print(f"[model] {stats['params']:,} params, {stats['macs'] / 1e6:.2f} M MACs at {stats['imgsz']}×{stats['imgsz']}")

    if args.weights:
        model.load_weights(str(args.weights))
        w = class_weights(samples["train"])
        train_ds, val_ds = dataset(samples["train"], cfg, train=True, weights=w), dataset(samples["val"], cfg)
    else:
        train_ds, val_ds = train(model, cfg, samples, out)

    target = cfg["threshold"]["target_recall"]
    yv, fv = scores(samples["val"], cfg, model=model)
    t_float = pick_threshold(yv, fv, target)

    # post-training INT8; QAT only if recall at the float threshold drops too much on val
    tfl = out / "model_int8.tflite"
    int8 = to_int8(model, cfg, samples, tfl)
    _, qv = scores(samples["val"], cfg, int8=int8)
    drop = metrics(yv, fv, t_float)["recall"] - metrics(yv, int8.logit(qv), t_float)["recall"]
    print(f"[int8] val recall drop at the float threshold: {drop * 100:.1f} points")
    used_qat = drop > cfg["quantize"]["max_recall_drop"]
    if used_qat:
        print("[int8] drop too large -> quantisation-aware training")
        qmodel = qat(model, cfg, train_ds, val_ds, out)
        int8 = to_int8(qmodel, cfg, samples, tfl)
        _, qv = scores(samples["val"], cfg, int8=int8)
        drop = metrics(yv, fv, t_float)["recall"] - metrics(yv, int8.logit(qv), t_float)["recall"]
        print(f"[int8] after QAT: drop {drop * 100:.1f} points")

    t_raw = int(pick_threshold(yv, qv, target))
    thr = {"target_recall": target,
           "float": {"logit": float(t_float), "prob": float(sigmoid(t_float))},
           "int8": {"raw": t_raw, "logit": float(int8.logit(t_raw)), "prob": float(sigmoid(int8.logit(t_raw)))}}
    (out / "threshold.json").write_text(json.dumps(thr, indent=2))
    print("[threshold]", json.dumps(thr))

    ops, peak = tflite_report(tfl)
    tflite = tfl.read_bytes()
    stats.update({"tflite_bytes": len(tflite), "peak_activation_bytes": int(peak), "ops": ops,
                  "non_axon_ops": sorted(set(ops) - AXON_OPS), "qat": used_qat,
                  "val_recall_drop_int8": float(drop),
                  "pc_tflite_latency": time_pc(int8, cfg["evaluate"]["timing_runs"]),
                  "pc_latency_note": "TFLite interpreter on this PC, 1 thread. NOT the nRF54LM20 board latency."})
    print(f"[int8] {tfl}: {len(tflite) / 1024:.1f} KB, peak activations {peak / 1024:.1f} KB (512 KB RAM), ops {ops}")
    if stats["non_axon_ops"]:
        print(f"[int8] WARNING: ops not accelerated by Axon: {stats['non_axon_ops']}")
    lat = stats["pc_tflite_latency"]
    print(f"[int8] PC interpreter {lat['median_ms']:.2f} ms median (sanity check only, NOT board latency)")

    result = {"split": "test", "model": stats, "threshold": thr, **evaluate(samples["test"], cfg, model, int8, thr)}
    (out / "metrics_test.json").write_text(json.dumps(result, indent=2))
    for k in ("float", "int8"):
        r = result[k]
        print(f"[test] {k:5s}: recall {r['recall']:.3f} precision {r['precision']:.3f} f1 {r['f1']:.3f} "
              f"specificity {r['specificity']:.3f} auc {r['roc_auc']:.3f} cm {r['confusion_matrix']}")
    if capsule:
        cap = {"split": "kvasir-capsule (domain shift)", "threshold": thr, **evaluate(capsule, cfg, model, int8, thr)}
        (out / "metrics_capsule.json").write_text(json.dumps(cap, indent=2))
        print(f"[capsule] int8 recall {cap['int8']['recall']:.3f} specificity {cap['int8']['specificity']:.3f}")
    else:
        print("[capsule] no Kvasir-Capsule polyp + normal frames, domain-shift eval skipped")

    write_header(out / "model_int8.h", tflite, int8, t_raw)
    write_preprocess(out / "preprocess.md", int8, cfg, thr, ops)
    print(f"[export] {out}/model_int8.{{tflite,h}}, preprocess.md, metrics_test.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any config key, e.g. --set imgsz=84 stage2.lr=1e-5")
    p.add_argument("--name", help="run name under output.project")
    p.add_argument("--weights", type=Path, help="skip training, start from these weights (.h5)")
    p.add_argument("--dry-run", action="store_true", help="print resolved config and split sizes, then exit")
    return p.parse_args()


if __name__ == "__main__":
    main()
