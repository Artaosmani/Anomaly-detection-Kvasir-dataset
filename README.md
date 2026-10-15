# Polyp detection on Kvasir-SEG

Two-stage pipeline for capsule endoscopy:

1. The camera pill sends a 336×336 frame to the **nRF54LM20 DK**.
2. The board runs the **polyp classifier** (`Classifier/`, INT8 on the Axon NPU) on every frame: polyp yes/no.
3. Over USB the board sends one result line per frame, plus the frame itself when the answer is yes.
4. The PC writes a CSV of every frame. **YOLO26n** (`YOLO/`, config `Configs/default.yml`) runs on frames with
   `detected=1` and outputs boxes for the drug-delivery simulator.

## Polyp classifier

MobileNetV1 α=0.25 (ImageNet weights) → average pool → dropout → 1×1 conv → **one logit**. There is no sigmoid in
the model; the board compares the raw int8 output with a threshold. Everything is in `Configs/classifier.yml`.

| File | What it does |
|---|---|
| `Configs/classifier.yml` | paths, negatives sources, input size (96 or 84), augmentation, two training stages, quantisation, threshold |
| `Classifier/data.py` | positives from `Data/kvasir-seg-split` (same train/val/test as YOLO26n), negatives from `negatives.sources`, de-duplication, video-level split, optional hard-negative crops |
| `Classifier/train.py` | training, val threshold, test eval float + INT8, INT8 export (QAT if needed), size/MACs/memory report |
| `Classifier/run_csv.py` | INT8 model over a folder of frames → one CSV row per frame |

### Setup

```bash
conda activate anomaly_detection
pip install -r requirements-classifier.txt
```

### Negatives (no polyp)

These aren't in the repo; download and unpack them yourself:

- HyperKvasir labelled images, lower GI normals only (`cecum`, `ileum`, `bbps-0-1`, `bbps-2-3`) → `Data/negatives/hyperkvasir/<class>/*.jpg`
- Kvasir-Capsule labelled images, `Normal clean mucosa` → `Data/negatives/kvasir-capsule/*.jpg`, keeping the
  `<video>_<frame>.jpg` names so frames are thinned to one per 15 and each video stays in one split
- Kvasir-Capsule `Polyp` frames → `Data/kvasir-capsule/polyp/` (only used for the domain-shift eval, never for training)

Until then, training uses Kvasir v1 `normal-cecum` (already in `Data/kvasir/`). Identical frames across sources are
dropped by image hash. Upper-GI classes (z-line, pylorus) are deliberately not used as negatives.

### Train and export

```bash
python Classifier/train.py --dry-run                 # resolved config + split sizes
python Classifier/train.py                           # train, evaluate, export -> runs/classifier/
python Classifier/train.py --set imgsz=84            # 84×84 = 1/4-scale JPEG decode of the 336×336 frame
python Classifier/train.py --set hard_negatives=true stage2.lr=5e-5
python Classifier/train.py --weights runs/classifier/best.weights.h5   # re-quantise/evaluate/export only
```

Outputs in `runs/<name>/` (ignored by git):

- `best.weights.h5`: float weights, best val AUC
- `threshold.json`: picked on val for ≥95 % recall, then lowered as far as possible without adding a val false
  positive; stored as a float logit/probability and as a raw int8 value
- `metrics_test.json`: float and INT8 recall, precision, F1, specificity, ROC-AUC, confusion matrix and threshold,
  plus params, MACs, `.tflite` size, peak activation memory, op list and PC interpreter timing
- `metrics_capsule.json`: the same on Kvasir-Capsule frames, if they exist (domain-shift experiment)
- `model_int8.tflite`, `model_int8.h` (C array, quantisation constants, `POLYP_THRESHOLD_INT8`, a 256-entry input LUT)
- `preprocess.md`: exact input scale/zero-point, resize method, and how to apply the threshold

The PC timing in `metrics_test.json` comes from the TFLite interpreter on the PC. **It is not the board latency.**

### Checking and compiling for the Axon NPU

The Nordic tools come from the nRF Connect SDK **Edge AI add-on** (not installed here). With the add-on installed:

1. Check that every op maps to Axon: run `axons_tflite_model_scan.py` from the add-on on
   `runs/classifier/model_int8.tflite`. `train.py` already lists the ops and warns about any outside
   Conv2D / depthwise conv / fully connected / pooling / ReLU. A normal export contains only `CONV_2D`,
   `DEPTHWISE_CONV_2D` and `AVERAGE_POOL_2D`.
2. Compile the same `.tflite` with the add-on's Axon TFLite compiler to get the NPU command stream for the firmware.

Check the add-on documentation for the exact script locations and arguments for your SDK version.

### CSV for the PC side

```bash
python Classifier/run_csv.py --frames path/to/frames --model runs/classifier/model_int8.tflite
```

One row per frame, in frame order:

```
frame_id,t_rx_board_ns,t_infer_start_ns,t_infer_end_ns,t_sent_ns,score,detected
```

`frame_id` is the last number in the filename (`frame_001001.jpg` → 1001), the same id as in `boxes_fake.csv`, so
YOLO26n results join on it. On the PC the timestamps come from the host clock; the board will send the same columns.
`score` is the sigmoid of the dequantised logit, and `detected` uses the int8 threshold from `threshold.json`.

### Design choices for the NPU

- Keras' own MobileNet pads with `ZeroPadding2D` (a `PAD` op), and `GlobalAveragePooling2D` exports as `MEAN`.
  Neither is an Axon op, so the network is rebuilt with `padding='same'` and an `AveragePooling2D` over the whole
  feature map, with the ImageNet weights copied over by layer name.
- The head is a 1×1 conv, not Flatten + Dense, so there is no `RESHAPE` op either.
- Input and output are int8. The input is calibrated to the −1..1 range, so a uint8 pixel maps to int8 through
  `polyp_input_lut` in `model_int8.h`.
- The size is about 297 KB (218,801 params), above the ~250 KB target. Per-channel quantisation parameters and
  flatbuffer metadata add to the 214 KB of int8 weights; this is the same size class as the MLPerf Tiny VWW model,
  and the board has 2 MB of flash.
- If QAT is triggered (INT8 val recall more than 2 points below float), the QAT export starts with a `QUANTIZE`
  (int8 → int8 requantise) op on the input. `train.py` flags it.

### Still open

- Board firmware: load the compiled model, the 1/4-scale JPEG decode or area resize, the input LUT and the threshold
  compare.
- USB transfer: one result line per frame plus the frame for `detected=1`, over the J3 high-speed USB port.
- Real latency on the board: measure on the nRF54LM20 DK; Nordic's figure for this architecture at 96×96 is
  about 14.4 ms on Axon.
- Real negatives: with only Kvasir v1 normal-cecum, polyp vs. no-polyp is too easy (different source images,
  val AUC close to 1). The numbers mean little until HyperKvasir and Kvasir-Capsule negatives are in.
