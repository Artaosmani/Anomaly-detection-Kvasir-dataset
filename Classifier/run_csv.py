"""Run the INT8 polyp classifier over a folder of frames and write one CSV row per frame (PC stand-in for the board).

    python Classifier/run_csv.py --frames path/to/frames --model runs/classifier/model_int8.tflite
    python Classifier/run_csv.py --frames path/to/frames --model ... --out classifier.csv
    python Classifier/run_csv.py --frames Data/test/images --labels Data/test/labels.csv --model ...   # + metrics

Columns: frame_id,t_rx_board_ns,t_infer_start_ns,t_infer_end_ns,t_sent_ns,score,detected
frame_id is the last number in the filename (frame_001001.jpg -> 1001), the same id as in boxes_fake.csv, so YOLO26n
results join on it. Every frame gets a row. Timings come from the host clock (time.time_ns); on the board,
t_rx = frame received, preprocessing runs between t_rx and t_infer_start, and the board sends the same columns over USB.
score = sigmoid of the dequantised logit, detected = raw int8 output >= the threshold in threshold.json.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

from common import FULL, Int8Model, load_image, metrics, normalise, sigmoid

COLUMNS = ["frame_id", "t_rx_board_ns", "t_infer_start_ns", "t_infer_end_ns", "t_sent_ns", "score", "detected"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def frames(folder):
    """[(frame_id, path)] sorted by id; the id is the last run of digits in the filename."""
    out = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() not in IMG_EXT:
            continue
        digits = re.findall(r"\d+", p.stem)
        if not digits:
            raise SystemExit(f"No frame number in filename: {p.name}")
        fid = int(digits[-1])
        if fid in out:
            raise SystemExit(f"Duplicate frame_id {fid}: {out[fid].name} and {p.name}")
        out[fid] = p
    if not out:
        raise SystemExit(f"No frames in {folder}")
    return sorted(out.items())


def main():
    args = parse_args()
    m = Int8Model(args.model)
    thr = args.threshold if args.threshold is not None else \
        json.loads((args.model.parent / "threshold.json").read_text())["int8"]["raw"]
    out = args.out or args.model.parent / f"classifier_{args.frames.name}.csv"
    n = hits = 0
    raws = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for fid, path in frames(args.frames):
            t_rx = time.time_ns()
            x = normalise(load_image(str(path), FULL, m.size)).numpy()
            t0 = time.time_ns()
            raw = raws[fid] = m(x)
            t1 = time.time_ns()
            detected = int(raw >= thr)
            w.writerow([fid, t_rx, t0, t1, time.time_ns(), f"{float(sigmoid(m.logit(raw))):.4f}", detected])
            n, hits = n + 1, hits + detected
    print(f"[csv] {n} frames, {hits} detected (int8 threshold {thr}) -> {out}")
    if args.labels:
        score_against(args.labels, raws, thr, out.with_suffix(".metrics.json"))


def score_against(labels, raws, thr, path):
    """Compare detections with a ground-truth CSV (frame_id,...,polyp) and write the metrics next to the CSV."""
    with open(labels) as f:
        gt = {int(r["frame_id"]): int(r["polyp"]) for r in csv.DictReader(f)}
    missing = sorted(set(gt) ^ set(raws))
    if missing:
        raise SystemExit(f"frame_ids in only one of labels/frames: {missing[:10]}")
    ids = sorted(gt)
    r = metrics([gt[i] for i in ids], [raws[i] for i in ids], thr)
    path.write_text(json.dumps(r, indent=2))
    print(f"[gt] recall {r['recall']:.3f} precision {r['precision']:.3f} f1 {r['f1']:.3f} "
          f"specificity {r['specificity']:.3f} auc {r['roc_auc']:.3f} cm {r['confusion_matrix']} -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, required=True, help="folder of frames, frame number in each filename")
    p.add_argument("--model", type=Path, required=True, help="model_int8.tflite (threshold.json read from its folder)")
    p.add_argument("--threshold", type=int, help="override the int8 threshold")
    p.add_argument("--out", type=Path, help="CSV path (default: next to the model)")
    p.add_argument("--labels", type=Path, help="ground truth CSV with frame_id and polyp (0/1) columns -> metrics")
    return p.parse_args()


if __name__ == "__main__":
    main()
