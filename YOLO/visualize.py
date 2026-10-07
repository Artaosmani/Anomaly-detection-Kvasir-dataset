"""Web visualizer: pick a checkpoint from runs/, pick a folder of images, see predicted boxes.

python YOLO/visualize.py [--port 8000]   ->  open http://127.0.0.1:8000
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src import DEFAULT_CONFIG, ROOT, resolve
from train import resolve_device

RUNS = ROOT / "runs"
MIN_CONF = 0.05  # server returns everything above this; the page's slider filters further
models = {}  # ckpt path -> loaded YOLO, so switching back is instant


def ground_truth():
    """{image stem: [[x0, y0, x1, y1], ...]} in original pixel coords, from the labels json in the config."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    ann = json.loads(resolve(cfg["paths"]["labels"]).read_text())
    return {k: [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in v["bbox"]] for k, v in ann.items()}


def checkpoints():
    """All .pt files under runs/, newest first."""
    return sorted(RUNS.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def predict(ckpt, image_bytes, device):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    if ckpt not in models:
        models[ckpt] = YOLO(str(ckpt))
    r = models[ckpt].predict(img, conf=MIN_CONF, device=device, verbose=False)[0]
    return [{"box": b, "conf": c, "name": r.names[int(k)]}
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


class Handler(BaseHTTPRequestHandler):
    device = "cpu"

    def send(self, code, body, ctype="application/json"):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/ckpts":
            self.send(200, json.dumps([
                {"path": str(p.relative_to(RUNS)),
                 "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
                for p in checkpoints()]))
        elif self.path == "/gt":
            self.send(200, GT_JSON)
        else:
            self.send(404, "{}")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/predict":
            return self.send(404, "{}")
        # only ever load checkpoints that live under runs/ (torch.load unpickles, so no arbitrary paths)
        name = parse_qs(url.query).get("ckpt", [""])[0]
        ckpt = next((p for p in checkpoints() if str(p.relative_to(RUNS)) == name), None)
        if ckpt is None:
            return self.send(400, json.dumps({"error": f"unknown checkpoint {name!r}"}))
        try:
            data = self.rfile.read(int(self.headers["Content-Length"]))
            self.send(200, json.dumps(predict(ckpt, data, self.device)))
        except ValueError as e:
            self.send(400, json.dumps({"error": str(e)}))

    def log_message(self, *args):
        pass


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Polyp Box Viewer</title>
<style>
  body { margin: 0; font: 14px system-ui, sans-serif; background: #111; color: #eee; }
  header { position: sticky; top: 0; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
           padding: 12px 16px; background: #1c1c1c; border-bottom: 1px solid #333; z-index: 1; }
  select, input, button { font: inherit; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; padding: 16px; }
  figure { margin: 0; background: #1c1c1c; border-radius: 6px; overflow: hidden; }
  canvas { width: 100%; display: block; }
  figcaption { padding: 6px 8px; font-size: 12px; color: #aaa; overflow-wrap: anywhere; }
  #status { color: #aaa; }
</style></head><body>
<header>
  <label>Checkpoint <select id="ckpt"></select></label>
  <label>Images <input type="file" id="folder" webkitdirectory multiple></label>
  <label>Conf ≥ <input type="range" id="conf" min="0.05" max="1" step="0.01" value="0.25">
    <span id="confv">0.25</span></label>
  <label><input type="checkbox" id="showgt" checked> Ground truth</label>
  <span><span style="color:#00e676">■</span> prediction <span style="color:#00BFFF">■</span> ground truth</span>
  <span id="status"></span>
</header>
<div id="grid"></div>
<script>
const $ = id => document.getElementById(id);
let items = [];   // {file, img, canvas, cap, boxes}
let GT = {};      // image stem -> [[x0, y0, x1, y1], ...]
fetch("/gt").then(r => r.json()).then(gt => { GT = gt; items.forEach(draw); });
let runId = 0;    // bumps when checkpoint/folder changes, so stale requests are dropped

fetch("/ckpts").then(r => r.json()).then(list => {
  if (!list.length) { $("status").textContent = "No .pt files under runs/"; return; }
  $("ckpt").innerHTML = list.map(c => `<option value="${c.path}">${c.path}  (${c.mtime})</option>`).join("");
});

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const area = r => (r[2] - r[0]) * (r[3] - r[1]);
  return w * h / (area(a) + area(b) - w * h || 1);
}

function draw(it) {
  const conf = +$("conf").value, c = it.canvas, g = c.getContext("2d");
  g.drawImage(it.img, 0, 0);
  const shown = (it.boxes || []).filter(b => b.conf >= conf);
  const lw = Math.max(2, c.width / 250);
  g.lineWidth = lw; g.font = `${lw * 7}px system-ui`;
  for (const b of shown) {
    const [x0, y0, x1, y1] = b.box, label = `${b.name} ${b.conf.toFixed(2)}`;
    g.strokeStyle = "#00e676"; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const tw = g.measureText(label).width + lw * 2, th = lw * 9;
    const ty = y0 - th < 0 ? y0 : y0 - th;
    g.fillStyle = "#00e676"; g.fillRect(x0, ty, tw, th);
    g.fillStyle = "#000"; g.fillText(label, x0 + lw, ty + th - lw * 2);
  }
  const gt = GT[it.file.name.replace(/\\.[^.]+$/, "")];
  if ($("showgt").checked && gt) {
    g.strokeStyle = "#00BFFF";
    for (const [x0, y0, x1, y1] of gt) g.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  // per GT box: IoU with the best-overlapping shown prediction (0 = missed)
  const ious = gt && it.boxes ? gt.map(t => Math.max(0, ...shown.map(b => iou(t, b.box)))) : null;
  it.cap.textContent = !it.boxes ? `${it.file.name} — ${it.error || "predicting…"}`
    : `${it.file.webkitRelativePath || it.file.name} — ${shown.length} pred / ${gt ? gt.length : "no"} GT` +
      (ious ? ` — IoU ${ious.map(v => v.toFixed(2)).join(", ")}` : "");
}

async function run() {
  const id = ++runId, ckpt = $("ckpt").value;
  if (!ckpt || !items.length) return;
  items.forEach(it => { it.boxes = null; it.error = null; if (it.img.complete) draw(it); });
  for (let i = 0; i < items.length; i++) {
    if (id !== runId) return;
    $("status").textContent = `Predicting ${i + 1}/${items.length}…`;
    const it = items[i];
    const r = await fetch(`/predict?ckpt=${encodeURIComponent(ckpt)}`, { method: "POST", body: it.file });
    if (id !== runId) return;
    const res = await r.json();
    if (r.ok) it.boxes = res; else it.error = res.error;
    await it.ready; draw(it);
  }
  $("status").textContent = `Done: ${items.length} images`;
}

$("folder").onchange = e => {
  items.forEach(it => URL.revokeObjectURL(it.img.src));
  $("grid").innerHTML = "";
  const files = [...e.target.files].filter(f => f.type.startsWith("image/"))
    .sort((a, b) => a.name.localeCompare(b.name));
  items = files.map(file => {
    const fig = document.createElement("figure"), canvas = document.createElement("canvas"),
          cap = document.createElement("figcaption"), img = new Image();
    fig.append(canvas, cap); $("grid").append(fig);
    const it = { file, img, canvas, cap, boxes: null };
    it.ready = new Promise(res => img.onload = () => {
      canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(it); res();
    });
    img.src = URL.createObjectURL(file);
    return it;
  });
  if (!files.length) $("status").textContent = "No images in that folder";
  run();
};
$("ckpt").onchange = run;
$("showgt").onchange = () => items.forEach(draw);
$("conf").oninput = () => { $("confv").textContent = (+$("conf").value).toFixed(2); items.forEach(draw); };
</script></body></html>
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | 0")
    a = p.parse_args()
    Handler.device = resolve_device(a.device)
    GT_JSON = json.dumps(ground_truth())
    print(f"[viz] {len(checkpoints())} checkpoint(s) under {RUNS}, device={Handler.device}")
    print(f"[viz] open http://127.0.0.1:{a.port}")
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
