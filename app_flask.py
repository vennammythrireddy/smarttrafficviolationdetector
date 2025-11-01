# app_flask.py
# Flask web app: upload IMAGE or VIDEO, calibrate zones, and see violations.
# Fast & minimal. No audio. Uses YOLOv8n (auto-downloads if models/yolov8n.pt missing).

import os
if not os.path.exists("yolov8n.pt"):
    os.system("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")

from pathlib import Path
from time import time
from flask import Flask, request, redirect, url_for, Response, render_template_string, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"; UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE / "models"; MODELS_DIR.mkdir(exist_ok=True)

# --------------- SPEED / QUALITY KNOBS ---------------
IMGSZ = 416     # 320–480 (lower = faster)
SCALE = 0.70    # downscale frame before inference (0.5–1.0)
SKIP  = 0       # process 1, skip N frames for video (0 = none)
CONF  = 0.35    # YOLO confidence
# ----------------------------------------------------

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# ========= DEFAULT ZONES (replace after calibration) =========
NO_PARK_ZONE      = np.array([[100,400],[600,400],[600,700],[100,700]], dtype=np.int32)
JUNCTION_ZONE     = np.array([[700,500],[1250,500],[1250,800],[700,800]], dtype=np.int32)
TRAFFIC_LIGHT_ROI = (1100,80,1180,170)  # (x1,y1,x2,y2)
# =============================================================

app = Flask(__name__)

# Load YOLO once (fast)
MODEL_PATH = MODELS_DIR / "yolov8n.pt"
model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else YOLO("yolov8n.pt")
model.fuse()

# ---------------- HTML ----------------
INDEX_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Traffic Violations</title>
<style>
body{font-family:Arial;margin:24px} .card{max-width:980px;margin:auto;padding:20px;border:1px solid #ddd;border-radius:12px}
.row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
button{padding:10px 16px;border:none;background:#0d6efd;color:#fff;border-radius:8px;cursor:pointer}
input[type=file]{padding:8px}
.small{font-size:13px;color:#555} img{width:100%;max-height:70vh;object-fit:contain;background:#000}
.hint{background:#f6f8ff;padding:10px;border-radius:8px;margin-top:12px}
code{background:#f3f3f3;padding:2px 6px;border-radius:6px}
</style></head><body>
<div class="card">
  <h2>Traffic Violation Detection (No-Parking + Red-Light)</h2>
  <div class="hint">Upload <b>Image</b> to get an immediate processed picture, or <b>Video</b> to stream live. Use <b>Calibrate</b> to click exact polygons & ROI so violations are correct.</div>

  <h3>Upload Image</h3>
  <form method="POST" action="/upload_image" enctype="multipart/form-data" class="row">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Process Image</button>
  </form>

  <h3 style="margin-top:16px;">Upload Video</h3>
  <form method="POST" action="/upload_video" enctype="multipart/form-data" class="row">
    <input type="file" name="video" accept="video/*" required>
    <button type="submit">Start Stream</button>
  </form>

  {% if img_name %}
    <h3 style="margin-top:18px;">Image Result</h3>
    <div class="row">
      <a href="{{ url_for('calibrate_image', filename=img_name) }}"><button type="button">Calibrate (Image)</button></a>
    </div>
    <img src="{{ url_for('processed_image', filename=img_name) }}">
  {% endif %}

  {% if vid_name %}
    <div class="row" style="margin-top:18px;">
      <a href="{{ url_for('calibrate_video', filename=vid_name) }}"><button type="button">Calibrate (Video)</button></a>
      <a href="{{ url_for('stream', filename=vid_name) }}"><button type="button">Open Stream</button></a>
    </div>
    <div class="small">Stream URL: <code>{{ url_for('stream', filename=vid_name) }}</code></div>
    <h3 style="margin-top:12px;">Live Stream</h3>
    <img src="{{ url_for('stream', filename=vid_name) }}">
  {% endif %}
</div></body></html>
"""

CAL_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Calibrate Zones</title>
<style>
body{font-family:Arial;margin:24px} .wrap{max-width:1100px;margin:auto}
#img{max-width:100%;border:1px solid #ddd;background:#000}
.badge{display:inline-block;background:#eef;padding:4px 8px;border-radius:8px;margin:4px 6px}
pre{background:#f6f8fa;padding:10px;border-radius:8px;white-space:pre-wrap}
</style></head><body>
<div class="wrap">
  <h2>Calibration – click points on the image</h2>
  <div>Order:
    <span class="badge">No-Parking: 4 clicks</span>
    <span class="badge">Junction: 4 clicks</span>
    <span class="badge">Traffic-light ROI: 2 clicks (top-left, bottom-right)</span>
  </div>
  <img id="img" src="{{ snap_url }}">
  <div id="status" style="margin:8px 0;color:#333"></div>
  <pre id="out"></pre>
</div>
<script>
const pts = [];
const img = document.getElementById('img');
const status = document.getElementById('status');
const out = document.getElementById('out');
img.addEventListener('click', (e)=>{
  const r = img.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) * {{ scaleW }});
  const y = Math.round((e.clientY - r.top)  * {{ scaleH }});
  pts.push([x,y]);
  status.innerText = Clicked: (${x}, ${y})  [${pts.length}/10];
  if(pts.length===10){
    const np = pts.slice(0,4).map(p=>[${p[0]}, ${p[1]}]).join(', ');
    const jn = pts.slice(4,8).map(p=>[${p[0]}, ${p[1]}]).join(', ');
    const tl = pts.slice(8,10);
    const code = `
# Replace these in app_flask.py:
NO_PARK_ZONE = np.array([[${np}]], dtype=np.int32)
JUNCTION_ZONE = np.array([[${jn}]], dtype=np.int32)
TRAFFIC_LIGHT_ROI = (${tl[0][0]}, ${tl[0][1]}, ${tl[1][0]}, ${tl[1][1]})
`.trim();
    out.innerText = code + "\\n\\nCopy into app_flask.py, save, and restart the server.";
  }
});
</script>
</body></html>
"""

# --------------- helpers ---------------
def is_red_light(roi_bgr):
    if roi_bgr.size == 0: return (False, 0.0)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,100,100), (10,255,255))
    m2 = cv2.inRange(hsv, (160,100,100), (179,255,255))
    m = cv2.bitwise_or(m1, m2)
    red_ratio = float(m.sum())/255.0/(m.shape[0]*m.shape[1])
    return (red_ratio > 0.02), red_ratio  # slightly easier than 0.03

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (int(pt[0]), int(pt[1])), False) >= 0

def scale_poly(poly, s):
    return np.array([(int(x*s), int(y*s)) for x,y in poly.reshape(-1,2)], dtype=np.int32)

def draw_and_detect(frame, np_zone, jn_zone, roi_rect):
    # red-light check
    x1,y1,x2,y2 = roi_rect
    red_on, red_ratio = is_red_light(frame[y1:y2, x1:x2])

    # YOLO inference
    res = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=0.5, verbose=False)[0]

    # guides
    cv2.polylines(frame, [np_zone], True, (0,0,255), 2)
    cv2.polylines(frame, [jn_zone], True, (255,0,0), 2)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
    cv2.putText(frame, f"RED: {'YES' if red_on else 'NO'} ({red_ratio*100:.1f}%)",
                (x1, max(22, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if red_on else (0,200,0), 2)

    if res.boxes is not None and len(res.boxes)>0:
        names = res.names
        for b in res.boxes:
            cls = int(b.cls[0])
            if cls not in VEHICLE_CLASS_IDS: continue
            x1b,y1b,x2b,y2b = map(int, b.xyxy[0])
            conf = float(b.conf[0])

            cv2.rectangle(frame, (x1b,y1b),(x2b,y2b),(255,255,0), 2)
            cv2.putText(frame, f"{names[cls]} {conf:.2f}", (x1b,y1b-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cx, cy = (x1b+x2b)//2, (y1b+y2b)//2
            cv2.circle(frame, (cx,cy), 3, (0,255,255), -1)

            if point_in_poly((cx,cy), np_zone):
                cv2.putText(frame, "VIOLATION: NO-PARKING", (x1b, y2b+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            if red_on and point_in_poly((cx,cy), jn_zone):
                cv2.putText(frame, "VIOLATION: RED LIGHT", (x1b, y2b+36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return frame

# --------------- video streaming ---------------
def stream_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        yield b""; return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    s = float(SCALE)
    outW, outH = int(W*s), int(H*s)

    np_zone = scale_poly(NO_PARK_ZONE, s)
    jn_zone = scale_poly(JUNCTION_ZONE, s)
    x1,y1,x2,y2 = [int(c*s) for c in TRAFFIC_LIGHT_ROI]

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_id += 1

        if s != 1.0:
            frame = cv2.resize(frame, (outW, outH), interpolation=cv2.INTER_LINEAR)

        if SKIP>0 and (frame_id % (SKIP+1)) != 1:
            continue

        frame = draw_and_detect(frame, np_zone, jn_zone, (x1,y1,x2,y2))

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

    cap.release()

# --------------- routes ---------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

# ---- IMAGE FLOW ----
@app.route("/upload_image", methods=["POST"])
def upload_image():
    f = request.files.get("image")
    if not f or f.filename == "": return redirect(url_for("index"))
    ext = Path(f.filename).suffix or ".jpg"
    fname = f"img_{int(time())}{ext}"
    p = UPLOAD_DIR / fname
    f.save(p)
    # Process once and store processed PNG alongside
    out_png = p.with_suffix(".processed.png")
    img = cv2.imread(str(p))
    if img is None: return redirect(url_for("index"))

    # scale zones/ROI to image scale if you want; here we keep original coords assuming same scene/resolution
    frame = draw_and_detect(img.copy(), NO_PARK_ZONE, JUNCTION_ZONE, TRAFFIC_LIGHT_ROI)

    cv2.imwrite(str(out_png), frame)
    return render_template_string(INDEX_HTML, img_name=fname)

@app.route("/processed_image/<filename>")
def processed_image(filename):
    p = (UPLOAD_DIR / filename).with_suffix(".processed.png")
    if not p.exists(): return "Not found", 404
    return send_from_directory(UPLOAD_DIR, p.name)

@app.route("/calibrate_image/<filename>")
def calibrate_image(filename):
    # Show the raw uploaded image for clicking; scale factors 1.0
    p = UPLOAD_DIR / filename
    if not p.exists(): return "Not found", 404
    snap_url = url_for("raw_upload", filename=filename)
    return render_template_string(CAL_HTML, snap_url=snap_url, scaleW=1.0, scaleH=1.0)

# ---- VIDEO FLOW ----
@app.route("/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("video")
    if not f or f.filename == "": return redirect(url_for("index"))
    ext = Path(f.filename).suffix or ".mp4"
    fname = f"vid_{int(time())}{ext}"
    p = UPLOAD_DIR / fname
    f.save(p)
    return render_template_string(INDEX_HTML, vid_name=fname)

@app.route("/stream/<filename>")
def stream(filename):
    p = UPLOAD_DIR / filename
    if not p.exists(): return "Not found", 404
    return Response(stream_video(p), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/calibrate_video/<filename>")
def calibrate_video(filename):
    # Show first frame (original size) for clicks; scale factors 1.0
    snap_url = url_for("snapshot", filename=filename)
    return render_template_string(CAL_HTML, snap_url=snap_url, scaleW=1.0, scaleH=1.0)

# raw file (image) for calibration
@app.route("/raw/<filename>")
def raw_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# first frame of a video for calibration
@app.route("/snapshot/<filename>")
def snapshot(filename):
    p = UPLOAD_DIR / filename
    cap = cv2.VideoCapture(str(p))
    ok, frame = cap.read()
    cap.release()
    if not ok: return "Cannot read video", 500
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok: return "encode error", 500
    from flask import make_response
    resp = make_response(jpg.tobytes())
    resp.headers["Content-Type"] = "image/jpeg"
    return resp

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
