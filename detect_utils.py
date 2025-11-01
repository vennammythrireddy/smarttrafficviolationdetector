# detect_utils.py (FAST, NO VOICE)
from __future__ import annotations
import math, time, cv2, numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO

# ===== speed/stability knobs =====
DEVICE = "cpu"          # set "cuda" if you actually have a working GPU
MAX_LONG_SIDE = 960     # downscale big images/frames to this long side
IMGZ_VEH = 960          # vehicle model imgsz
IMGZ_HLM = 640          # helmet model imgsz
# =================================

@dataclass
class DetectionSummary:
    vehicle_counts: Dict[str, int]
    motorcycle_count: int
    no_helmet_violations: int
    no_parking_violations: int
    red_light_violations: int
    messages: List[str]

# cache models in memory
_MODEL_CACHE: Dict[str, YOLO] = {}

def _load_model(path: str) -> YOLO:
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    m = YOLO(path)
    m.to(DEVICE)
    _MODEL_CACHE[path] = m
    return m

def _veh_model() -> YOLO:
    return _load_model(str(Path("models") / "yolov8n.pt"))

def _helmet_model() -> Optional[YOLO]:
    p = Path("models") / "helmet.pt"
    return _load_model(str(p)) if p.exists() else None

VEHICLE_LIKE = {"car", "motorcycle", "bus", "truck", "bicycle", "van"}

def _draw_box(img, xyxy, label, color=(0, 255, 0), thick=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(0, y1 - 6)
    cv2.rectangle(img, (x1, y - h - 6), (x1 + w + 6, y), color, -1)
    cv2.putText(img, label, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def _iou(a, b) -> float:
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def _center(box): x1, y1, x2, y2 = box; return ((x1+x2)/2.0, (y1+y2)/2.0)
def _inside(pt, rect): x,y = pt; x1,y1,x2,y2 = rect; return x1<=x<=x2 and y1<=y<=y2

def _red_ratio(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0: return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,120,100), (10,255,255))
    m2 = cv2.inRange(hsv, (160,120,100), (180,255,255))
    mask = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(mask)) / (mask.size + 1e-6)

DEFAULT_CFG = {
    "no_parking_rect": [0.05, 0.55, 0.45, 0.95],  # x1,y1,x2,y2 (normalized)
    "stop_line_y": 0.60,                          # normalized Y
    "light_roi": [0.80, 0.05, 0.97, 0.25],        # red-light ROI
    "red_ratio_thresh": 0.06,
    "dwell_frames": 30
}

def _denorm(rect, W, H):
    x1,y1,x2,y2 = rect
    return [int(x1*W), int(y1*H), int(x2*W), int(y2*H)]

def _maybe_resize(img):
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_LONG_SIDE: return img, 1.0
    scale = MAX_LONG_SIDE / float(long_side)
    new = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return new, scale

def warmup():
    """load models and run 1 tiny inference to avoid first-request lag."""
    vm = _veh_model()
    dummy = np.zeros((640,640,3), np.uint8)
    vm.predict(dummy, device=DEVICE, verbose=False, imgsz=IMGZ_VEH)
    hm = _helmet_model()
    if hm is not None:
        hm.predict(dummy, device=DEVICE, verbose=False, imgsz=IMGZ_HLM)

# -------------------- IMAGE --------------------
def process_image(path: str,
                  vehicle_conf: float = 0.35,
                  helmet_conf: float = 0.35,
                  config: Optional[dict] = None) -> Tuple[np.ndarray, DetectionSummary]:
    t0 = time.time()
    cfg = {**DEFAULT_CFG, **(config or {})}
    vm = _veh_model()
    hm = _helmet_model()

    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    img, _ = _maybe_resize(img)
    H, W = img.shape[:2]
    ann = img.copy()

    np_rect = _denorm(cfg["no_parking_rect"], W, H)
    sy      = int(cfg["stop_line_y"] * H)
    lr      = _denorm(cfg["light_roi"], W, H)
    cv2.rectangle(ann, (np_rect[0],np_rect[1]), (np_rect[2],np_rect[3]), (90,90,255), 2)
    cv2.line(ann, (0,sy), (W,sy), (255,255,0), 2)
    cv2.rectangle(ann, (lr[0],lr[1]), (lr[2],lr[3]), (0,140,255), 2)

    r = vm.predict(img, conf=vehicle_conf, device=DEVICE, verbose=False, imgsz=IMGZ_VEH)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
    clsi  = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), int)
    names = [r.names[i] for i in clsi]

    counts = {k:0 for k in VEHICLE_LIKE}
    persons, bikes, vehicles = [], [], []
    for b,n in zip(boxes, names):
        if n in VEHICLE_LIKE:
            counts[n] = counts.get(n,0) + 1
            vehicles.append((b,n))
        if n == "motorcycle": bikes.append(b)
        if n == "person":     persons.append(b)

    helmet_boxes = []
    if hm is not None:
        hr = hm.predict(img, conf=helmet_conf, device=DEVICE, verbose=False, imgsz=IMGZ_HLM)[0]
        if hr.boxes is not None:
            hb = hr.boxes.xyxy.cpu().numpy()
            hcls = hr.boxes.cls.cpu().numpy().astype(int)
            hnames = [hr.names[i] for i in hcls]
            for b,n in zip(hb,hnames):
                if "helmet" in n.lower(): helmet_boxes.append(b)

    v_helmet = 0
    for p in persons:
        if not any(_iou(p, m) > 0.05 for m in bikes): 
            continue
        x1,y1,x2,y2 = p
        head = [x1, y1, x2, y1 + 0.3*(y2-y1)]
        has_helmet = any(_iou(head, hb) > 0.15 for hb in helmet_boxes)
        if has_helmet: _draw_box(ann, p, "Rider: Helmet OK", (0,200,0), 2)
        else:          _draw_box(ann, p, "Rider: NO HELMET!", (0,0,255), 3); v_helmet += 1

    v_nopark = 0
    for b,n in vehicles:
        if _inside(_center(b), np_rect):
            _draw_box(ann, b, f"{n} in NO-PARK", (60,60,255), 2); v_nopark += 1
        else:
            _draw_box(ann, b, n.title(), (0,180,255), 2)

    rr = _red_ratio(img[lr[1]:lr[3], lr[0]:lr[2]])
    red_on = rr >= cfg["red_ratio_thresh"]
    cv2.putText(ann, f"RED={'ON' if red_on else 'OFF'} ({rr:.2f})",
                (lr[0], max(10, lr[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,140,255), 2, cv2.LINE_AA)

    msgs = []
    if v_helmet: msgs.append("Violation: rider without helmet.")
    if v_nopark: msgs.append("Violation: vehicle in NO-PARK zone.")
    # image path cannot prove red-jump motion -> no count here
    v_red = 0

    print(f"[process_image] {time.time()-t0:.2f}s")
    return ann, DetectionSummary(
        vehicle_counts=counts,
        motorcycle_count=counts.get("motorcycle",0),
        no_helmet_violations=v_helmet,
        no_parking_violations=v_nopark,
        red_light_violations=v_red,
        messages=msgs
    )

# -------------------- VIDEO --------------------
def process_video(*args, **kwargs):
    raise NotImplementedError("Video disabled in fast/no-voice build. Use image processing for now.")