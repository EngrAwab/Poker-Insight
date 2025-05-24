import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import Counter
import joblib
import os

# ---------------------------------------------------------------------------
#  Paths & ROI loading
# ---------------------------------------------------------------------------
ROI_FILE       = 'boxes.txt'
CLASSES_FILE   = r"D:\Programmes\Freelance\Poker-Insight\Code+Model\classes.txt"
KMEANS_PATH    = r"D:\Programmes\Freelance\Poker-Insight\Code+Model\kmeans_model.joblib"
SVM_PATH       = r"D:\Programmes\Freelance\Poker-Insight\Code+Model\svm_model.joblib"

rois = []
try:
    with open(ROI_FILE, 'r') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 5:
                _, x1, y1, x2, y2 = p[:5]
                rois.append(tuple(map(int, (x1, y1, x2, y2))))
    print(f"Loaded {len(rois)} ROIs from {ROI_FILE}")
except FileNotFoundError:
    print(f"ROI file {ROI_FILE} not found. No ROIs loaded.")

# ---------------------------------------------------------------------------
#  Model initialisation
# ---------------------------------------------------------------------------
model = YOLO(r"D:\Programmes\Freelance\Poker-Insight\Models\Full.pt")
ocr   = PaddleOCR(use_angle_cls=False, lang="en")

with open(CLASSES_FILE, encoding="utf-8") as f:
    class_names = [ln.strip() for ln in f if ln.strip()]

kmeans = joblib.load(KMEANS_PATH)
svm    = joblib.load(SVM_PATH)

# ---------------------------------------------------------------------------
#  Card-classification helpers
# ---------------------------------------------------------------------------
def sift_desc(img_gray):
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(img_gray, None)
    return desc

def bow_hist(desc):
    k = kmeans.n_clusters
    h = np.zeros(k, np.float32)
    if desc is not None and len(desc):
        ids = kmeans.predict(desc)
        counts, _ = np.histogram(ids, bins=np.arange(k + 1))
        h = counts.astype(float) / (counts.sum() + 1e-7)
    return h.reshape(1, -1)

def infer_card(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    idx = svm.predict(bow_hist(sift_desc(img)))[0]
    return class_names[idx] if 0 <= idx < len(class_names) else f"Unknown({idx})"

# ---------------------------------------------------------------------------
#  Video capture setup
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(r"D:\Programmes\Freelance\Poker-Insight\videoplayback720.mp4")
if not cap.isOpened():
    raise RuntimeError("Cannot open video file.")

conf_th   = 0.85
start_min = 47
pad_r     = 0.1
exp_r     = 0.2
skip_r    = 4

fps = cap.get(cv2.CAP_PROP_FPS) or 1
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * start_min * 60))

# ---------------------------------------------------------------------------
#  Buffers / bookkeeping
# ---------------------------------------------------------------------------
buffers = {i: {'name': [], 'investment': [], 'state': []} for i in range(len(rois))}
name_by_roi: dict[int, str] = {}
extracted: dict[str, dict[str, str]] = {}      # per-player info (now with 'card')

def try_commit(i, field):
    buf = buffers[i][field]
    if len(buf) < 5:
        return
    top, cnt = Counter(buf).most_common(1)[0]
    if len(buf) == 5 and cnt < 3:
        return
    buffers[i][field].clear()
    if field == 'name':
        name_by_roi[i] = top
        extracted.setdefault(top, {'investment': '-', 'state': '-', 'card': '-'})
    else:
        if i in name_by_roi:
            extracted[name_by_roi[i]][field] = top

# ---------------------------------------------------------------------------
#  Main processing loop
# ---------------------------------------------------------------------------
frame_c = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_c += 1
    if frame_c % skip_r:
        continue

    annotated = frame.copy()
    h, w = frame.shape[:2]
    card_labels_global = []                       # for top-left overlay

    # -----------------------------------------------------------------------
    #  Iterate over ROIs
    # -----------------------------------------------------------------------
    for i, (sx, sy, ex, ey) in enumerate(rois):
        x1, y1 = max(0, sx), max(0, sy)
        x2, y2 = min(w, ex), min(h, ey)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        for box in model(crop)[0].boxes:
            if float(box.conf[0]) < conf_th:
                continue
            cls = model.names[int(box.cls[0])].lower()

            # -------------------- Card branch ------------------------------
            if cls == 'card':
                b = list(map(int, box.xyxy[0]))
                card_crop = crop[b[1]:b[3], b[0]:b[2]]
                if card_crop.size == 0:
                    continue
                card_name = infer_card(card_crop)
                card_labels_global.append(card_name)

                # store per-player card
                if i in name_by_roi:
                    pl = name_by_roi[i]
                    if 'card' not in extracted[pl] or extracted[pl]['card'] == '-':
                        extracted[pl]['card'] = card_name
                    elif card_name not in extracted[pl]['card']:
                        extracted[pl]['card'] += f", {card_name}"

                ax1, ay1 = x1 + b[0], y1 + b[1]
                ax2, ay2 = x1 + b[2], y1 + b[3]
                cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2)
                continue

            # ---------------- Name / Investment / State --------------------
            if cls not in ('name', 'investment', 'state'):
                continue
            b = list(map(int, box.xyxy[0]))
            bw, bh = b[2] - b[0], b[3] - b[1]
            ex_pad, ey_pad = int(bw * exp_r), int(bh * exp_r)
            ebx1, eby1 = max(0, b[0] - ex_pad), max(0, b[1] - ey_pad)
            ebx2, eby2 = min(crop.shape[1], b[2] + ex_pad), min(crop.shape[0], b[3] + ey_pad)
            pw, ph = int((ebx2 - ebx1) * pad_r), int((eby2 - eby1) * pad_r)
            px1, py1 = max(0, ebx1 - pw), max(0, eby1 - ph)
            px2, py2 = min(crop.shape[1], ebx2 + pw), min(crop.shape[0], eby2 + ph)
            roi_crop = crop[py1:py2, px1:px2]

            try:
                res = ocr.ocr(roi_crop, cls=True)
                txt = res[0][0][1][0].strip() if res else ''
            except Exception:
                txt = ''
            if not txt:
                continue

            buffers[i][cls].append(txt)
            try_commit(i, cls)

            ax1, ay1 = x1 + px1, y1 + py1
            ax2, ay2 = x1 + px2, y1 + py2
            cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
            cv2.putText(annotated, cls.capitalize(), (ax1, ay1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # -----------------------------------------------------------------------
    #  Overlay all card names (top-left of main frame)
    # -----------------------------------------------------------------------
    # for j, nm in enumerate(card_labels_global):
    #     cv2.putText(annotated, nm, (10, 30 + j * 25),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # -----------------------------------------------------------------------
    #  Build summary window with per-player Card column
    # -----------------------------------------------------------------------
    lines = [
        f"Name: {p} || Investment: {v['investment']} || State: {v['state']} || Card: {v.get('card','-')}"
        for p, v in extracted.items()
    ]
    line_h, win_w = 25, 1200
    win_h = line_h * max(1, len(lines)) + 20
    summary = np.full((win_h, win_w, 3), 255, np.uint8)
    for k, ln in enumerate(lines):
        cv2.putText(summary, ln, (10, 20 + k * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # -----------------------------------------------------------------------
    #  Display
    # -----------------------------------------------------------------------
    cv2.imshow("YOLO+OCR in ROIs", annotated)
    cv2.imshow("ROI Summary", summary)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
