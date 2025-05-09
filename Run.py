import cv2
import os
import numpy as np
from paddleocr import PaddleOCR
from calibrate import run_labeler_mode

# ╔══════════════════════════ SETTINGS ═════════════════════════╗
VIDEO_PATH   = r'D:\Programmes\Freelance\Poker-Insight\Game\videoplayback.mp4'
OUTPUT_TXT   = 'boxes.txt'
MAX_WIDTH    = 1280
MAX_HEIGHT   = 800
FONT         = cv2.FONT_HERSHEY_SIMPLEX
COORD_SCALE  = 2           # scale factor for old coordinates

# 1-based class_id → (name, label, BGR colour)
CLASSES = {
    1: ('Player',     'P', (  0,255,255)),    # yellow
    2: ('Name',       'N', (255,  0,255)),    # magenta  → OCR
    3: ('Investment', 'I', (255,255,  0)),    # cyan     → OCR
}
# ╚═════════════════════════════════════════════════════════════╝
# ───────────────────────────────────────────────────────────────
def vid_props(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 30)


def calculate_scale(orig_w, orig_h):
    return min(MAX_WIDTH / orig_w, MAX_HEIGHT / orig_h)


def scale_coords(x, y, s):
    return int(x * s), int(y * s)


def load_existing_boxes():
    boxes = {}
    if os.path.exists(OUTPUT_TXT):
        with open(OUTPUT_TXT) as f:
            for ln in f:
                frm, x1, y1, x2, y2, cl = map(int, ln.strip().split(','))
                if cl in (0, 1, 2):  # shift any old 0-based IDs
                    cl += 1
                boxes.setdefault(frm, []).append(
                    (int(x1 * COORD_SCALE), int(y1 * COORD_SCALE),
                     int(x2 * COORD_SCALE), int(y2 * COORD_SCALE), cl))
    return boxes
# ───────────────── OCR helper (original logic + guards) ─────────────────
def ocr_text(frame, rect, ocr):
    x1, y1, x2, y2, _ = rect
    crop = frame[max(0, y1):min(frame.shape[0], y2),
                 max(0, x1):min(frame.shape[1], x2)]
    if crop.size < 10:
        return ''
    try:
        res = ocr.ocr(crop, cls=True)
    except Exception:
        return ''
    txt_parts = []
    for line in res or []:
        for w in line or []:
            try:
                t = w[1][0]
                if t:
                    txt_parts.append(t)
            except Exception:
                continue
    return ' '.join(txt_parts).strip()
# ───────────── VIEW-ONLY MODE ─────────────
def run_viewer(boxes):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError('Cannot open video file')

    w, h, fps = vid_props(cap)
    s, (dw, dh) = calculate_scale(w, h), (int(w * calculate_scale(w, h)),
                                          int(h * calculate_scale(w, h)))

    flat        = [r for rs in boxes.values() for r in rs]
    players_idx = [(idx, r) for idx, r in enumerate(flat, 1) if r[4] == 1]
    rect2idx    = {tuple(r): idx for idx, r in enumerate(flat, 1)}

    ocr = PaddleOCR(use_angle_cls=False, show_log=False)

    cv2.namedWindow('Video Viewer', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Viewer', dw, dh)
    cv2.namedWindow('Info', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Info', 400, 400)

    info_total = {}                  # ── NEW (persistent log) ──

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        disp = cv2.resize(frame, (dw, dh))

        # ---- per-frame scan for new OCR text ----
        for rect in boxes.get(f_idx, []):
            x1, y1, x2, y2, cl = rect
            parent_id = None
            if cl != 1:
                for p_idx, (px1, py1, px2, py2, _) in players_idx:
                    if x1 >= px1 <= px2 >= x2 and y1 >= py1 <= py2 >= y2:
                        parent_id = p_idx
                        break
            label_id = parent_id or rect2idx.get(tuple(rect), 0)

            if cl in (2, 3):
                txt = ocr_text(frame, rect, ocr)
                if txt:
                    entry = info_total.setdefault(           # ── NEW
                        label_id, {'name': '', 'inv': ''})
                    if cl == 2:
                        entry['name'] = txt
                    else:
                        entry['inv'] = txt

        # ---- draw rectangles ----
        for i, (x1, y1, x2, y2, cl) in enumerate(flat, 1):
            parent_id = None
            if cl != 1:
                for p_idx, (px1, py1, px2, py2, _) in players_idx:
                    if x1 >= px1 <= px2 >= x2 and y1 >= py1 <= py2 >= y2:
                        parent_id = p_idx
                        break
            inside = parent_id is not None
            col    = (0, 0, 255) if inside else CLASSES[cl][2]
            lbl_id = parent_id or i

            sx1, sy1 = scale_coords(x1, y1, s)
            sx2, sy2 = scale_coords(x2, y2, s)
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), col, 2)
            cv2.putText(disp, f'{lbl_id}-{CLASSES[cl][1]}',
                        (sx1 + 3, sy1 + 18), FONT, 0.6, col, 2)
            mx, my   = (x1 + x2)//2, (y1 + y2)//2
            cv2.circle(disp, scale_coords(mx, my, s), 4, col, -1)

        cv2.putText(disp, f'Frame: {f_idx}', (10, 30), FONT,
                    0.7, (0, 255, 0), 2)
        cv2.imshow('Video Viewer', disp)

        # ---- build persistent Info window ----
        if info_total:
            lines = [f"{v['name']} : {v['inv']}"
                     for _, v in sorted(info_total.items())
                     if v['name'] or v['inv']]
            h_can  = max(60, 30*len(lines)+20)
            canvas = np.zeros((h_can, 400, 3), dtype=np.uint8)
            y = 25
            for ln in lines:
                cv2.putText(canvas, ln, (10, y), FONT, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                y += 30
        else:
            canvas = np.zeros((60, 400, 3), dtype=np.uint8)
            cv2.putText(canvas, 'No data', (10, 35), FONT, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Info', canvas)                 # ── NEW

        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# ───────────── MAIN ─────────────
if __name__ == '__main__':
    existing = load_existing_boxes()
    if existing:
        print('1) View  2) Edit')
        ch = input('Choice: ').strip()
        run_viewer(existing) if ch == '1' else run_labeler_mode(existing)
    else:
        run_labeler_mode()
