import cv2
import os
import numpy as np
from paddleocr import PaddleOCR
from calibrate import run_labeler_mode   # your existing label-editing helper

# ╔══════════════════════════ SETTINGS ═════════════════════════╗
VIDEO_PATH   = r'D:\Programmes\Freelance\Poker-Insight\Game\videoplayback.mp4'
OUTPUT_TXT   = 'boxes.txt'

MAX_WIDTH    = 1280
MAX_HEIGHT   = 800
FONT         = cv2.FONT_HERSHEY_SIMPLEX
COORD_SCALE  = 2           # scale factor applied when loading boxes

# HSV range for colour tracking / line detection
LOWER_HSV = np.array([28,  84, 100])
UPPER_HSV = np.array([68, 200, 255])

# 1-based class_id → (name, short-label, BGR colour)
CLASSES = {
    1: ('Player',     'P', (  0,255,255)),  # yellow  → blue if on turn
    2: ('Name',       'N', (255,  0,255)),  # magenta → OCR
    3: ('Investment', 'I', (255,255,  0)),  # cyan    → OCR
}

# --- status words ----------------------------------------------------------
KEY_STATUSES = {'fold', 'all in', 'raise' , 'bet', 'call' , 'check'}   # accepted “action” words
ANTE_WORDS   = {'ante', 'antee'}             # triggers new round
# ╚═════════════════════════════════════════════════════════════╝
# ───────────────────────────────────────────────────────────────


def vid_props(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS) or 30)


def calculate_scale(w, h):
    return min(MAX_WIDTH / w, MAX_HEIGHT / h)


def scale_coords(x, y, s):
    return int(x * s), int(y * s)


def load_existing_boxes():
    """Read rectangles saved by the labeler (scale back to full-res)."""
    boxes = {}
    if os.path.exists(OUTPUT_TXT):
        with open(OUTPUT_TXT) as f:
            for ln in f:
                frm, x1, y1, x2, y2, cl = map(int, ln.strip().split(','))
                if cl in (0, 1, 2):          # shift old 0-based IDs forward
                    cl += 1
                boxes.setdefault(frm, []).append(
                    (int(x1 * COORD_SCALE), int(y1 * COORD_SCALE),
                     int(x2 * COORD_SCALE), int(y2 * COORD_SCALE), cl))
    return boxes


def ocr_text(frame, rect, ocr):
    """Run OCR on a cropped rectangle; return plain text ('' if nothing)."""
    x1, y1, x2, y2, _ = rect
    crop = frame[max(0, y1):min(frame.shape[0], y2),
                 max(0, x1):min(frame.shape[1], x2)]
    if crop.size < 10:
        return ''
    try:
        res = ocr.ocr(crop, cls=True)
    except Exception:
        return ''
    words = []
    for line in res or []:
        for w in line or []:
            try:
                t = w[1][0]
                if t:
                    words.append(t)
            except Exception:
                continue
    return ' '.join(words).strip()


# ───────────── VIEW-ONLY MODE ─────────────
def run_viewer(boxes):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    w, h, fps = vid_props(cap)
    fps_int = max(1, int(round(fps)))      # ≈1-second frame interval
    s = calculate_scale(w, h)
    disp_w, disp_h = int(w * s), int(h * s)

    # static lookup tables
    flat = [r for rs in boxes.values() for r in rs]
    players_idx = [(idx, r) for idx, r in enumerate(flat, 1) if r[4] == 1]
    rect2idx = {tuple(r): idx for idx, r in enumerate(flat, 1)}

    ocr = PaddleOCR(use_angle_cls=False, show_log=False)

    cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Viewer", disp_w, disp_h)
    cv2.namedWindow("Info", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Info", 750, 400)      # widened for “Round” header

    # persistent state ------------------------------------------------------
    info_total = {}        # {pid: {'name':…, 'inv':…}}
    player_status = {}     # {pid: status string ('-', 'Fold', …)}
    round_num = 1
    last_ocr_frame = -fps_int               # force OCR on very first frame

    # ----------------------------------------------------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # ---------- find green BB (whose turn?) ----------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        green_bb = None
        if contours:
            big = max(contours, key=cv2.contourArea)
            if cv2.contourArea(big) > 100:
                x, y, wc, hc = cv2.boundingRect(big)
                # cv2.rectangle(frame, (x, y), (x + wc, y + hc), (0, 255, 0), 2)
                green_bb = (x, y, x + wc, y + hc)

        # ---------- turn status ----------
        turn_status = {}
        for pid, (x1, y1, x2, y2, _) in players_idx:
            inside = (green_bb and green_bb[0] >= x1 and green_bb[1] >= y1
                      and green_bb[2] <= x2 and green_bb[3] <= y2)
            turn_status[pid] = 'Yes' if inside else 'No'

        # ---------- decide if we will OCR heavy (status) this frame ----------
        ocr_this_frame = (f_idx - last_ocr_frame) >= fps_int
        if ocr_this_frame:
            last_ocr_frame = f_idx

        # ---------- PASS 1: basic Name / Investment OCR  (RUN EVERY FRAME) ----------
        for rect in boxes.get(f_idx, []):
            x1, y1, x2, y2, cl = rect
            parent_id = None
            if cl != 1:
                for p_idx, (px1, py1, px2, py2, _) in players_idx:
                    if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                        parent_id = p_idx
                        break
            pid = parent_id or rect2idx.get(tuple(rect), 0)
            if cl in (2, 3):
                # Only a handful of rectangles on labelled frames ⇒ cheap.
                txt = ocr_text(frame, rect, ocr)
                if txt:
                    entry = info_total.setdefault(pid, {'name': '', 'inv': ''})
                    if cl == 2 and not entry['name']:
                        entry['name'] = txt
                    elif cl == 3 and not entry['inv']:
                        entry['inv'] = txt

        # ---------- PASS 2: status / round handling (ONCE PER SECOND) ----------
        ante_detected = False
        if ocr_this_frame and f_idx > 85:   # skip early warm-up
            for (x1, y1, x2, y2, cl) in flat:
                if cl != 2:
                    continue   # only Name rectangles

                # parent player
                parent_id = None
                for p_idx, (px1, py1, px2, py2, _) in players_idx:
                    if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                        parent_id = p_idx
                        break
                if not parent_id:
                    continue

                txt = ocr_text(frame, (x1, y1, x2, y2, cl), ocr)
                if not txt:
                    continue
                txt_l = txt.strip().lower()

                # ----- “ante” triggers new round --------------------------
                if txt_l in ANTE_WORDS:
                    ante_detected = True
                    break

                # ----- normal status words --------------------------------
                base_name = info_total.get(parent_id, {}).get('name', '').lower()
                current   = player_status.get(parent_id, '').lower()
                if (txt_l in KEY_STATUSES
                        and txt_l != base_name
                        and txt_l != current):
                    player_status[parent_id] = txt  # exact spelling

        # ---------- handle new round if “ante” was seen ----------
        if ante_detected:
            round_num += 1
            for pid, _ in players_idx:
                player_status[pid] = '-'      # reset every player’s status

        # ---------- draw rectangles ----------
        disp = cv2.resize(frame.copy(), (disp_w, disp_h))
        for i, (x1, y1, x2, y2, cl) in enumerate(flat, 1):
            # colour / parent lookup
            parent_id = None
            if cl != 1:
                for p_idx, (px1, py1, px2, py2, _) in players_idx:
                    if x1 >= px1 and y1 >= py1 and x2 <= px2 and y2 <= py2:
                        parent_id = p_idx
                        break

            if cl == 1 and turn_status.get(i) == 'Yes':
                col = (255, 0, 0)          # blue if that player’s turn
            else:
                col = (0, 0, 255) if parent_id else CLASSES[cl][2]

            lbl_id = parent_id or i
            sx1, sy1 = scale_coords(x1, y1, s)
            sx2, sy2 = scale_coords(x2, y2, s)
            # cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), col, 2)
            # cv2.putText(disp, f'{lbl_id}-{CLASSES[cl][1]}',
            #             (sx1 + 3, sy1 + 18), FONT, 0.6, col, 2)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            # cv2.circle(disp, scale_coords(mx, my, s), 4, col, -1)

        # ---------- Info window ----------
        if info_total:
            lines = [f"Round : {round_num}"]   # header
            for pid, v in sorted(info_total.items()):
                if not (v['name'] or v['inv']):
                    continue
                turn   = turn_status.get(pid, 'No')
                status = player_status.get(pid,
                           'Playing' if turn == 'Yes' else 'Waiting')
                lines.append(f"{v['name']} : {v['inv']} : {turn} : {status}")

            h = max(60, 30 * len(lines) + 20)
            canvas = np.zeros((h, 750, 3), dtype=np.uint8)
            y = 25
            for ln in lines:
                cv2.putText(canvas, ln, (10, y), FONT, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                y += 30
        else:
            canvas = np.zeros((60, 750, 3), dtype=np.uint8)
            cv2.putText(canvas, 'No data', (10, 35), FONT, 0.7,
                        (255, 255, 255), 1)

        cv2.imshow("Viewer", disp)
        cv2.imshow("Info", canvas)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ───────────── MAIN ─────────────
if __name__ == "__main__":
    boxes = load_existing_boxes()
    if boxes:
        print("1) View  2) Edit")
        if input("Choice: ").strip() == '1':
            run_viewer(boxes)
        else:
            run_labeler_mode(boxes)
    else:
        run_labeler_mode()
