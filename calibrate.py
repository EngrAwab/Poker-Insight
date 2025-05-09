import cv2
import os
from paddleocr import PaddleOCR
VIDEO_PATH   = r'D:\Programmes\Freelance\Poker-Insight\Game\videoplayback.mp4'
OUTPUT_TXT   = 'boxes.txt'
MAX_WIDTH    = 1280
MAX_HEIGHT   = 800
BTN_W, BTN_H = 90, 40
FONT         = cv2.FONT_HERSHEY_SIMPLEX
CLASSES = {
    1: ('Player',     'P', (  0,255,255)),   # yellow
    2: ('Name',       'N', (255,  0,255)),   # magenta  → OCR
    3: ('Investment', 'I', (255,255,  0)),   # cyan     → OCR
}
# ───────────── LABEL / EDIT MODE ─────────────
def print_ocr(frame, rect, ocr):
    """Save crop, run OCR, print result or 'Nothing'."""
    global ocr_count
    x1, y1, x2, y2, cl = rect
    crop = frame[max(0, y1):min(frame.shape[0], y2),
                 max(0, x1):min(frame.shape[1], x2)]

    os.makedirs('ocr_crops', exist_ok=True)
    cv2.imwrite(f'ocr_crops/ocr_crop_{ocr_count}.png', crop)
    ocr_count += 1

    if crop.size < 10:
        print('Nothing')
        return

    try:
        res = ocr.ocr(crop, cls=True)
    except Exception as e:
        print('OCR error:', e)
        return
    try:
        for line in res:
            for word_info in line:
                text = word_info[1][0]
                print(text)
    except:
        print ("FFF")
def inside(x, y, x1, y1, x2, y2):
    return x1 <= x <= x2 and y1 <= y <= y2
def vid_props(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return w, h, fps
def calculate_scale(orig_w, orig_h):
    return min(MAX_WIDTH / orig_w, MAX_HEIGHT / orig_h)


def scale_coords(x, y, s):
    return int(x * s), int(y * s)

def corner_hit(x, y, rect, tol=10):
    x1, y1, x2, y2, _ = rect
    for i, (cx, cy) in enumerate([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]):
        if abs(x - cx) <= tol and abs(y - cy) <= tol:
            return i
    return None
def unscale_coords(x, y, s):
    return int(x / s), int(y / s)
def run_labeler_mode(existing=None):
    boxes = {f: list(rs) for f, rs in (existing or {}).items()}
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError('Cannot open video file')

    w, h, fps = vid_props(cap)
    s = calculate_scale(w, h)
    dw, dh = int(w * s), int(h * s)
    delay = int(1000 / (fps * 2))

    paused = finished = delete_mode = False
    drawing = moving = resizing = False
    sp = cp = off = None
    sel_idx = sel_corner = None
    cur_cls = 1

    ocr = PaddleOCR(use_angle_cls=False, show_log=False)

    def on_mouse(ev, x, y, flags, param):
        nonlocal paused, finished, delete_mode
        nonlocal drawing, moving, resizing, sp, cp, off
        nonlocal sel_idx, sel_corner

        ox, oy = unscale_coords(x, y, s)

        if ev == cv2.EVENT_LBUTTONDOWN:
            if inside(x, y, 10, 10, 10 + BTN_W, 10 + BTN_H):
                paused = not paused
                return
            if inside(x, y, 120, 10, 120 + BTN_W, 10 + BTN_H):
                finished = True
                return
            if inside(x, y, 230, 10, 230 + BTN_W, 10 + BTN_H):
                delete_mode = not delete_mode
                return
            if not paused:
                return

            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            fb = boxes.setdefault(fid, [])

            for i, r in enumerate(fb):
                c = corner_hit(ox, oy, r)
                if c is not None:
                    sel_idx, sel_corner = i, c
                    resizing = True
                    return

            for i, (x1, y1, x2, y2, cl) in enumerate(fb):
                if inside(ox, oy, x1, y1, x2, y2):
                    if delete_mode:
                        del fb[i]
                        return
                    moving = True
                    sel_idx = i
                    off = (ox - x1, oy - y1)
                    return

            drawing = True
            sp = cp = (ox, oy)

        elif ev == cv2.EVENT_MOUSEMOVE:
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if drawing:
                cp = (ox, oy)
            elif resizing:
                x1, y1, x2, y2, cl = boxes[fid][sel_idx]
                if sel_corner == 0:
                    x1, y1 = ox, oy
                elif sel_corner == 1:
                    x2, y1 = ox, oy
                elif sel_corner == 2:
                    x2, y2 = ox, oy
                else:
                    x1, y2 = ox, oy
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                boxes[fid][sel_idx] = (x1, y1, x2, y2, cl)
            elif moving:
                x1, y1, x2, y2, cl = boxes[fid][sel_idx]
                w_ = x2 - x1
                h_ = y2 - y1
                x1 = ox - off[0]
                y1 = oy - off[1]
                boxes[fid][sel_idx] = (x1, y1, x1 + w_, y1 + h_, cl)

        elif ev == cv2.EVENT_LBUTTONUP:
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if drawing:
                x1, y1 = sp
                x2, y2 = cp
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    x1, x2 = sorted((x1, x2))
                    y1, y2 = sorted((y1, y2))
                    rect = (x1, y1, x2, y2, cur_cls)
                    boxes.setdefault(fid, []).append(rect)
                    if cur_cls in (2, 3):
                        print_ocr(frame, rect, ocr)
            drawing = moving = resizing = False
            sel_idx = sel_corner = None

    cv2.namedWindow('Video Labeler', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Labeler', dw, dh + BTN_H + 20)
    cv2.setMouseCallback('Video Labeler', on_mouse)

    while cap.isOpened() and not finished:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        disp = cv2.resize(frame, (dw, dh))
        # buttons
        cv2.rectangle(disp, (10, 10), (10 + BTN_W, 10 + BTN_H),
                      (0, 255, 0) if paused else (0, 0, 255), -1)
        cv2.putText(disp, 'Resume' if paused else 'Pause',
                    (16, 10 + BTN_H - 10), FONT, 0.55, (255, 255, 255), 1)
        cv2.rectangle(disp, (120, 10), (120 + BTN_W, 10 + BTN_H), (255, 0, 0), -1)
        cv2.putText(disp, 'Finish', (126, 10 + BTN_H - 10),
                    FONT, 0.55, (255, 255, 255), 1)
        cv2.rectangle(disp, (230, 10), (230 + BTN_W, 10 + BTN_H),
                      (0, 0, 255) if delete_mode else (255, 0, 255), -1)
        cv2.putText(disp, 'Delete', (236, 10 + BTN_H - 10),
                    FONT, 0.55, (255, 255, 255), 1)

        # draw boxes
        for idx, (x1, y1, x2, y2, cl) in enumerate(boxes.get(fid, []), 1):
            col = CLASSES[cl][2]
            sx1, sy1 = scale_coords(x1, y1, s)
            sx2, sy2 = scale_coords(x2, y2, s)
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), col, 2)
            cv2.putText(disp, f'{idx}-{CLASSES[cl][1]}',
                        (sx1 + 3, sy1 + 18), FONT, 0.6, col, 2)

        if drawing:
            cv2.rectangle(disp, scale_coords(*sp, s),
                          scale_coords(*cp, s), CLASSES[cur_cls][2], 1)

        cv2.imshow('Video Labeler', disp)

        k = cv2.waitKey(delay if not paused else 30) & 0xFF
        if k == ord('q'):
            finished = True
        elif k in (ord('1'), ord('2'), ord('3')):
            cur_cls = int(chr(k))

    cap.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_TXT, 'w') as f:
        for frm, rs in boxes.items():
            for x1, y1, x2, y2, cl in rs:
                f.write(f'{frm},{x1},{y1},{x2},{y2},{cl}\n')
    print('Saved:', os.path.abspath(OUTPUT_TXT))

