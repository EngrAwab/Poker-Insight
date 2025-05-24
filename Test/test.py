# # """
# # Detect → crop → query server → file-away.

# # Console output  :  <image_file> <server_reply>
# # Saved crops path :  DEST_ROOT/<server_reply>/<image_stem>_det<i>.png

# # Requirements: ultralytics, opencv-python, numpy, requests
# # """

# # from ultralytics import YOLO
# # import cv2
# # import numpy as np
# # from pathlib import Path
# # import requests, io, re

# # # ───────── USER CONFIGURATION ─────────
# # MODEL_WEIGHTS = r"D:\Programmes\Freelance\Poker-Insight\card.pt"
# # IMAGE_DIR     = Path(r"D:\Programmes\Freelance\Poker-Insight\DATASET\Test")

# # CONF_THRES    = 0.85           # YOLO confidence threshold
# # MAX_IMGS      = 50             # limit images processed
# # CROP_SIZE     = (256, 256)     # None → keep native size

# # SERVER_URL    = "http://192.168.41.160:5000/predict"
# # PROMPT        = ('''Please choose one of the given option based on the given iamge make sure to give the whole word
# # Ac – Ace of Clubs  
# # 2c Two of Clubs  
# # 3c Three of Clubs  
# # 4c Four of Clubs  
# # 5c Five of Clubs  
# # 6c Six of Clubs  
# # 7c Seven of Clubs  
# # 8c Eight of Clubs  
# # 9c Nine of Clubs  
# # 10c  Ten of Clubs  
# # Jc  Jack of Clubs  
# # Qc  Queen of Clubs  
# # Kc  King of Clubs  
# # As  Ace of Spades  
# # 2s  Two of Spades  
# # 3s  Three of Spades  
# # 4s  Four of Spades  
# # 5s  Five of Spades  
# # 6s  Six of Spades  
# # 7s  Seven of Spades  
# # 8s  Eight of Spades  
# # 9s  Nine of Spades  
# # 10s  Ten of Spades  
# # Js  Jack of Spades  
# # Qs  Queen of Spades  
# # Ks  King of Spades  
# # Ah  Ace of Hearts  
# # 2h  Two of Hearts  
# # 3h  Three of Hearts  
# # 4h  Four of Hearts  
# # 5h  Five of Hearts  
# # 6h  Six of Hearts  
# # 7h  Seven of Hearts  
# # 8h  Eight of Hearts  
# # 9h  Nine of Hearts  
# # 10h Ten of Hearts  
# # Jh  Jack of Hearts  
# # Qh  Queen of Hearts  
# # Kh  King of Hearts  
# # Ad  Ace of Diamonds  
# # 2d  Two of Diamonds  
# # 3d  Three of Diamonds  
# # 4d  Four of Diamonds  
# # 5d  Five of Diamonds  
# # 6d  Six of Diamonds  
# # 7d  Seven of Diamonds  
# # 8d  Eight of Diamonds  
# # 9d  Nine of Diamonds  
# # 10d Ten of Diamonds  
# # Jd  Jack of Diamonds  
# # Qd  Queen of Diamonds  
# # Kd  King of Diamonds
# # other''')
# # REQUEST_TIMEOUT = 30           # seconds

# # DEST_ROOT     = Path(r"D:\Programmes\Freelance\Poker-Insight\sorted_crops")
# # # ──────────────────────────────────────

# # # set of all allowed two-letter / three-letter codes, lower-case
# # VALID_CODES = {
# #     f"{n}{s}" for n in list("a23456789") + ["10", "j", "q", "k"]
# #                  for s in ("c", "s", "h", "d")
# # }

# # def send_to_server(img_bgr: np.ndarray,
# #                    url: str,
# #                    prompt: str,
# #                    *,
# #                    fmt: str = ".png",
# #                    timeout: int = 30,
# #                    session: requests.Session) -> str:
# #     """POST crop to *url* and return the server's reply (or '<SERVER_ERR>')."""
# #     ok, buf = cv2.imencode(fmt, img_bgr)
# #     if not ok:
# #         return "<ENCODE_ERR>"

# #     files = {
# #         "image": (
# #             f"upload{fmt}",
# #             io.BytesIO(buf.tobytes()),
# #             "image/png" if fmt == ".png" else "image/jpeg",
# #         )
# #     }
# #     data = {"prompt": prompt}

# #     try:
# #         resp = session.post(url, files=files, data=data, timeout=timeout)
# #         resp.raise_for_status()
# #         return resp.json().get("description", "").strip()
# #     except Exception:
# #         return "<SERVER_ERR>"

# # def extract_code(raw: str) -> str:
# #     """
# #     Return the first alnum token (lower-case) from *raw* – e.g. "2c" from
# #     "2C\nsure!", or '' if nothing sensible is found.
# #     """
# #     m = re.search(r"[A-Za-z0-9]+", raw)
# #     return m.group(0).lower() if m else ""

# # def main() -> None:
# #     DEST_ROOT.mkdir(parents=True, exist_ok=True)
# #     model = YOLO(MODEL_WEIGHTS)

# #     img_paths = sorted(
# #         p for p in IMAGE_DIR.glob("*")
# #         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# #     )[:MAX_IMGS]

# #     sess = requests.Session()

# #     for img_path in img_paths:
# #         frame = cv2.imread(str(img_path))
# #         if frame is None:
# #             continue  # silently skip unreadable files

# #         dets = model(frame, conf=CONF_THRES, verbose=False)[0]

# #         for det_i, box in enumerate(dets.boxes, start=1):
# #             x0, y0, x1, y1 = box.xyxy.int().tolist()[0]
# #             h, w = frame.shape[:2]
# #             x0, y0 = max(x0, 0), max(y0, 0)
# #             x1, y1 = min(x1, w), min(y1, h)
# #             crop = frame[y0:y1, x0:x1]
# #             if crop.size == 0:
# #                 continue

# #             if CROP_SIZE:
# #                 crop = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_CUBIC)

# #             reply = send_to_server(
# #                 crop,
# #                 SERVER_URL,
# #                 PROMPT,
# #                 fmt=".png",
# #                 timeout=REQUEST_TIMEOUT,
# #                 session=sess,
# #             )

# #             # —— console output (single line) ——
# #             print(f"{img_path.name} {reply}")

# #             # —— save crop if reply looks like a valid card code ——
# #             code = extract_code(reply)
# #             if code in VALID_CODES:
# #                 target_dir = DEST_ROOT / code
# #                 target_dir.mkdir(parents=True, exist_ok=True)
# #                 out_name = f"{img_path.stem}_det{det_i}.png"
# #                 cv2.imwrite(str(target_dir / out_name), crop)

# # if __name__ == "__main__":
# #     main()
# """
# Full pipeline:

# 1. Detect cards with YOLO (infer_boxes).
# 2. For every detection:
#       – crop (and optionally resize),
#       – send to Qwen-VL server,
#       – get text reply,
#       – turn reply into a safe Windows folder name,
#       – save the crop in DEST_ROOT / <folder> / <original>_det<i>.png
# 3. Log every step to the console.

# Author: ChatGPT (May 2025)
# """

# from __future__ import annotations

# import io
# import re
# from pathlib import Path
# from typing import List, Tuple, Optional

# import cv2
# import numpy as np
# import requests
# from ultralytics import YOLO

# # ───────── USER CONFIGURATION ─────────
# MODEL_WEIGHTS = r"D:\Programmes\Freelance\Poker-Insight\card.pt"
# IMAGE_DIR     = Path(r"D:\Programmes\Freelance\Poker-Insight\DATASET\DATASET")

# CONF_THRES    = 0.98          # YOLO confidence threshold
# MAX_IMGS      = None           # None = no limit
# CROP_SIZE     = (256, 256)     # None = leave original crop size

# SERVER_URL    = "http://192.168.133.160:5000/predict"
# PROMPT        = ('''Please give me the card name only don't send me anything else e.g if card is King of Spades then send King of Spades nothing else '''
# )
# REQUEST_TIMEOUT = 30           # seconds

# DEST_ROOT     = Path(r"D:\Programmes\Freelance\Poker-Insight\sorted_crops")
# # ──────────────────────────────────────


# # ───────── INFERENCE FUNCTION ─────────
# def infer_boxes(
#     model: YOLO,
#     bgr_img: np.ndarray,
#     conf: float = CONF_THRES
# ) -> List[Tuple[int, int, int, int]]:
#     """
#     Run YOLO on an already-loaded BGR image and return bounding-box coords.

#     Parameters
#     ----------
#     model : YOLO
#         A loaded ultralytics.YOLO model.
#     bgr_img : np.ndarray
#         OpenCV image in BGR ordering.
#     conf : float
#         Confidence threshold.

#     Returns
#     -------
#     List[Tuple[int, int, int, int]]
#         A list of (x0, y0, x1, y1) integers for every detection passing 'conf'.
#     """
#     result = model(bgr_img, conf=conf, verbose=False)[0]
#     boxes  = []
#     for b in result.boxes:
#         x0, y0, x1, y1 = b.xyxy.int().tolist()[0]
#         boxes.append((x0, y0, x1, y1))
#     return boxes


# # ───────── SERVER HELPER ─────────
# def send_crop_to_server(
#     crop_bgr: np.ndarray,
#     *,
#     url: str = SERVER_URL,
#     prompt: str = PROMPT,
#     timeout: int = REQUEST_TIMEOUT,
#     session: Optional[requests.Session] = None,
#     fmt: str = ".png"
# ) -> str:
#     """Send a crop to the Qwen-VL Flask endpoint and return the raw reply."""
#     ok, buf = cv2.imencode(fmt, crop_bgr)
#     if not ok:
#         raise ValueError("Failed to encode crop with OpenCV")

#     mime = "image/png" if fmt == ".png" else "image/jpeg"
#     files = {"image": (f"upload{fmt}", io.BytesIO(buf.tobytes()), mime)}
#     data  = {"prompt": prompt}
#     sess  = session or requests.Session()

#     try:
#         resp = sess.post(url, files=files, data=data, timeout=timeout)
#         resp.raise_for_status()
#         payload = resp.json()          # raises ValueError if not JSON
#     except (requests.RequestException, ValueError):
#         return "<<SERVER_DOWN>>"

#     raw = payload.get("description", "")
#     # If the backend embeds a mini-conversation, keep only the assistant part.
#     if "\nassistant\n" in raw:
#         raw = raw.split("\nassistant\n")[-1]
#     return raw.strip()


# # ───────── FOLDER-NAME SANITISER ─────────
# SAFE_CHARS = re.compile(r"[^a-z0-9]")      # allow only a-z, 0-9

# def safe_folder_name(reply: str) -> str:
#     """
#     Convert an arbitrary server reply into a legal Windows folder name.

#     * Take only the first line.
#     * Lower-case it.
#     * Remove all characters except a-z and 0-9.
#     * Fall back to 'other' if nothing remains.
#     """
#     token = SAFE_CHARS.sub("", reply.splitlines()[0].strip().lower())
#     return token or "other"


# # ───────── MAIN PIPELINE ─────────
# def main() -> None:
#     # 0) Create root directory if needed
#     DEST_ROOT.mkdir(parents=True, exist_ok=True)

#     # 1) Load YOLO model
#     model = YOLO(MODEL_WEIGHTS)
#     sess  = requests.Session()

#     # 2) Collect image paths
#     paths = sorted(
#         p for p in IMAGE_DIR.glob("*")
#         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#     )
#     if MAX_IMGS:
#         paths = paths[:MAX_IMGS]

#     # 3) Loop
#     for img_path in paths:
#         frame = cv2.imread(str(img_path))
#         if frame is None:
#             print(f"[WARN] Could not read {img_path}")
#             continue

#         # --- inference (coordinates) ---
#         boxes = infer_boxes(model, frame)

#         for det_i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
#             # clamp to image bounds
#             h, w = frame.shape[:2]
#             x0, y0 = max(x0, 0), max(y0, 0)
#             x1, y1 = min(x1, w), min(y1, h)

#             crop = frame[y0:y1, x0:x1]
#             if crop.size == 0:
#                 continue

#             if CROP_SIZE:
#                 crop = cv2.resize(crop, CROP_SIZE, interpolation=cv2.INTER_CUBIC)

#             # --- call Qwen-VL server ---
#             raw_reply = send_crop_to_server(crop, session=sess)
#             print (raw_reply)
#             folder    = safe_folder_name(raw_reply)

#             # --- save ---
#             target_dir = DEST_ROOT / folder
#             target_dir.mkdir(parents=True, exist_ok=True)

#             out_name = f"{img_path.stem}_det{det_i}.png"
#             cv2.imwrite(str(target_dir / out_name), crop)

#             # --- console log ---
#             print(f"{img_path.name}  →  raw: “{raw_reply}”, folder: {folder}")

#     print("[DONE] All images processed.")


# if __name__ == "__main__":
#     main()



#### TURN
# import cv2
# import numpy as np


# def detect_yellow_ring(image_path,
#                        min_r=30, max_r=150,
#                        debug=False):
#     """
#     Detect the gold circular player badge.

#     Parameters
#     ----------
#     image_path : str
#     min_r, max_r : int
#         Expected radius range for HoughCircles (pixels).
#         Tweak if your input frames are larger / smaller.
#     debug : bool
#         If True, return extra visualisations.

#     Returns
#     -------
#     found : bool
#         True  → yellow ring present
#         False → no ring
#     preview : np.ndarray or None
#         If debug, original frame with the detected ring drawn in green.
#     mask : np.ndarray
#         Binary mask of yellow regions (0 / 255).
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(image_path)

#     # --------------------------------------------------
#     # 1) Binary mask of yellow / gold pixels  (HSV range)
#     # --------------------------------------------------
#     hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower = np.array([15,  70,  70], np.uint8)   # tune Hue & Sat if needed
#     upper = np.array([65, 255, 255], np.uint8)
#     mask  = cv2.inRange(hsv, lower, upper)

#     # simple morphological cleanup – helps thin rings survive
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # --------------------------------------------------
#     # 2) Look for *round* shapes in the frame
#     # --------------------------------------------------
#     gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges   = cv2.Canny(gray, 100, 200)
#     circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,
#                                dp=1.2, minDist=50,
#                                param1=120, param2=30,
#                                minRadius=min_r, maxRadius=max_r)

#     found = False
#     best  = None

#     if circles is not None:
#         circles = np.round(circles[0]).astype(int)

#         # --------------------------------------------------
#         # 3) For each circle → check if its *ring* is yellow
#         # --------------------------------------------------
#         for (x, y, r) in circles:
#             annulus = np.zeros_like(mask)              # hollow ring mask
#             outer_r = int(r * 0.95)
#             inner_r = int(r * 0.70)                    # tune thickness here
#             cv2.circle(annulus, (x, y), outer_r, 255, -1)
#             cv2.circle(annulus, (x, y), inner_r,   0, -1)

#             # pixels that are both yellow *and* in the annulus
#             yellow_on_ring = cv2.bitwise_and(mask, mask, mask=annulus)
#             yellow_px      = cv2.countNonZero(yellow_on_ring)
#             ring_px        = cv2.countNonZero(annulus)

#             if ring_px == 0:
#                 continue
#             ratio = yellow_px / ring_px

#             if ratio > 0.25:      # ≥25 % of ring pixels are yellow → success
#                 found = True
#                 best  = (x, y, r)
#                 break

#     # ------------- DEBUG VISUALS -------------
#     if debug:
#         vis = img.copy()
#         if best:
#             x, y, r = best
#             cv2.circle(vis, (x, y), r, (0, 255, 0), 3)   # green outline
#         return found, vis, mask

#     # ------------- NON-DEBUG RETURN ----------
#     return found, None, mask


# # ----------------- quick CLI test -----------------
# if __name__ == "__main__":
#     ok, annotated, binary = detect_yellow_ring("D:\Programmes\Freelance\Poker-Insight\Kali_Pcker\combined_profiles.jpg", debug=True)
#     print("Yellow ring detected:", ok)

#     # show both images for quick inspection
#     cv2.imshow("binary yellow mask", binary)
#     cv2.imshow("annotated frame"  , annotated)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



#!/usr/bin/env python3
"""
Single-image BoVW classification (hard-coded paths, no user input)
"""

import os
import time
import cv2
import numpy as np
import joblib

# ---------- CONFIG (edit if needed) ------------------------------------------
TEST_IMAGE     = "test.PNG"            # path to the image you want to classify
CLASSES_FILE   = r"D:\Programmes\Freelance\Poker-Insight\Code+Model\classes.txt"         # one class name per line
KMEANS_PATH    = r"D:\Programmes\Freelance\Poker-Insight\Code+Model\kmeans_model.joblib" # trained k-means vocabulary
SVM_PATH       = r"D:\Programmes\Freelance\Poker-Insight\Code+Model\svm_model.joblib"    # trained SVM classifier
# -----------------------------------------------------------------------------


def load_class_names(file_path: str) -> list[str]:
    """Read class names from a text file (one per line)."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cannot find '{file_path}'.")
    with open(file_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise RuntimeError(f"No class names found in '{file_path}'.")
    return names


def extract_sift_descriptors(img_gray):
    """Compute SIFT descriptors from an already-loaded grayscale image."""
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img_gray, None)
    return descriptors


def build_bow_histogram(descriptors, kmeans):
    """Convert local SIFT descriptors into a normalized BoVW histogram."""
    n_clusters = kmeans.n_clusters
    hist = np.zeros(n_clusters, dtype=np.float32)

    if descriptors is not None and len(descriptors):
        cluster_ids = kmeans.predict(descriptors)
        counts, _ = np.histogram(cluster_ids, bins=np.arange(n_clusters + 1))
        hist = counts.astype(float) / (counts.sum() + 1e-7)

    return hist.reshape(1, -1)  # shape (1, n_clusters)


def main():
    # 1) Load resources -------------------------------------------------------
    class_names = load_class_names(CLASSES_FILE)
    kmeans      = joblib.load(KMEANS_PATH)
    svm         = joblib.load(SVM_PATH)

    # 2) Read test image ------------------------------------------------------
    img = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image '{TEST_IMAGE}'.")

    # 3) Inference ------------------------------------------------------------
    t0          = time.perf_counter()
    descriptors = extract_sift_descriptors(img)
    hist        = build_bow_histogram(descriptors, kmeans)
    idx         = svm.predict(hist)[0]
    elapsed     = time.perf_counter() - t0

    label = class_names[idx] if 0 <= idx < len(class_names) else f"Unknown({idx})"
    print(f"[{os.path.basename(TEST_IMAGE)}] → {label}   (elapsed {elapsed*1000:.1f} ms)")


if __name__ == "__main__":
    main()
