# # import cv2
# # import time
# # from ultralytics import YOLO

# # def display_and_annotate(input_video_path, model_path):
# #     # Load the YOLO model
# #     model = YOLO(model_path)

# #     # Open video
# #     cap = cv2.VideoCapture(input_video_path)
# #     if not cap.isOpened():
# #         raise IOError(f"Cannot open video file {input_video_path}")

# #     # Get video FPS to calculate display and inference intervals
# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     fps = fps if fps > 0 else 30
# #     display_delay_ms = int(1000 / fps)
# #     inference_interval = 0.5  # seconds, twice per second

# #     last_inference_time = 0
# #     annotated_frame = None

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break  # Video ended

# #         current_time = time.time()
# #         # Perform inference at defined interval
# #         if current_time - last_inference_time >= inference_interval:
# #             results = model(frame)[0]

# #             # Annotate detections
# #             annotated_frame = frame.copy()
# #             for box in results.boxes:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
# #                 conf = float(box.conf[0])
# #                 cls_id = int(box.cls[0])
# #                 label = f"{model.names[cls_id]} {conf:.2f}"
# #                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
# #                             0.5, (0, 255, 0), 2)

# #             last_inference_time = current_time

# #         # Display annotated frame if available, else original
# #         display_frame = annotated_frame if annotated_frame is not None else frame
# #         cv2.imshow("YOLO Inference", display_frame)

# #         # Wait to match original video pace
# #         if cv2.waitKey(display_delay_ms) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     input_video = r"D:\Programmes\Freelance\Poker-Insight\Game\videoplayback.mp4"
# #     model_path = r"D:\Programmes\Freelance\Poker-Insight\card.pt"
# #     display_and_annotate(input_video, model_path)

# # # Requirements:
# # # pip install ultralytics opencv-python


# import os

# # Specify the directory to scan
# root_dir = r"D:\Programmes\Freelance\Poker-Insight\sorted_crops"

# # Common image file extensions
# def is_image_file(filename):
#     extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
#     return filename.lower().endswith(extensions)

# # Count images recursively
# def count_images(directory):
#     count = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for fname in filenames:
#             if is_image_file(fname):
#                 count += 1
#     return count

# if __name__ == "__main__":
#     total_images = count_images(root_dir)
#     print(f"Total image files in '{root_dir}' and its subdirectories: {total_images}")
import os
import math
import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Define supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def is_image_file(fname):
    return os.path.splitext(fname.lower())[1] in IMAGE_EXTENSIONS

def process_and_grid_images(folder_path, model_path, pad_ratio=0.1, max_images=20, cols=5):
    """
    Processes up to max_images in folder_path with YOLO+OCR,
    then arranges them in a grid with given number of columns.
    """
    # Initialize YOLO and OCR
    model = YOLO(model_path)
    ocr = PaddleOCR(use_angle_cls=False, lang="en")

    # Gather image files
    all_files = sorted(f for f in os.listdir(folder_path) if is_image_file(f))
    selected = all_files[:max_images]
    if not selected:
        raise ValueError(f"No image files found in {folder_path}")

    processed = []
    for fname in selected:
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # YOLO inference
        results = model(img)[0]
        annotated = img.copy()
        texts = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label.lower() in {"card", "profile"}:
                continue

            # Padding
            bw, bh = x2 - x1, y2 - y1
            pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
            x1p = max(0, x1 - pad_w)
            y1p = max(0, y1 - pad_h)
            x2p = min(w - 1, x2 + pad_w)
            y2p = min(h - 1, y2 + pad_h)
            if x2p <= x1p or y2p <= y1p:
                continue

            crop = img[y1p:y2p, x1p:x2p]
            start = time.time()
            ocr_res = ocr.ocr(crop, cls=True)
            print (ocr_res)
            duration = time.time() - start

            # Robust text extraction: find first tuple element in each OCR line
            text_line = ""
            if ocr_res:
                # for entry in ocr_res:
                #     if not isinstance(entry, (list, tuple)):
                #         continue
                #     # Search for a tuple with the text
                #     for elem in entry:
                #         if isinstance(elem, tuple) and len(elem) >= 1:
                #             text_line = elem[0]
                #             break
                try:
                    text_line=(ocr_res[0][0][1][0])
                except:
                    text_line = ''
                    print (ocr_res)
                    # if text_line:
                    #     break
            texts.append(text_line or f"[{label}: no text]")

            # Draw padded box
            cv2.rectangle(annotated, (x1p, y1p), (x2p, y2p), (0,255,0), 2)
            cv2.putText(annotated, label, (x1p, y1p-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Overlay extracted texts
        y0 = 20
        for t in texts:
            cv2.putText(annotated, t, (5, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,), 1, cv2.LINE_AA)
            y0 += 20

        processed.append(annotated)

    if not processed:
        raise RuntimeError("No images processed.")

    # Determine common image size
    heights = [img.shape[0] for img in processed]
    widths = [img.shape[1] for img in processed]
    min_h, min_w = min(heights), min(widths)

    # Resize all to common size
    grid_imgs = [cv2.resize(img, (min_w, min_h)) for img in processed]

    # Compute grid layout
    n = len(grid_imgs)
    rows = math.ceil(n / cols)
    total = rows * cols

    # Blank padding images if needed
    import numpy as np
    blank = 255 * np.ones((min_h, min_w, 3), dtype=np.uint8)
    grid_imgs += [blank.copy() for _ in range(total - n)]

    # Build rows via horizontal concat
    row_images = []
    for r in range(rows):
        row = grid_imgs[r*cols:(r+1)*cols]
        row_images.append(cv2.hconcat(row))

    # Stack rows vertically
    grid = cv2.vconcat(row_images)

    # Display grid
    cv2.imshow("Image Grid OCR", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = r"D:\Programmes\Freelance\Poker-Insight\DATASET\DATASET"
    model_path = r"D:\Programmes\Freelance\Poker-Insight\Full.pt"
    process_and_grid_images(folder_path, model_path, pad_ratio=0.1, max_images=20, cols=5)

# Requirements:
# pip install ultralytics paddleocr opencv-python
