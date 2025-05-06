
import cv2
import numpy as np

# ------------------ Configuration ------------------
video_path = r"D:\Programmes\Freelance\Poker-Insight\Game\input.mp4"  # Replace with your video file
# Increased HSV bounds for a broader color range.
# Hue expanded from [38, 58] to [28, 68],
# Saturation from [94, 174] to [84, 200],
# Value from [133, 213] to [100, 255].
lower_bound = np.array([28, 84, 100])
upper_bound = np.array([68, 200, 255])
# Kernel for morphological operations (if needed).
kernel = np.ones((5, 5), np.uint8)

# ------------------ Global Variables for ROI Selection ------------------
roi_list = []          # List of ROI rectangles: each as (x1, y1, x2, y2)
current_roi_start = None
current_roi_end = None
roi_drawing = False
frame_for_roi = None   # Frame used for ROI selection

# ------------------ Mouse Callback for ROI Selection ------------------
def roi_mouse_callback(event, x, y, flags, param):
    global roi_drawing, current_roi_start, current_roi_end, roi_list, frame_for_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_drawing = True
        current_roi_start = (x, y)
        current_roi_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_drawing:
            current_roi_end = (x, y)
            temp = frame_for_roi.copy()
            for (rx1, ry1, rx2, ry2) in roi_list:
                cv2.rectangle(temp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.rectangle(temp, current_roi_start, current_roi_end, (0, 255, 0), 2)
            cv2.imshow("ROI Selection", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_drawing = False
        current_roi_end = (x, y)
        # Normalize coordinates.
        x1 = min(current_roi_start[0], current_roi_end[0])
        y1 = min(current_roi_start[1], current_roi_end[1])
        x2 = max(current_roi_start[0], current_roi_end[0])
        y2 = max(current_roi_start[1], current_roi_end[1])
        roi_list.append((x1, y1, x2, y2))
        temp = frame_for_roi.copy()
        for (rx1, ry1, rx2, ry2) in roi_list:
            cv2.rectangle(temp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        cv2.imshow("ROI Selection", temp)

# ------------------ Video Processing ------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Convert frame to HSV and create binary mask for full frame.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours on the full-frame mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Calculate centroid if needed.
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Arrow drawing removed as per request.

    # On the 100th frame, allow ROI selection if not done.
    if frame_count == 100 and len(roi_list) == 0:
        frame_for_roi = frame.copy()
        cv2.namedWindow("ROI Selection")
        cv2.setMouseCallback("ROI Selection", roi_mouse_callback)
        print("Draw one or more rectangles on the 100th frame.\nWhen done, press 'q' in the ROI window.")
        while True:
            temp = frame_for_roi.copy()
            for (rx1, ry1, rx2, ry2) in roi_list:
                cv2.rectangle(temp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            if roi_drawing and current_roi_start and current_roi_end:
                cv2.rectangle(temp, current_roi_start, current_roi_end, (0, 255, 0), 2)
            cv2.imshow("ROI Selection", temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("ROI Selection")

    # Process each defined ROI.
    roi_rows = []
    for (rx1, ry1, rx2, ry2) in roi_list:
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        # Convert ROI to HSV and generate its binary mask.
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_mask = cv2.inRange(roi_hsv, lower_bound, upper_bound)
        # Use the binary mask directly for edge detection.
        edges = cv2.Canny(roi_mask, 30, 100, apertureSize=3)
        # Use HoughLinesP with lower thresholds to detect lines.
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
        line_detected = False
        if lines is not None:
            line_detected = True
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                cv2.line(roi, (x1_line, y1_line), (x2_line, y2_line), (255, 0, 0), 2)
        # Create a BGR version of the binary mask.
        roi_mask_bgr = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        # If a line is detected, highlight the ROI rectangle in red on the main frame.
        if line_detected:
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        # Concatenate the ROI (annotated) and its binary mask side by side.
        roi_pair = np.hstack((roi, roi_mask_bgr))
        roi_rows.append(roi_pair)
    
    # Pad ROI rows so that they all have the same width before stacking vertically.
    if len(roi_rows) > 0:
        max_width = max(row.shape[1] for row in roi_rows)
        padded_rows = []
        for row in roi_rows:
            h, w, c = row.shape
            if w < max_width:
                pad_width = max_width - w
                row_padded = np.pad(row, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            else:
                row_padded = row
            padded_rows.append(row_padded)
        roi_display = np.vstack(padded_rows)
    else:
        roi_display = np.zeros_like(frame)

    # Convert the full-frame mask to BGR.
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    full_display = np.hstack((frame, mask_bgr))
    # Resize ROI display to match full_display width if needed.
    full_width = full_display.shape[1]
    if roi_display.shape[1] != full_width:
        roi_display = cv2.resize(roi_display, (full_width, roi_display.shape[0]))
    combined = np.vstack((full_display, roi_display))
    
    cv2.imshow("Video Tracking and ROI Crops", combined)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
