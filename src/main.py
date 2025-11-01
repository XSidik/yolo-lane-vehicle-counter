from ultralytics import YOLO
import cv2
import torch
from collections import defaultdict, deque
import numpy as np
import os
from datetime import datetime
import time

allowed_classes = {
         2: "car",
         4: "motorcycle",
         8: "truck",
     }

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path):
    model = YOLO(weights_path)
    return model

def setup_counting_lines(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count_outgoing_line = int(height * 0.40)
    count_incoming_line = int(height * 0.55)
    return {
        'outgoing': {'start':(0, count_outgoing_line), 'end':(width, count_outgoing_line), 'y': count_outgoing_line},
        'incoming': {'start':(0, count_incoming_line), 'end':(width, count_incoming_line), 'y': count_incoming_line}
    }

def crossed_line(prev_cy, curr_cy, count_line_y, direction):    
    if prev_cy is None or curr_cy is None:
        return False

    if direction == "outgoing":
        return prev_cy > count_line_y and curr_cy < count_line_y
    elif direction == "incoming":
        return prev_cy < count_line_y and curr_cy > count_line_y
    return False

def draw_separation_lines_count(frame, counts):    
    height, width = frame.shape[:2]
    split_x = int(width * 0.50)

    cv2.line(frame, (split_x, 0), (split_x, height), (0, 0, 255), 2)

     # Left lane info box (Outgoing)
    left_overlay = frame.copy()
    cv2.rectangle(left_overlay, (10, 10), (300, 110), (0, 0, 0), -1)  # black filled box
    frame = cv2.addWeighted(left_overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, "LEFT LANE (Outgoing)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Cars: {counts['left']['car']}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Motorcycles: {counts['left']['motorcycle']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Trucks: {counts['left']['truck']}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Right lane info box (Incoming)    
    right_overlay = frame.copy()
    cv2.rectangle(right_overlay, (width - 310, 10), (width - 10, 110), (0, 0, 0), -1) # black filled box
    frame = cv2.addWeighted(right_overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, "RIGHT LANE (Incoming)", (width - 300, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Cars: {counts['right']['car']}", (width - 300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Motorcycles: {counts['right']['motorcycle']}", (width - 300, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Trucks: {counts['right']['truck']}", (width - 300, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return frame           

def process(frame, model, count_lines, counts, track_history):    
    results = model.track(frame, persist=True, conf=0.5, verbose=False)
    if not results:
        return frame

    counted_ids = set()

    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue

        for box in result.boxes:
            data = extract_box_data(box)
            if data is None:
                continue

            cls_id, conf, track_id, (x1, y1, x2, y2) = data
            if cls_id not in allowed_classes:
                continue

            class_name = allowed_classes[cls_id]
            color = get_color(cls_id)

            cx, cy = get_center(x1, y1, x2, y2)
            counted_ids.add(track_id)
            track_history[track_id].append((cx, cy))

            draw_box_and_label(frame, (x1, y1, x2, y2), class_name, conf, color)
            draw_trajectory(frame, track_history[track_id], color)
            update_counts(track_history[track_id], count_lines, counts, class_name)

    cleanup_history(track_history, counted_ids)
    draw_count_lines(frame, count_lines)
    return draw_separation_lines_count(frame, counts)

def extract_box_data(box):
    try:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        return cls_id, conf, track_id, (x1, y1, x2, y2)
    except Exception:
        return None

def get_color(cls_id):
    colors = {
        2: (0, 0, 255),   # car
        4: (0, 255, 0),   # motorcycle
        8: (255, 0, 255)  # truck
    }
    return colors.get(cls_id, (0, 255, 0))

def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def draw_box_and_label(frame, bbox, class_name, conf, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_trajectory(frame, points, color):
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 2)

def update_counts(history, count_lines, counts, class_name):
    if len(history) < 2:
        return
    prev_cy, curr_cy = history[-2][1], history[-1][1]

    if crossed_line(prev_cy, curr_cy, count_lines['outgoing']['y'], 'outgoing'):
        counts['left'][class_name] += 1
    elif crossed_line(prev_cy, curr_cy, count_lines['incoming']['y'], 'incoming'):
        counts['right'][class_name] += 1

def cleanup_history(track_history, active_ids):
    for obj_id in list(track_history.keys()):
        if obj_id not in active_ids:
            del track_history[obj_id]

def draw_count_lines(frame, count_lines):
    cv2.line(frame, count_lines['outgoing']['start'], count_lines['outgoing']['end'], (23, 23, 29), 3)
    cv2.line(frame, count_lines['incoming']['start'], count_lines['incoming']['end'], (232, 232, 239), 3)

def output(fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(time.strftime("%Y%m%d_%H%M%S") + "_result.mp4", fourcc, fps, (width, height))
    return out

def main():
    device = select_device()
    print("Using device:", device)

    model = load_model("best.pt")
    if model is None:
        return
    
    model.to(device)

    cap = cv2.VideoCapture("../input.mp4")    
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error reading video frame")
        return
    
    count_lines = setup_counting_lines(cap)
    counts = {
        "left": {"car": 0, "motorcycle": 0, "truck": 0},
        "right": {"car": 0, "motorcycle": 0, "truck": 0},
    }

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = output(fps, width, height)

    track_history = defaultdict(lambda: deque(maxlen=30))
    
    frame_count = 0   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 4 == 0: 
            processed_frame = process(frame, model, count_lines, counts, track_history)
            out.write(processed_frame)       
            cv2.imshow("Vehicle Counting", cv2.resize(processed_frame, (960, 540)))            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
