import cv2
import numpy as np
import pandas as pd
import pickle
from ultralytics import YOLO

# Load class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load saved parking spots from file
with open("parking_spots.pkl", "rb") as f:
    parking_spots = pickle.load(f)  # Each spot is a list of 4 (x, y) points

# Open video
cap = cv2.VideoCapture("easy1.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

count = 0

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame, verbose=False)

    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    car_centers = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        class_id = int(row[5])
        class_name = class_list[class_id] if class_id < len(class_list) else "Unknown"

        if class_name == "car":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            car_centers.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Check parking spots (polygons)
    free_count = 0
    for idx, spot in enumerate(parking_spots):
        spot_np = np.array(spot, np.int32)
        occupied = False
        for center in car_centers:
            if point_in_polygon(center, spot_np):
                occupied = True
                break

        color = (0, 0, 255) if occupied else (0, 255, 0)
        status = "Occupied" if occupied else "Free"
        if not occupied:
            free_count += 1
        cv2.polylines(frame, [spot_np], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, status, tuple(spot[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Dashboard
    total_spots = len(parking_spots)
    cv2.rectangle(frame, (0, 0), (250, 60), (50, 50, 50), -1)
    cv2.putText(frame, f"Free: {free_count}/{total_spots}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
