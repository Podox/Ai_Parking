import cv2
import numpy as np
import pandas as pd
import pickle
from ultralytics import YOLO
import mysql.connector

# Connexion à MySQL
db_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # ⇦ à adapter
    database="parking_db"
)
db_cursor = db_conn.cursor()

# Load class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load saved parking spots
with open("parking_spots.pkl", "rb") as f:
    parking_spots = pickle.load(f)  # Each is a dict with 'id' and 'points'

# Open video
cap = cv2.VideoCapture("easy1.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# Initialize tracking dictionary
spot_states = {
    spot['id']: {
        'occupied': False,
        'last_seen': 0,
        'entry_frame': None,
        'total_time': 0,  # in frames
        'entries': 0
    } for spot in parking_spots
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict([frame], verbose=False)
    boxes = results[0].boxes.data
    px = pd.DataFrame(boxes).astype("float")

    car_keypoints = []  # store (front_center, back_center) for each car

    for _, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        class_id = int(row[5])
        class_name = class_list[class_id] if class_id < len(class_list) else "Unknown"

        if class_name == "car":
            center_front = ((x1 + x2) // 2, y2)  # bottom middle
            center_back = ((x1 + x2) // 2, y1)   # top middle
            car_keypoints.append((center_front, center_back))

            # Drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.circle(frame, center_front, 5, (0, 255, 0), -1)
            cv2.circle(frame, center_back, 5, (0, 255, 255), -1)

    free_count = 0
    for spot in parking_spots:
        spot_id = spot['id']
        spot_np = np.array(spot['points'], np.int32)

        currently_occupied = any(
            point_in_polygon(front, spot_np) or point_in_polygon(back, spot_np)
            for front, back in car_keypoints
        )

        # State transitions
        prev_occupied = spot_states[spot_id]['occupied']
        if currently_occupied and not prev_occupied:
            spot_states[spot_id]['entries'] += 1
            spot_states[spot_id]['entry_frame'] = frame_count
        elif not currently_occupied and prev_occupied:
            entry = spot_states[spot_id]['entry_frame']
            if entry is not None:
                spot_states[spot_id]['total_time'] += (frame_count - entry)
                spot_states[spot_id]['entry_frame'] = None

        spot_states[spot_id]['occupied'] = currently_occupied

        # Draw parking spot
        color = (0, 0, 255) if currently_occupied else (0, 255, 0)
        status = "Occupied" if currently_occupied else "Free"
        if not currently_occupied:
            free_count += 1
        cv2.polylines(frame, [spot_np], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, f"ID:{spot_id} {status}", tuple(spot['points'][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Create 2D dashboard view
    dashboard_width = 300
    cell_height = 50
    margin = 10

    # Sort parking spots: occupied first
    sorted_spots = sorted(parking_spots, key=lambda s: not spot_states[s['id']]['occupied'])

    # Adjust height to fit all
    dashboard_height = len(sorted_spots) * (cell_height + margin) + margin
    dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8) + 50

    for i, spot in enumerate(sorted_spots):
        y = i * (cell_height + margin)
        occupied = spot_states[spot['id']]['occupied']
        status_color = (0, 0, 255) if occupied else (0, 255, 0)
        status_text = "Occupied" if occupied else "Free"

        cv2.rectangle(dashboard, (10, y + 10), (dashboard_width - 10, y + cell_height), status_color, -1)
        cv2.putText(dashboard, f"ID:{spot['id']} {status_text}", (15, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw main frame dashboard header
    total_spots = len(parking_spots)
    cv2.rectangle(frame, (0, 0), (300, 60), (50, 50, 50), -1)
    cv2.putText(frame, f"Free: {free_count}/{total_spots}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Match dashboard height to frame height if necessary
    frame_h = frame.shape[0]
    if dashboard.shape[0] > frame_h:
        pad_h = dashboard.shape[0] - frame_h
        frame = cv2.copyMakeBorder(frame, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))
    elif dashboard.shape[0] < frame_h:
        pad_h = frame_h - dashboard.shape[0]
        dashboard = cv2.copyMakeBorder(dashboard, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))

    # Combine original frame with 2D dashboard
    combined = cv2.hconcat([frame, dashboard])
    cv2.imshow('FRAME + MAP', combined)

    # Save stats to CSV and MySQL for current frame
    rows = []
    for spot_id, state in spot_states.items():
        duration_seconds = state['total_time'] / fps
        if state['occupied'] and state['entry_frame'] is not None:
            duration_seconds += (frame_count - state['entry_frame']) / fps
        rows.append({
            "spot_id": spot_id,
            "times_occupied": state['entries'],
            "total_time_occupied_seconds": round(duration_seconds, 2),
            "is_occupied": 1 if state['occupied'] else 0
        })

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv("parking_usage_stats.csv", index=False)

    # Save to MySQL
    db_cursor.execute("DELETE FROM parking_usage")
    for row in rows:
        db_cursor.execute("""
                          INSERT INTO parking_usage (spot_id, times_occupied, total_time_occupied_seconds, is_occupied)
                          VALUES (%s, %s, %s, %s)
                          """, (row['spot_id'], row['times_occupied'], row['total_time_occupied_seconds'],
                                row['is_occupied']))
    db_conn.commit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalize any still-occupied spots
for spot_id, state in spot_states.items():
    if state['occupied'] and state['entry_frame'] is not None:
        spot_states[spot_id]['total_time'] += (frame_count - state['entry_frame'])

cap.release()
cv2.destroyAllWindows()

# Close database connection
db_cursor.close()
db_conn.close()

print("✔ Real-time stats saved to CSV and MySQL.")