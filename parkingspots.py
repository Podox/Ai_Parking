import cv2
import pickle

drawing = False
points = []
parking_spots = []

def draw_spots(img, spots):
    for spot in spots:
        if len(spot) == 4:
            cv2.polylines(img, [np.array(spot, dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=2)

def mouse_callback(event, x, y, flags, param):
    global points, parking_spots

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 4:
            parking_spots.append(points.copy())
            points = []

    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()

import numpy as np

# Load a frame to draw on (you can also use cap.read())
frame = cv2.imread("park1.jpg")
frame = cv2.resize(frame, (1020, 500))
# Replace with a frame from your video if needed
if frame is None:
    print("Image not found.")
    exit()

cv2.namedWindow("Draw Parking Spots")
cv2.setMouseCallback("Draw Parking Spots", mouse_callback)

while True:
    display = frame.copy()

    # Draw current spot in progress
    for point in points:
        cv2.circle(display, point, 5, (0, 0, 255), -1)
    if len(points) > 1:
        cv2.polylines(display, [np.array(points, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

    # Draw saved spots
    draw_spots(display, parking_spots)

    cv2.putText(display, f"Spots: {len(parking_spots)} | Press 's' to save, 'q' to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Draw Parking Spots", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        with open("parking_spots.pkl", "wb") as f:
            pickle.dump(parking_spots, f)
        print(f"Saved {len(parking_spots)} spots to parking_spots.pkl")

cv2.destroyAllWindows()
