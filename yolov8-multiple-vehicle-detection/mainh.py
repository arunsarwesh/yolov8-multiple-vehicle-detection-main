import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from tracker import Tracker  # Ensure the Tracker class is defined correctly in tracker.py

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Initialize video capture
cap = cv2.VideoCapture('tf.mp4')

# Open the 'coco.txt' file containing class names and read its content
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize tracker and counting parameters
tracker = Tracker()
cy1 = 184        # Y-coordinate for the counting line
offset = 8       # Allowable range for detecting crossing
count = 0        # Frame counter
vehicle_count = 0
counted_ids = set()  # To ensure each vehicle is counted only once

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:  # Process every third frame for speed
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Predict objects using YOLO
    results = model.predict(frame)
    detections = results[0].boxes.data
    df = pd.DataFrame(detections).astype("float")

    vehicles = []
    for index, row in df.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        try:
            c = class_list[d]
        except IndexError:
            continue
        if 'car' in c or 'bus' in c or 'truck' in c:
            # Convert from [x1, y1, x2, y2] to [x, y, w, h]
            w = x2 - x1
            h = y2 - y1
            vehicles.append([x1, y1, w, h])

    # Update tracker with current detections
    tracked_objects = tracker.update(vehicles)

    # Draw the counting line
    cv2.line(frame, (1, cy1), (1018, cy1), (0, 255, 0), 2)

    # Check each tracked object for crossing the line and count once per vehicle
    for obj in tracked_objects:
        x, y, w, h, obj_id = obj
        cx = x + w // 2
        cy = y + h // 2

        # # Draw bounding box and display only the unique vehicle number
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # cvzone.putTextRect(frame, f'{obj_id}', (x, y - 10), scale=1, thickness=1)

        # If center is within the line's vertical range and hasn't been counted yet
        if (cy > cy1 - offset) and (cy < cy1 + offset):
            if obj_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(obj_id)

    cvzone.putTextRect(frame, f'Vehicle Count: {vehicle_count}', (50, 50), scale=2, thickness=2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    print(f'Total vehicle count: {vehicle_count}')

cap.release()
cv2.destroyAllWindows()
