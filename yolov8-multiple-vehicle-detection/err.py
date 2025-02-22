import cv2
import pandas as pd
import time
from datetime import datetime, timedelta
from ultralytics import YOLO
import cvzone
from tracker import Tracker  # Ensure your Tracker class is defined correctly

# ----- Configuration -----
video_file = 'tf.mp4'
# Define time offsets for each lane in milliseconds
lane_offsets = {
    'North': 0,
    'East': 10000,   # 10 seconds offset
    'South': 20000,  # 20 seconds offset
    'West': 30000    # 30 seconds offset
}
lanes = ['North', 'East', 'South', 'West']

# Traffic Signal Timing Parameters (in seconds)
default_green_duration = 10    # Base green time
extended_green_duration = 20   # Extended green time if vehicle count is high
yellow_duration = 3            # Yellow phase duration
all_red_duration = 1           # All-red clearance duration
MAX_WAIT = timedelta(seconds=120)  # Maximum 2-minute wait per lane

# Vehicle detection/counting parameters (for each feed's coordinate system)
offset = 8                   # Vertical margin for counting line
lane_count_line_y = 50       # Counting line position in each quadrant (assumed crop height ~250)

# ----- Global Variables -----
# Load the YOLO model
model = YOLO('yolov11s.pt')

# Load class names from coco.txt
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Create separate VideoCapture objects for each lane with time offsets
cap_dict = {}
for lane in lanes:
    cap_obj = cv2.VideoCapture(video_file)
    cap_obj.set(cv2.CAP_PROP_POS_MSEC, lane_offsets[lane])
    cap_dict[lane] = cap_obj

# Create a Tracker and vehicle count storage for each lane
trackers = {lane: Tracker() for lane in lanes}
vehicle_counts = {lane: 0 for lane in lanes}
counted_ids = {lane: set() for lane in lanes}
last_green_time = {lane: datetime.min for lane in lanes}

# Global signal state: only one lane is green at any time
signal_state = {lane: "Red" for lane in lanes}
current_green_lane = None  # Lane that currently has green
phase_start_time = time.time()
current_phase_duration = default_green_duration  # Adaptive green duration

# ----- Functions -----
def split_frame(frame):
    """
    Split the input frame (assumed 1020x500) into 4 quadrants.
    Returns a dictionary mapping lane names to cropped frames.
    Assignment:
      North: top-left, East: top-right, South: bottom-left, West: bottom-right.
    """
    h, w, _ = frame.shape
    half_w = w // 2   # ~510
    half_h = h // 2   # ~250
    return {
        'North': frame[0:half_h, 0:half_w],
        'East':  frame[0:half_h, half_w:w],
        'South': frame[half_h:h, 0:half_w],
        'West':  frame[half_h:h, half_w:w]
    }

def read_lane_frame(lane):
    """
    Read a frame from the VideoCapture for a given lane.
    If the video ends, loop it by resetting the position to the lane's offset.
    """
    cap_obj = cap_dict[lane]
    ret, frame = cap_obj.read()
    if not ret:
        cap_obj.set(cv2.CAP_PROP_POS_MSEC, lane_offsets[lane])
        ret, frame = cap_obj.read()
    # Resize to quadrant size (510x250)
    frame = cv2.resize(frame, (510, 250))
    return frame

def process_quadrant(crop, lane):
    """
    Process a quadrant crop to detect vehicles using YOLO and update its Tracker.
    Draw bounding boxes and count vehicles crossing a fixed counting line.
    Returns the processed crop.
    """
    results = model.predict(crop)
    detections = results[0].boxes.data
    df = pd.DataFrame(detections).astype("float")
    vehicles = []
    for _, row in df.iterrows():
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
            w_box = x2 - x1
            h_box = y2 - y1
            vehicles.append([x1, y1, w_box, h_box])
    tracked_objects = trackers[lane].update(vehicles)
    for obj in tracked_objects:
        x, y, w, h, obj_id = obj
        cx = x + w // 2
        cy = y + h // 2
        cv2.rectangle(crop, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cvzone.putTextRect(crop, f'{obj_id}', (x, y - 10), scale=1, thickness=1, colorR=(0,0,0))
        if (cy > lane_count_line_y - offset) and (cy < lane_count_line_y + offset):
            if obj_id not in counted_ids[lane]:
                vehicle_counts[lane] += 1
                counted_ids[lane].add(obj_id)
    cv2.line(crop, (1, lane_count_line_y), (crop.shape[1]-1, lane_count_line_y), (0, 255, 0), 2)
    return crop

def decide_signal():
    """
    Determine which lane should get the green signal.
    Priority:
      - If any lane hasn't been green for MAX_WAIT, force that lane.
      - Otherwise, choose the lane with the highest vehicle count.
    Returns the chosen lane and its vehicle count.
    """
    now = datetime.now()
    forced = [lane for lane in lanes if now - last_green_time[lane] >= MAX_WAIT]
    if forced:
        return forced[0], vehicle_counts[forced[0]]
    chosen = max(lanes, key=lambda lane: vehicle_counts[lane])
    return chosen, vehicle_counts[chosen]

def update_signal_state(chosen_lane, elapsed, count_threshold):
    """
    Update the global signal state.
    Transitions: Green -> Yellow -> Red -> New Green.
    Reset vehicle count and counted IDs when a lane becomes green.
    """
    global current_green_lane, phase_start_time, current_phase_duration
    if current_green_lane is None:
        current_green_lane = chosen_lane
        signal_state[current_green_lane] = "Green"
        phase_start_time = time.time()
        if vehicle_counts[current_green_lane] >= count_threshold:
            current_phase_duration = extended_green_duration
        else:
            current_phase_duration = default_green_duration
        # Reset vehicle count and IDs for this lane
        vehicle_counts[current_green_lane] = 0
        counted_ids[current_green_lane].clear()
    else:
        if signal_state[current_green_lane] == "Green":
            if elapsed >= current_phase_duration:
                signal_state[current_green_lane] = "Yellow"
                phase_start_time = time.time()
        elif signal_state[current_green_lane] == "Yellow":
            if elapsed >= yellow_duration:
                signal_state[current_green_lane] = "Red"
                phase_start_time = time.time()
        elif signal_state[current_green_lane] == "Red":
            if elapsed >= all_red_duration:
                current_green_lane = chosen_lane
                signal_state[current_green_lane] = "Green"
                phase_start_time = time.time()
                last_green_time[current_green_lane] = datetime.now()
                # Reset vehicle count and IDs for the new green lane
                vehicle_counts[current_green_lane] = 0
                counted_ids[current_green_lane].clear()
    return

# ----- Main Simulation Loop -----
print("Starting four-lane traffic signal simulation with independent time offsets...")

while True:
    # Read and process frames for each lane independently
    crops = {}
    for lane in lanes:
        crop = read_lane_frame(lane)
        processed_crop = process_quadrant(crop, lane)
        crops[lane] = processed_crop

    # Decide which lane should be green based on vehicle counts
    chosen_lane, chosen_count = decide_signal()
    elapsed = time.time() - phase_start_time
    update_signal_state(chosen_lane, elapsed, count_threshold=15)
    
    # For each lane, overlay the signal state, timer, and vehicle count
    for lane in lanes:
        if lane == current_green_lane:
            remaining_time = max(0, current_phase_duration - elapsed)
            timer_text = f"Timer: {remaining_time:.1f}s"
        else:
            timer_text = "Timer: Waiting"
        cvzone.putTextRect(crops[lane], f"{lane} Signal: {signal_state[lane]}", (10, 30), scale=1.5, thickness=2, colorR=(0,0,0))
        cvzone.putTextRect(crops[lane], timer_text, (10, 60), scale=1.5, thickness=2, colorR=(0,0,0))
        cvzone.putTextRect(crops[lane], f"Count: {vehicle_counts[lane]}", (10, 90), scale=1.5, thickness=2, colorR=(0,0,0))
    
    # Combine quadrants into a single window (2x2 grid)
    top_row = cv2.hconcat([crops['North'], crops['East']])
    bottom_row = cv2.hconcat([crops['South'], crops['West']])
    combined_frame = cv2.vconcat([top_row, bottom_row])
    
    # Global overlay: show which lane is currently green
    cvzone.putTextRect(combined_frame, f"Current Green: {current_green_lane}", (10, 125), scale=2, thickness=2, colorR=(0,0,0))
    
    # ----- New Overlay Text for Highest Waiting Count -----
    highest_lane = max(vehicle_counts, key=lambda lane: vehicle_counts[lane])
    highest_count = vehicle_counts[highest_lane]
    cvzone.putTextRect(combined_frame, f"Highest Waiting: {highest_lane} ({highest_count})", (10, 380), scale=2, thickness=2, colorR=(0,0,0))
    
    cv2.imshow("Four-Lane Traffic Signal Simulation", combined_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc'
        break
    
    print(f"Green: {current_green_lane}, Count: {vehicle_counts[current_green_lane]}")
    
    # ----- Fast-Forwarding Logic -----
    # Baseline: Skip one extra frame for each lane (simulate faster playback for all feeds)
    for lane in lanes:
        cap_dict[lane].grab()
    # Additional fast-forward for the lane with green signal: skip one more frame
    if current_green_lane is not None and signal_state[current_green_lane] == "Green":
        for lane in lanes:
            cap_dict[lane].grab()

# Release all VideoCapture objects
for lane in lanes:
    cap_dict[lane].release()
cv2.destroyAllWindows()
