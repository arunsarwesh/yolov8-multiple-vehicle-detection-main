import cv2
import pandas as pd
import time
from datetime import datetime, timedelta
from ultralytics import YOLO
import cvzone

# ----- Configuration -----
video_file = 'tf.mp4'
lane_offsets = {
    'North': 0,
    'East': 10000,   # 10 seconds offset
    'South': 20000,  # 20 seconds offset
    'West': 30000    # 30 seconds offset
}
lanes = ['North', 'East', 'South', 'West']

# Traffic Signal Timing Parameters
default_green_duration = 10
extended_green_duration = 20
yellow_duration = 3
all_red_duration = 1
MAX_WAIT = timedelta(seconds=120)

# Vehicle counting parameters
offset = 8
lane_count_line_y = 50

# ----- Global Variables -----
model = YOLO('yolov11s.pt')  # Make sure to use correct model

with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Video capture objects
cap_dict = {}
for lane in lanes:
    cap_obj = cv2.VideoCapture(video_file)
    cap_obj.set(cv2.CAP_PROP_POS_MSEC, lane_offsets[lane])
    cap_dict[lane] = cap_obj

# Vehicle counting variables
vehicle_counts = {lane: 0 for lane in lanes}          # Counts per second
frame_counts = {lane: 0 for lane in lanes}            # Temporary frame accumulation
last_update_time = time.time()
signal_state = {lane: "Red" for lane in lanes}
current_green_lane = None
phase_start_time = time.time()
current_phase_duration = default_green_duration
last_green_time = {lane: datetime.min for lane in lanes}

# ----- Modified Functions -----
def process_quadrant(crop, lane):
    """Process frame and return vehicle count"""
    results = model.predict(crop)
    detections = results[0].boxes.data
    df = pd.DataFrame(detections).astype("float")
    
    count = 0
    for _, row in df.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        try:
            c = class_list[d]
        except IndexError:
            continue
        if any(v in c for v in ['car', 'bus', 'truck']):
            count += 1
            cv2.rectangle(crop, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    cv2.line(crop, (1, lane_count_line_y), (crop.shape[1]-1, lane_count_line_y), (0, 255, 0), 2)
    return crop, count

def update_counts_per_second():
    """Update vehicle counts every second"""
    global last_update_time, frame_counts, vehicle_counts
    current_time = time.time()
    if current_time - last_update_time >= 1.0:
        for lane in lanes:
            vehicle_counts[lane] = frame_counts[lane]  # Store the counted vehicles
            frame_counts[lane] = 0  # Reset frame count after updating
        last_update_time = current_time


# ----- Main Loop Modifications -----
def decide_signal():
    """Decide which lane gets the green light based on vehicle counts."""
    max_lane = max(vehicle_counts, key=vehicle_counts.get)
    return max_lane, vehicle_counts[max_lane]

def update_signal_state(chosen_lane, elapsed, count_threshold=15):
    """Update signal states for lanes based on elapsed time."""
    global current_green_lane, phase_start_time, signal_state, current_phase_duration, default_green_duration, yellow_duration
    if current_green_lane is None:
        current_green_lane = chosen_lane
        phase_start_time = time.time()
        signal_state[chosen_lane] = "Green"
        for lane in signal_state:
            if lane != chosen_lane:
                signal_state[lane] = "Red"
    else:
        if elapsed < current_phase_duration:
            signal_state[current_green_lane] = "Green"
        else:
            signal_state[current_green_lane] = "Yellow"
            if elapsed >= current_phase_duration + yellow_duration:
                signal_state[current_green_lane] = "Red"
                current_green_lane = chosen_lane
                phase_start_time = time.time()
                current_phase_duration = default_green_duration  # reset or adjust phase duration as needed
                signal_state[chosen_lane] = "Green"
                for lane in signal_state:
                    if lane != chosen_lane:
                        signal_state[lane] = "Red"

print("Starting four-lane traffic simulation with per-second vehicle counts...")

while True:
    # Process all lanes
    crops = {}
    for lane in lanes:
        ret, frame = cap_dict[lane].read()
        if not ret:
            cap_dict[lane].set(cv2.CAP_PROP_POS_MSEC, lane_offsets[lane])
            continue
            
        # Process frame and get count
        crop = cv2.resize(frame, (510, 250))
        processed_crop, count = process_quadrant(crop, lane)
        crops[lane] = processed_crop
        frame_counts[lane] += count
    
    # Update per-second counts
    update_counts_per_second()
    
    # Signal control logic (using per-second counts)
    chosen_lane, chosen_count = decide_signal()
    elapsed = time.time() - phase_start_time
    update_signal_state(chosen_lane, elapsed, count_threshold=15)
    
    # Display per-second counts
    for lane in lanes:
        display_text = [
            f"{lane} Signal: {signal_state[lane]}",
            f"Timer: {max(0, current_phase_duration - elapsed):.1f}s" if lane == current_green_lane else "Timer: Waiting",
            f"Count/sec: {vehicle_counts[lane]}"
        ]
        for i, text in enumerate(display_text):
            cvzone.putTextRect(crops[lane], text, (10, 30 + 30*i), 
                             scale=1, thickness=2, colorR=(0,0,0))
    
    # Combine and display frames
    top_row = cv2.hconcat([crops['North'], crops['East']])
    bottom_row = cv2.hconcat([crops['South'], crops['West']])
    combined_frame = cv2.vconcat([top_row, bottom_row])
    
    # Show global information
    cvzone.putTextRect(combined_frame, f"Green: {current_green_lane}", (10, 125), scale=2, thickness=2)
    cvzone.putTextRect(combined_frame, 
                     f"Busiest: {max(vehicle_counts, key=vehicle_counts.get)} ({max(vehicle_counts.values())})", 
                     (10, 380), scale=1.5, thickness=2)
    
    cv2.imshow("Traffic Simulation", combined_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
for lane in lanes:
    cap_dict[lane].release()
cv2.destroyAllWindows()