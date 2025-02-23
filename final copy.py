import cv2
import time
import random
from datetime import datetime, timedelta
from ultralytics import YOLO
import cvzone
import numpy as np

###############################################
# Configuration & Model Loading
###############################################

video_inputs = {
    "North": "cars2.mp4",  # Replace with your North video input
    "East": "pt.mp4",  # Replace with your East video input
    "South": "cars.mp4",  # Replace with your South video input
    "West": "tf.mp4",  # Replace with your West video input
}

lane_offsets = {
    "North": 0,
    "East": 2000,  # No offset for East in this case
    "South": 0,  # 20 sec offset
    "West": 0,  # 30 sec offset
}
lanes = ["North", "East", "South", "West"]

# Signal timing parameters (in seconds)
default_green_duration = 10  # Base green time
extended_green_duration = 10  # Extended green time if high count
yellow_duration = random.uniform(2, 3)  # Yellow phase duration (2-3 sec)
all_red_duration = 1  # Clearance period between signals
MAX_WAIT = timedelta(seconds=110)  # Maximum waiting period for any lane

# Additional initial wait time (in seconds) before simulation begins
initial_wait_time = 1  # All lanes paused during this period

# Load models:
traffic_model = YOLO("yolov11s.pt")  # Vehicle detection model
emergency_model = YOLO(
    "best.pt"
)  # Emergency detection model (trained with data.yaml: emergency and non-emergency)
helmet_model = YOLO("helmet_met.pt")  # Helmet detection model

# For traffic model, assume class_list is loaded from coco.txt
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# For emergency detection, use the following class names (order matters)
emergency_class_names = ["emergency", "non-emergency"]

# Define helmet class labels to avoid NameError
helmet_class_labels = ["helmet", "no_helmet"]

###############################################
# Video Capture Setup (four independent inputs)
###############################################

cap_dict = {}
for lane in lanes:
    cap_obj = cv2.VideoCapture(video_inputs[lane])
    cap_obj.set(cv2.CAP_PROP_POS_MSEC, lane_offsets[lane])
    cap_dict[lane] = cap_obj

###############################################
# Global Variables for Traffic Simulation
###############################################

vehicle_counts = {lane: 0 for lane in lanes}  # Number of vehicles per lane
signal_state = {lane: "Red" for lane in lanes}  # Each lane: Green, Yellow, or Red
last_green_time = {lane: datetime.min for lane in lanes}  # Last time lane had green

current_green_lane = None
phase_start_time = time.time()  # When current phase started
current_phase_duration = default_green_duration

last_emergency_check = time.time()  # For emergency detection timing

# To support pausing, store previous frames for each lane
prev_crop = {lane: None for lane in lanes}

###############################################
# Helper Functions
###############################################


def split_frame(frame):
    h, w, _ = frame.shape
    half_w = w // 2
    half_h = h // 2
    return {
        "North": frame[0:half_h, 0:half_w],
        "East": frame[0:half_h, half_w:w],
        "South": frame[half_h:h, 0:half_w],
        "West": frame[half_h:h, half_w:w],
    }


def read_lane_frame(lane):
    cap_obj = cap_dict[lane]
    ret, frame = cap_obj.read()
    if not ret or frame is None or frame.size == 0:
        cap_obj.set(cv2.CAP_PROP_POS_MSEC, lane_offsets[lane])
        ret, frame = cap_obj.read()
        if not ret or frame is None or frame.size == 0:
            return np.zeros((250, 510, 3), dtype=np.uint8)
    return cv2.resize(frame, (510, 250))


def process_quadrant_traffic(crop):
    results = traffic_model.predict(crop)
    detections = results[0].boxes.data
    if hasattr(detections, "cpu"):
        detections = detections.cpu().numpy()
    count = 0
    bike_detected = False
    for detection in detections:
        x1, y1, x2, y2, conf, cls_idx = detection
        cls_idx = int(cls_idx)
        try:
            c = class_list[cls_idx]
        except IndexError:
            continue
        lower_c = c.lower()
        if any(keyword in lower_c for keyword in ["car", "bus", "truck"]):
            count += 1
        if any(keyword in lower_c for keyword in ["bike", "bicycle", "motorcycle"]):
            bike_detected = True
    return crop, count, bike_detected


def process_quadrant_emergency(crop):
    """
    Run the emergency detection model and draw bounding boxes over detected emergency vehicles.
    This function uses the emergency_class_names list.
    """
    results = emergency_model.predict(crop)
    emergency_detected = False
    for r in results:
        for box in r.boxes:
            if hasattr(box.xyxy[0], "cpu"):
                coords = box.xyxy[0].cpu().numpy()
            else:
                coords = box.xyxy[0]
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = (
                emergency_class_names[cls_idx]
                if cls_idx < len(emergency_class_names)
                else "Unknown"
            )
            if label == "emergency" and conf >= 0.85:
                emergency_detected = True
                cvzone.putTextRect(
                    crop,
                    f"Emergency ({conf*100:.1f}%)",
                    (x1, y1 - 10),
                    scale=0.8,
                    thickness=2,
                    colorR=(0, 0, 255),
                )
                cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return crop, emergency_detected


def process_helmet_detection(crop):
    results = helmet_model.predict(crop)
    best_det = None
    best_conf = 0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_det = box
    if best_det is not None:
        d = int(best_det.cls[0])
        try:
            predicted_class = helmet_class_labels[d]
        except IndexError:
            predicted_class = "Unknown"
        if hasattr(best_det.xyxy[0], "cpu"):
            coords = best_det.xyxy[0].cpu().numpy()
        else:
            coords = best_det.xyxy[0]
        x1, y1, x2, y2 = map(int, coords)
        cvzone.putTextRect(
            crop,
            f"{predicted_class} ({best_conf*100:.1f}%)",
            (x1, y2 - 10),
            scale=0.8,
            thickness=2,
            colorR=(0, 255, 255),
        )
        cv2.rectangle(crop, (x1, y1), (x2, y2), (255, 0, 0), 2)
    else:
        cvzone.putTextRect(
            crop,
            "No Helmet Detected",
            (10, crop.shape[0] - 10),
            scale=0.8,
            thickness=2,
            colorR=(0, 255, 255),
        )
    return crop


def process_ambulance_detection(crop):
    """
    Specifically detect ambulance in the given crop.
    Draw a blue bounding box if an ambulance is detected.
    """
    results = emergency_model.predict(crop)
    ambulance_detected = False
    for r in results:
        for box in r.boxes:
            if hasattr(box.xyxy[0], "cpu"):
                coords = box.xyxy[0].cpu().numpy()
            else:
                coords = box.xyxy[0]
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = (
                emergency_class_names[cls_idx]
                if cls_idx < len(emergency_class_names)
                else "Unknown"
            )
            if label == "emergency" and conf >= 0.85:
                ambulance_detected = True
                cvzone.putTextRect(
                    crop,
                    f"Ambulance ({conf*100:.1f}%)",
                    (x1, y1 - 10),
                    scale=0.8,
                    thickness=2,
                    colorR=(255, 0, 0),
                )
                cv2.rectangle(crop, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return crop, ambulance_detected


###############################################
# Signal Management Functions
###############################################


def decide_signal():
    """
    Decide the next lane to turn green.
    Priority is given if a lane has waited too long.
    Otherwise, choose the lane with the highest vehicle count.
    """
    now = datetime.now()
    forced = [lane for lane in lanes if (now - last_green_time[lane]) >= MAX_WAIT]
    if forced:
        chosen = random.choice(forced)
        return chosen, vehicle_counts[chosen]
    chosen = max(lanes, key=lambda lane: vehicle_counts[lane])
    return chosen, vehicle_counts[chosen]


def update_signal_state(chosen_lane, elapsed, count_threshold=15):
    """
    Manages the state transitions for the current green lane.
    Transition order: Green -> Yellow -> Red.
    When transitioning to a new green lane, set its phase timer based on vehicle count.
    """
    global current_green_lane, phase_start_time, current_phase_duration, signal_state
    if current_green_lane is None:
        current_green_lane = chosen_lane
        signal_state[current_green_lane] = "Green"
        phase_start_time = time.time()
        current_phase_duration = (
            extended_green_duration
            if vehicle_counts[current_green_lane] >= count_threshold
            else default_green_duration
        )
        vehicle_counts[current_green_lane] = 0
    else:
        if signal_state[current_green_lane] == "Green":
            if elapsed >= current_phase_duration:
                signal_state[current_green_lane] = "Yellow"
                phase_start_time = time.time()
        elif signal_state[current_green_lane] == "Yellow":
            if elapsed >= yellow_duration:
                signal_state[current_green_lane] = "Red"
                last_green_time[current_green_lane] = datetime.now()
                time.sleep(all_red_duration)
                chosen, _ = decide_signal()
                current_green_lane = chosen
                signal_state[current_green_lane] = "Green"
                phase_start_time = time.time()
                current_phase_duration = (
                    extended_green_duration
                    if vehicle_counts[current_green_lane] >= count_threshold
                    else default_green_duration
                )
                vehicle_counts[current_green_lane] = 0
    return


def fast_forward():
    FAST_FORWARD_FRAMES = 5
    for _ in range(FAST_FORWARD_FRAMES):
        for lane in lanes:
            cap_dict[lane].grab()


###############################################
# Initial Setup: Capture First Frame for Each Lane
###############################################

simulation_start = time.time()
for lane in lanes:
    initial_frame = read_lane_frame(lane)
    prev_crop[lane] = initial_frame.copy()

###############################################
# Set up window size (increase window frame size)
###############################################
cv2.namedWindow("Integrated Traffic Signal Simulation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Integrated Traffic Signal Simulation", 1280, 720)

###############################################
# Main Simulation Loop
###############################################

print("Starting integrated traffic signal simulation...")

while True:
    # During the initial waiting period, keep all lanes paused.
    if time.time() - simulation_start < initial_wait_time:
        crops = {lane: prev_crop[lane] for lane in lanes}
    else:
        crops = {}
        emergency_flags = {lane: False for lane in lanes}
        current_time = time.time()
        do_emergency = current_time - last_emergency_check >= 1
        if do_emergency:
            last_emergency_check = current_time

        # Process each lane: read frame, detect vehicles, emergency, helmet, etc.
        for lane in lanes:
            frame = read_lane_frame(lane)
            processed_crop, count, bike_detected = process_quadrant_traffic(
                frame.copy()
            )
            vehicle_counts[lane] = count

            if do_emergency:
                processed_crop, emergency_detected = process_quadrant_emergency(
                    processed_crop
                )
                if emergency_detected:
                    emergency_flags[lane] = True
                    cvzone.putTextRect(
                        processed_crop,
                        "EMERGENCY",
                        (10, 120),
                        scale=1,
                        thickness=2,
                        colorR=(0, 0, 255),
                    )
            if bike_detected:
                processed_crop = process_helmet_detection(processed_crop)
            crops[lane] = processed_crop

        # For lanes with a Red signal, pause them by reusing the previous frame.
        paused_flags = {}
        for lane in lanes:
            if signal_state[lane] == "Red":
                paused_flags[lane] = True
                if prev_crop[lane] is not None:
                    crops[lane] = prev_crop[lane]
            else:
                paused_flags[lane] = False
                prev_crop[lane] = crops[lane]

        # Check for ambulance detection override on all lanes.
        ambulance_override = False
        for lane in lanes:
            ambulance_crop, ambulance_detected = process_ambulance_detection(
                crops[lane]
            )
            if ambulance_detected:
                ambulance_override = True
                current_green_lane = lane
                signal_state[lane] = "Green"
                phase_start_time = time.time()
                last_green_time[lane] = datetime.now()
                vehicle_counts[lane] = 0
                crops[lane] = ambulance_crop
                print(f"Ambulance detected in {lane} lane. Overriding signal!")
                break  # Immediate override if ambulance detected

        # When ambulance override occurs, force all other lanes to Red.
        if ambulance_override:
            for lane in lanes:
                if lane != current_green_lane:
                    signal_state[lane] = "Red"
        else:
            # If no ambulance override, check for emergency override and then normal timing.
            if any(emergency_flags.values()):
                chosen_lane = max(
                    [lane for lane in lanes if emergency_flags[lane]],
                    key=lambda lane: vehicle_counts[lane],
                )
                current_green_lane = chosen_lane
                signal_state[current_green_lane] = "Green"
                phase_start_time = time.time()
                last_green_time[current_green_lane] = datetime.now()
                vehicle_counts[current_green_lane] = 0
            else:
                chosen_lane, _ = decide_signal()
                elapsed = time.time() - phase_start_time
                update_signal_state(chosen_lane, elapsed, count_threshold=15)

    # Overlay signal info and timers on each lane's crop.
    for lane in lanes:
        if lane == current_green_lane:
            if signal_state[lane] == "Green":
                remaining = max(
                    0, current_phase_duration - (time.time() - phase_start_time)
                )
                timer_text = f"Green Timer: {remaining:.1f}s"
            elif signal_state[lane] == "Yellow":
                remaining = max(0, yellow_duration - (time.time() - phase_start_time))
                timer_text = f"Yellow Timer: {remaining:.1f}s"
        else:
            wait_time = (
                (datetime.now() - last_green_time[lane]).total_seconds()
                if last_green_time[lane] != datetime.min
                else 0
            )
            timer_text = f"Waiting: {wait_time:.0f}s"

        cvzone.putTextRect(
            crops[lane],
            f"{lane} Signal: {signal_state[lane]}",
            (10, 30),
            scale=1.5,
            thickness=2,
            colorR=(0, 0, 0),
        )
        cvzone.putTextRect(
            crops[lane], timer_text, (10, 60), scale=1.5, thickness=2, colorR=(0, 0, 0)
        )
        cvzone.putTextRect(
            crops[lane],
            f"Vehicles: {vehicle_counts[lane]}",
            (10, 90),
            scale=1.5,
            thickness=2,
            colorR=(0, 0, 0),
        )
        if signal_state[lane] == "Red":
            border_color = (0, 0, 255)  # Red
        elif signal_state[lane] == "Green":
            border_color = (0, 255, 0)  # Green
        elif signal_state[lane] == "Yellow":
            border_color = (0, 255, 255)  # Yellow
        else:
            border_color = (255, 255, 255)  # White

        cv2.rectangle(
            crops[lane],
            (0, 0),
            (crops[lane].shape[1] - 1, crops[lane].shape[0] - 1),
            border_color,
            thickness=5,
        )

    top_row = cv2.hconcat([crops["North"], crops["East"]])
    bottom_row = cv2.hconcat([crops["South"], crops["West"]])
    combined_frame = cv2.vconcat([top_row, bottom_row])
    cvzone.putTextRect(
        combined_frame,
        f"Current Green: {current_green_lane}",
        (10, 125),
        scale=2,
        thickness=2,
    )
    busiest_lane = max(vehicle_counts, key=vehicle_counts.get)
    cvzone.putTextRect(
        combined_frame,
        f"Busiest: {busiest_lane} ({vehicle_counts[busiest_lane]})",
        (10, 380),
        scale=1.5,
        thickness=2,
    )

    cv2.imshow("Integrated Traffic Signal Simulation", combined_frame)

    if current_green_lane is not None:
        print(
            f"Green: {current_green_lane}, Vehicles: {vehicle_counts[current_green_lane]}"
        )
    else:
        print("No lane is currently green")

    if cv2.waitKey(1) & 0xFF == 27:
        break

    fast_forward()

for lane in lanes:
    cap_dict[lane].release()
cv2.destroyAllWindows()
