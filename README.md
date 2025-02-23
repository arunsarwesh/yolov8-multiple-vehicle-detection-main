
# Integrated Traffic Signal Simulation with Emergency & Safety Detection

## Overview

This project provides a comprehensive, integrated solution for simulating adaptive traffic signal control at a four-way intersection using computer vision and deep learning. It merges multiple detection tasks into one system:

- **Traffic Vehicle Detection:**  
  Detect and count vehicles (cars, buses, trucks) from four independent video feeds corresponding to the North, East, South, and West lanes.

- **Emergency Vehicle Detection & Priority:**  
  Detect emergency vehicles (e.g., ambulance, fire truck, police car) using a custom-trained YOLO model. When an emergency vehicle is detected, the system immediately prioritizes that lane by overriding the normal signal cycle.

- **Helmet and Overloading Detection:**  
  Use a dedicated model to check for helmet usage (or overloading) on bikes to promote rider safety.

- **Spot Detection:**  
  Highlight road hazards (spots) using segmentation outputs from the emergency detection model.

- **Real-Time Visualization & Fast-Forward:**  
  All four directional video feeds are combined into one window. Each feed displays its current signal state, a timer (only for the active green lane), and per‑second vehicle counts. A fast‑forward mechanism accelerates playback.

## Features

- **Adaptive Traffic Signal Control:**  
  Adjusts signal phases (Green, Yellow, Red) dynamically based on per‑second vehicle counts and ensures that no lane waits more than 2 minutes.

- **Emergency Vehicle Override:**  
  Emergency detection immediately forces the affected lane to turn green. In cases where multiple lanes have emergency vehicles, the lane with the highest emergency count is prioritized.

- **Safety Monitoring:**  
  Helmet and overloading detection are integrated to monitor rider safety. Road hazard (spot) detection is also included to improve safety insights.

- **Multi-Input Integration:**  
  Supports four independent video inputs representing different intersection directions, each with configurable time offsets.

- **Real-Time Overlays:**  
  Displays signal state, timer, and vehicle counts on each quadrant, along with colored borders to indicate the current state (green for active, yellow for transitioning, blue for red).

- **Fast-Forward Playback:**  
  Accelerates simulation by skipping frames, with extra frame skipping for the active green lane.

## Dataset & Model Training

The emergency vehicle detection model (`best.pt`) was trained using a custom dataset defined by a `data.yaml` file. The dataset consists of:

- **train.csv:** Contains image names and labels (emergency or non-emergency) for 70% of the images.
- **test.csv:** Contains image names for the remaining 30% of the images.
- **images Folder:** Contains 2352 images for both training and testing.
- **sample_submission.csv:** Provides the expected submission format.

For helmet and overloading detection, a model (`helmet_met.pt`) was trained on a relevant dataset. Ensure that the training labels in your `data.yaml` match the class names used during inference.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- Pandas
- Numpy
- Ultralytics YOLO (`ultralytics`)
- CVZone
- TensorFlow

You can install the required packages using:

```bash
pip install opencv-python pandas numpy ultralytics cvzone tensorflow
```

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/arunsarwesh/yolov8-multiple-vehicle-detection-main.git
   cd yolov8-multiple-vehicle-detection-main
   ```

2. **Place the following files in the project root:**
   - Video files (e.g., `cars2.mp4`, `p1.mp4`, `cars.mp4`, `tf.mp4`)
   - Model files: `yolov11s.pt`, `best.pt`, `helmet_met.pt`
   - `coco.txt` (class names file)
   - `data.yaml` (if needed for documentation)

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *(If you don’t have a `requirements.txt`, use the pip install command above.)*

## Usage

To run the integrated simulation, execute the main script:

```bash
python "final copy.py"
```

This will launch a window showing the combined four-lane simulation with real-time overlays. Press `Esc` to exit.

## Project Structure

```
.
├── cars2.mp4            # North video input
├── p1.mp4               # East video input
├── cars.mp4             # South video input
├── tf.mp4               # West video input
├── best.pt              # Emergency detection model
├── helmet_met.pt        # Helmet/overloading detection model
├── yolov11s.pt          # Traffic vehicle detection model
├── coco.txt             # Class names file
├── "final copy.py"  # Integrated simulation script
└── README.md            # This file
```

## Troubleshooting & Improvements

- **Emergency Detection:**  
  If emergency vehicles are not being detected properly, verify that:
  - The `best.pt` model is trained correctly on your emergency dataset.
  - The `emergency_classes` list accurately reflects the classes in your dataset.
  - The confidence threshold is set appropriately (try lowering from 0.9 to 0.85 if needed).

- **Performance:**  
  Running multiple YOLO models simultaneously can be resource-intensive. Consider using a GPU or optimizing the inference pipeline (batching, quantization) if necessary.

- **Synchronization:**  
  Ensure that the video inputs are synchronized for realistic simulation. Adjust the `lane_offsets` as needed.

- **Display Clarity:**  
  Modify overlay positions, fonts, and colors in the code to suit your preferences and improve clarity.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Thanks to the Ultralytics YOLO team for the powerful detection framework.
- CVZone and OpenCV for enabling rich computer vision functionalities.
- The dataset providers and the contributors of this project.
