# YOLOv8 Object Detection, Tracking, and Heatmap

This project provides a flexible Python-based framework for object detection, tracking, and heatmap generation using the [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) model. It supports three main modes:

* **Tracking**: Track and count unique objects across frames, draw trajectories, and display counts.
* **Predict**: Perform detection and draw bounding boxes (with optional hidden labels).
* **Heatmap**: Generate and overlay detection heatmaps to visualize object density over time.

A simple GUI built with PyQt5 allows real-time parameter tuning and visualization.

---

## Features

* **Three processing modes**:

  * `tracking`: unique object tracking and counting with trajectory lines.
  * `predict`: basic detection with customizable bounding boxes and label display.
  * `heatmap`: accumulation of detections into a heatmap overlay.
* **Configurable parameters**:

  * Confidence threshold, image size, frame skipping, start frame.
  * Trajectory tail length & line thickness for tracking.
  * Heatmap alpha, radius, blur options.
* **CLI and GUI**:

  * Command-line interface for batch processing.
  * PyQt5 GUI for interactive control, live display, and progress logging.
* **Results**:

  * Save annotated frames to disk and compile into a video.
  * Optional on-screen display during processing.

---

## Repository Structure

```
├── handler/
│   ├── base.py            # Abstract base class with core processing loop
│   ├── config.py          # CLI argument parser
│   ├── constants.py       # Predefined image sizes
│   ├── predict.py         # Predictor mode implementation
│   ├── track.py           # Tracker mode implementation
│   ├── heatmap.py         # Heatmap mode implementation
│   └── custom_tracker.yaml# ByteTrack tracker settings
├── data/                  # Example videos and output folders
│   ├── videos/
│   └── results/           # Frames for video saving
├── gui.py                 # PyQt5 graphical interface
├── main.py                # Entry point for CLI processing
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)
```

---

## Installation

1. **Clone the repository**:

   ```bash
   ...
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # on Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download or train a YOLOv11 model** and place it under `models/` (e.g., `models/best.pt`).

---

## Command-Line Usage

Run `main.py` with the desired mode and options:

```bash
python main.py <mode> [options]
```

### Modes

* `tracking`
* `predict`
* `heatmap`

### Common Options

| Option          | Description                                  | Default               |
| --------------- | -------------------------------------------- | --------------------- |
| `-m, --model`   | Path to YOLOv11 model (`.pt`)                 | `models/best.pt`      |
| `-v, --video`   | Input video file                             | `data/videos/SAR.mp4` |
| `-s, --save`    | Output video path                            | `data/result.mp4`     |
| `-S, --show`    | Display frames live instead of saving images | off                   |
| `-c, --conf`    | Confidence threshold (0.0–1.0)               | `0.01`                |
| `--imgsz`       | Image size key (`sd`, `s1K`, `s2K`, `s4K`)   | `s1K`                 |
| `--skip_frames` | Skip frames to speed up processing           | `0`                   |
| `--start_frame` | Start from this frame index                  | `0`                   |

### Tracking Options (mode=`tracking`)

| Option            | Description                         | Default |
| ----------------- | ----------------------------------- | ------- |
| `--draw_lines`    | Draw trajectories on screen         | off     |
| `--lines_history` | Number of frames to keep in history | `50`    |

### Heatmap Options (mode=`heatmap`)

| Option     | Description                        | Default |
| ---------- | ---------------------------------- | ------- |
| `--alpha`  | Heatmap overlay strength (0.0–1.0) | `0.4`   |
| `--radius` | Radius of each detection circle    | `15`    |
| `--blur`   | Apply Gaussian blur to heatmap     | on      |

---

## GUI Usage

The `gui.py` script launches a PyQt5-based interface:

```bash
python gui.py
```

Through the GUI, you can:

* Select mode (Tracking, Predict, Heatmap).
* Browse for model, input video, and output file paths.
* Adjust sliders and options in real-time.
* Start, pause, resume, and stop processing.
* View live video, progress bar, stats (current count, total, coverage, FPS), and logs.

---

## Customization

* **Bounding box color & thickness**: Modify `HandlerBase.custom_box()` in `handler/base.py`.
* **Tracker settings**: Tweak thresholds in `models/custom_tracker.yaml`.
* **HUD display**: Adjust or disable via `counter_box`, `info_box`, and `_draw_hud` methods.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
