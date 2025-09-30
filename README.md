# Red-Light Violation Detection (YOLOv12 + ByteTrack + HSV)

This project detects **red-light violations** in traffic videos using:

* **YOLOv12** for vehicle detection
* **ByteTrack** for multi-object tracking
* **HSV** color analysis for traffic-light state (per ROI)

A violation is counted when **the center of a tracked vehicle box crosses the drawn line** **while ANY traffic-light ROI is red**. Violating vehicles stay highlighted in **red** with the label **“VIOLATION”** until the end of the clip. The output video is saved and then auto-opened.

---

## 1. Features

* Single **violation line** (red) — draw it once per video.
* **Multiple traffic-light ROIs** — add as many as you need; each is evaluated independently.
* Robust HSV color detection (downscaling, CLAHE on V channel, morphological cleanup, brightness-weighted voting).
* ByteTrack-based ID tracking to ensure center-line crossing is measured accurately.
* Final video automatically opens after processing.

---

## 2. Controls (Setup Screen)

* **L** → start drawing the **line**: click **two** points (start & end).
* **R** → add a **traffic-light ROI**: click **two** points (top-left & bottom-right). Repeat **R** to add more ROIs.
* **U** → undo the **last ROI** (or the line if no ROI exists).
* **C** → clear **all ROIs**.
* **SPACE** → start processing when at least one line and one ROI are defined.
* **Q / Esc** → quit.

---

## 3. Project Structure

```
project/
├─ main.py                  #  script
├─ requirements.txt         # dependencies
├─ bytetrack.yaml           # tracker config
├─ yolo12l.pt               # YOLOv12 
```

> In the code, set `VIDEO_IN` to your video path (e.g., `input/my_video.mp4`).
> Output will be saved to `output_violations.mp4`.

---



## Clone the Repository
```
git clone https://github.com/yanal154/Red_Light_Violation_Detection
cd Red_Light_Violation_Detection
```
## 4. Download YOLOv12 Model

Before running the project, you need to download the YOLOv12 weights file (`yolo12l.pt`) from the following link:

[Download YOLOv12l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt)

Place `yolo12l.pt` in the project root (next to `main.py`).

---

## 5. Create & Activate a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> To deactivate later: `deactivate`

---

## 6. Install Dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## 7. Run

Update `VIDEO_IN` in the script to point to your input video (e.g., `"input/my_clip.mp4"`), then:

```bash
python main.py
```

1. A setup window appears — draw the **line** (press **L**, click two points), then add one or more **ROIs** (press **R**, two points each).
2. Press **SPACE** to start processing.
3. When processing finishes, the output video (`output_violations.mp4`) is saved and **auto-opened**.

---

## 8. How It Works (Brief)

* **Traffic-light color** is inferred per ROI in HSV space with:

  * Downscaling large crops for speed/stability
  * CLAHE on the V channel (contrast-limited histogram equalization)
  * Morphological open/close to remove noise
  * Brightness-weighted pixel counts for red/yellow/green masks
* **Global light state** = RED if **any** ROI is red; else yellow/green/unknown.
* **Violation rule**: if the **segment from previous center to current center** intersects the **violation line** **while** any ROI is red, the track ID is marked as violating (box turns **red** and stays red).

---




