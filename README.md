# VN Motion Reader

**VN Motion Reader** is a Python application that allows users to control visual novel (VN) text progression using hand gesture speed. By leveraging a webcam and MediaPipe's hand tracking, it analyzes the circular motion of your index finger to determine reading speed, dynamically sending key presses to progress the text in a visual novel.

## Features

- **Gesture-Based Control**: Progress through visual novels using a circular motion with your index finger.
- **Dynamic Speed Detection**: Detects slow and fast motions, adjusting the frequency of key presses accordingly.
- **HUD Interface**: A clean overlay on the webcam feed showing current status, speed, active key, and motion trail.
- **Finger Jitter Calibration**: A built-in calibration tool to ignore natural hand shake.
- **Global Hotkeys**: Toggle tracking or safely exit at any time without needing focus on the app window.
- **Configurable Keys**: Easily switch between standard VN progression keys (`Enter`, `Space`, `Down Arrow`, `Right Arrow`).

## Requirements

- Python 3.8 or higher
- A working webcam

### Python Dependencies

The required packages are listed in `requirements.txt`:
- `opencv-python>=4.8.0`
- `mediapipe>=0.10.0`
- `pynput>=1.7.6`
- `pyautogui>=0.9.54`

## Installation

1. Clone or download this repository.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The application will automatically download the required MediaPipe model on its first run.)*

## Usage

Start the application by running the `main.py` script:

```bash
python main.py
```

### Controls

Once running, you can use the following controls via the OpenCV window or global hotkeys (where supported):

- **`F8` or `f`**: Toggle Gesture Control ON / OFF.
- **`c`**: Calibrate finger jitter. Hold your finger completely still for 2 seconds when prompted.
- **`k`**: Change the visual novel next-text key (cycles through Enter, Space, Down Arrow, Right Arrow).
- **`ESC` or `q`**: Emergency Stop / Quit the application.

### How it Works

1. **Camera Tracking (`camera.py`)**: Captures webcam input and uses MediaPipe to identify hand landmarks. It isolates the tip of the index finger.
2. **Gesture Analysis (`gesture.py`)**: Tracks the position of the index finger over a sliding time window. It calculates the revolutions per second (rev/s) of circular motion and determines if the motion is `SLOW`, `FAST`, or `INACTIVE`.
3. **Controller (`controller.py`)**: Based on the current gesture state (and defined speed thresholds), it dispatches the configured key press at specific intervals using `pynput` or `pyautogui`.
4. **Main Loop (`main.py`)**: Ties the components together, rendering the debug HUD to provide visual feedback to the user.
