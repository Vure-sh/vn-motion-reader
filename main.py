"""VN Motion Reader - Main Application Entry Point

Controls Visual Novel text progression using hand gesture speed.
"""

import sys
import time
import cv2
import numpy as np

from camera import HandTracker
from gesture import CircularMotionAnalyzer, GestureState, CalibrationState
from controller import KeyboardController

# Optional global keyboard listener for F8 and ESC hotkeys
try:
    from pynput import keyboard as pynput_keyboard
    GLOBAL_HOTKEYS_AVAILABLE = True
except ImportError:
    GLOBAL_HOTKEYS_AVAILABLE = False


class VNMotionReaderApp:
    def __init__(self):
        print("==================================================")
        print("         VN MOTION READER - STARTING            ")
        print("==================================================")
        print("Controls:")
        print("  F8 or 'f'  : Toggle Gesture Control ON / OFF")
        print("  'c'        : Calibrate finger jitter (2-sec hold still)")
        print("  'k'        : Change VN next-text key")
        print("  ESC or 'q' : Emergency Stop / Quit")
        print("==================================================")

        self.tracker = HandTracker(camera_index=0)
        if not self.tracker.is_opened():
            print("[Error] Could not open webcam.")
            sys.exit(1)

        self.analyzer = CircularMotionAnalyzer(
            window_duration=1.2,
            slow_threshold=0.3,
            fast_threshold=1.0,
            min_radius_px=20
        )
        self.controller = KeyboardController(target_key='enter')

        self.running = True
        self.last_press_visual_time = 0.0

        # Global hotkey listener
        self.listener = None
        if GLOBAL_HOTKEYS_AVAILABLE:
            self.setup_global_hotkeys()

    def setup_global_hotkeys(self):
        def on_press(key):
            try:
                if key == pynput_keyboard.Key.f8:
                    new_state = self.controller.toggle_enabled()
                    print(f"[Hotkey F8] Gesture Control Enabled: {new_state}")
                elif key == pynput_keyboard.Key.esc:
                    print("[Hotkey ESC] Emergency Stop triggered!")
                    self.running = False
            except Exception as e:
                pass

        self.listener = pynput_keyboard.Listener(on_press=on_press)
        self.listener.daemon = True
        self.listener.start()

    def draw_hud(self, frame, state, rev_per_sec, trail, centroid, radius, key_triggered):
        h, w, _ = frame.shape

        # Semi-transparent top header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 105), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title & Safety status
        cv2.putText(frame, "VN MOTION READER", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        enabled_str = "CONTROL: ENABLED" if self.controller.enabled else "CONTROL: DISABLED (F8)"
        enabled_color = (0, 255, 0) if self.controller.enabled else (0, 0, 255)
        cv2.putText(frame, enabled_str, (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, enabled_color, 2)

        # Calibration state or active tracking stats
        calib_state = self.analyzer.calibration_state
        if calib_state == CalibrationState.CALIBRATING:
            elapsed = time.time() - self.analyzer.calibration_start_time
            remaining = max(0.0, 2.0 - elapsed)
            calib_str = f"CALIBRATING... Hold finger still ({remaining:.1f}s)"
            cv2.putText(frame, calib_str, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            # Gesture State Badge
            state_colors = {
                GestureState.INACTIVE: (150, 150, 150),
                GestureState.SLOW: (0, 220, 255),
                GestureState.FAST: (0, 255, 0)
            }
            state_color = state_colors.get(state, (255, 255, 255))
            cv2.putText(frame, f"State: {state}", (15, 65), cv2.FONT_HERSHEY_DUPLEX, 0.7, state_color, 2)

            # Speed in rev/s
            cv2.putText(frame, f"Speed: {rev_per_sec:.2f} rev/s", (210, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Active Key
            cv2.putText(frame, f"Key: [{self.controller.target_key.upper()}]", (w - 240, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Instructions Footer
        cv2.putText(frame, "[C] Calibrate | [F8/F] Toggle | [K] Key | [ESC/Q] Quit", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Draw motion trail and centroid
        if trail and len(trail) > 1:
            points = np.array([(p[0], p[1]) for p in trail], dtype=np.int32)
            cv2.polylines(frame, [points], False, (255, 100, 0), 2)

        if centroid and radius > 0:
            cv2.circle(frame, centroid, int(radius), (255, 255, 0), 1)
            cv2.circle(frame, centroid, 4, (255, 255, 0), -1)

        # Visual flash when auto-key is sent
        if key_triggered:
            self.last_press_visual_time = time.time()

        if (time.time() - self.last_press_visual_time) < 0.25:
            cv2.rectangle(frame, (w - 240, 75), (w - 15, 98), (0, 255, 0), -1)
            cv2.putText(frame, ">> KEY SENT <<", (w - 230, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def run(self):
        cv2.namedWindow("VN Motion Reader Debug", cv2.WINDOW_AUTOSIZE)

        while self.running:
            frame, index_tip = self.tracker.read_frame()
            if frame is None:
                break

            # Process motion analyzer
            state, rev_per_sec, trail, centroid, radius = self.analyzer.update(index_tip)

            # Process keyboard output controller
            key_triggered = self.controller.update(state)

            # Render overlay HUD
            self.draw_hud(frame, state, rev_per_sec, trail, centroid, radius, key_triggered)

            cv2.imshow("VN Motion Reader Debug", frame)

            # Handle OpenCV window key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q
                print("[Exit] Emergency Stop / Quit requested.")
                self.running = False
            elif key == ord('f') or key == ord('F'):
                new_st = self.controller.toggle_enabled()
                print(f"[Toggle] Gesture Control Enabled: {new_st}")
            elif key == ord('c') or key == ord('C'):
                print("[Calibration] Hold finger still for 2 seconds...")
                self.analyzer.start_calibration()
            elif key == ord('k') or key == ord('K'):
                new_key = self.controller.cycle_target_key()
                print(f"[Key Config] VN Next-Text Key set to: {new_key.upper()}")

        self.cleanup()

    def cleanup(self):
        if self.listener:
            self.listener.stop()
        self.tracker.release()
        print("[System] VN Motion Reader shut down safely.")


if __name__ == '__main__':
    app = VNMotionReaderApp()
    app.run()
