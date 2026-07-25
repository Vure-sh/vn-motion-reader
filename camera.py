"""Camera and MediaPipe hand tracking module for VN Motion Reader."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os

class HandTracker:
    def __init__(self, camera_index=0, max_hands=1, detection_con=0.7, track_con=0.7):
        # Try finding a working camera index and backend
        self.cap = None
        
        # Test candidate camera indices (e.g. 0, 1, 2, 3)
        candidate_indices = [camera_index] + [i for i in range(4) if i != camera_index]
        
        for idx in candidate_indices:
            # Try DirectShow backend first (Windows default), then fallback
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                cap_test = cv2.VideoCapture(idx, backend)
                if cap_test.isOpened():
                    ret, test_frame = cap_test.read()
                    if ret and test_frame is not None:
                        self.cap = cap_test
                        print(f"[Camera] Successfully opened webcam index {idx}")
                        break
                    else:
                        cap_test.release()
            if self.cap and self.cap.isOpened():
                break

        # If none opened, do a standard fallback
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)

        # Download the required model file if it doesn't exist
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe model (first run only)...")
            urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.width = 640
        self.height = 480

        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        """Reads a frame, flips it horizontally for mirror view, and tracks hand."""
        if not self.is_opened():
            return None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        # Flip horizontally for natural mirror interaction
        frame = cv2.flip(frame, 1)
        self.height, self.width, _ = frame.shape

        # Convert to RGB for MediaPipe Tasks API
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the image
        results = self.detector.detect(mp_image)

        index_tip = None

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Manual skeleton drawing
                connections = [
                    (0,1), (1,2), (2,3), (3,4), # Thumb
                    (0,5), (5,6), (6,7), (7,8), # Index
                    (5,9), (9,10), (10,11), (11,12), # Middle
                    (9,13), (13,14), (14,15), (15,16), # Ring
                    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start_pt = hand_landmarks[start_idx]
                        end_pt = hand_landmarks[end_idx]
                        sx, sy = int(start_pt.x * self.width), int(start_pt.y * self.height)
                        ex, ey = int(end_pt.x * self.width), int(end_pt.y * self.height)
                        cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2)
                        cv2.circle(frame, (sx, sy), 3, (0, 0, 255), -1)
                        cv2.circle(frame, (ex, ey), 3, (0, 0, 255), -1)

                # Landmark 8 = INDEX_FINGER_TIP
                if len(hand_landmarks) > 8:
                    lm_8 = hand_landmarks[8]
                    cx, cy = int(lm_8.x * self.width), int(lm_8.y * self.height)
                    index_tip = (cx, cy, time.time())

                    # Highlight index tip with prominent circle
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 14, (0, 200, 255), 2)

        return frame, index_tip

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
