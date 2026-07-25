"""Camera and MediaPipe hand tracking module for VN Motion Reader."""

import cv2
import mediapipe as mp
import time


class HandTracker:
    def __init__(self, camera_index=0, max_hands=1, detection_con=0.7, track_con=0.7):
        self.cap = cv2.VideoCapture(camera_index)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.width = 640
        self.height = 480

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def is_opened(self):
        return self.cap.isOpened()

    def read_frame(self):
        """Reads a frame, flips it horizontally for mirror view, and tracks hand."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Flip horizontally for natural mirror interaction
        frame = cv2.flip(frame, 1)
        self.height, self.width, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        index_tip = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw standard hand skeleton
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )

                # Landmark 8 = INDEX_FINGER_TIP
                lm_8 = hand_landmarks.landmark[8]
                cx, cy = int(lm_8.x * self.width), int(lm_8.y * self.height)
                index_tip = (cx, cy, time.time())

                # Highlight index tip with prominent circle
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (cx, cy), 14, (0, 200, 255), 2)

        return frame, index_tip

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
