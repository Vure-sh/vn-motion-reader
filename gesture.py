"""Gesture recognition and circular motion detection module for VN Motion Reader."""

import collections
import math
import time
import numpy as np


class GestureState:
    INACTIVE = "INACTIVE"
    SLOW = "SLOW"
    FAST = "FAST"


class CalibrationState:
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATING = "CALIBRATING"
    CALIBRATED = "CALIBRATED"


class CircularMotionAnalyzer:
    def __init__(
        self,
        window_duration=1.2,
        slow_threshold=0.3,
        fast_threshold=1.0,
        min_radius_px=20
    ):
        """
        :param window_duration: Sliding window duration in seconds.
        :param slow_threshold: rev/s lower bound for SLOW state.
        :param fast_threshold: rev/s lower bound for FAST state.
        :param min_radius_px: Default minimum radius to ignore stationary noise.
        """
        self.window_duration = window_duration
        self.slow_threshold = slow_threshold
        self.fast_threshold = fast_threshold
        self.min_radius_px = min_radius_px

        # Sliding window buffer storing (x, y, timestamp)
        self.history = collections.deque()

        # Calibration properties
        self.calibration_state = CalibrationState.NOT_CALIBRATED
        self.calibration_start_time = None
        self.calibration_samples = []
        self.noise_threshold = min_radius_px
        self.calibrated_center = None

        # Tracking state output
        self.current_rev_per_sec = 0.0
        self.current_state = GestureState.INACTIVE
        self.smoothed_rev_per_sec = 0.0

    def start_calibration(self):
        """Triggers a 2-second stationary finger calibration."""
        self.calibration_state = CalibrationState.CALIBRATING
        self.calibration_start_time = time.time()
        self.calibration_samples = []

    def update_calibration(self, point):
        """Processes finger position during the 2-second calibration window."""
        if not point or self.calibration_state != CalibrationState.CALIBRATING:
            return

        cx, cy, ts = point
        self.calibration_samples.append((cx, cy))
        elapsed = ts - self.calibration_start_time

        if elapsed >= 2.0 and len(self.calibration_samples) > 10:
            samples = np.array(self.calibration_samples)
            center = np.mean(samples, axis=0)
            distances = np.linalg.norm(samples - center, axis=1)
            max_jitter = np.max(distances)

            # Set noise threshold with safety margin
            self.noise_threshold = max(self.min_radius_px, max_jitter * 1.8)
            self.calibrated_center = (int(center[0]), int(center[1]))
            self.calibration_state = CalibrationState.CALIBRATED
            return True
        return False

    def update(self, point):
        """
        Updates motion tracking with a new fingertip position (x, y, timestamp).
        Returns tuple: (state, rev_per_sec, trail_points, centroid, radius)
        """
        now = time.time()

        if self.calibration_state == CalibrationState.CALIBRATING:
            self.update_calibration(point)
            return self.calibration_state, 0.0, [], None, self.noise_threshold

        if not point:
            # Clear history if tracking is lost
            self.history.clear()
            self.current_rev_per_sec = 0.0
            self.smoothed_rev_per_sec = 0.0
            self.current_state = GestureState.INACTIVE
            return self.current_state, 0.0, [], None, self.noise_threshold

        # Append new point and prune old points outside window
        self.history.append(point)
        while self.history and (now - self.history[0][2]) > self.window_duration:
            self.history.popleft()

        if len(self.history) < 5:
            self.current_state = GestureState.INACTIVE
            return self.current_state, 0.0, list(self.history), None, self.noise_threshold

        # Calculate centroid of current motion window
        coords = np.array([(p[0], p[1]) for p in self.history])
        centroid = np.mean(coords, axis=0)
        cx, cy = centroid[0], centroid[1]

        # Calculate average radius from centroid
        radii = np.linalg.norm(coords - centroid, axis=1)
        avg_radius = np.mean(radii)

        # Ignore tiny movements below noise threshold
        if avg_radius < self.noise_threshold:
            self.smoothed_rev_per_sec = 0.0
            self.current_state = GestureState.INACTIVE
            return self.current_state, 0.0, list(self.history), (int(cx), int(cy)), avg_radius

        # Compute cumulative angular displacement around centroid
        total_cw_angle = 0.0
        time_span = self.history[-1][2] - self.history[0][2]

        if time_span <= 0.1:
            return self.current_state, 0.0, list(self.history), (int(cx), int(cy)), avg_radius

        prev_angle = None
        for p in self.history:
            # In screen coords (y down), atan2(y-cy, x-cx) increases clockwise
            angle = math.atan2(p[1] - cy, p[0] - cx)
            if prev_angle is not None:
                d_theta = angle - prev_angle
                # Normalize d_theta into [-pi, pi]
                while d_theta > math.pi:
                    d_theta -= 2 * math.pi
                while d_theta < -math.pi:
                    d_theta += 2 * math.pi

                # Clockwise rotation in screen coords has d_theta > 0
                if d_theta > 0:
                    total_cw_angle += d_theta
                else:
                    # Ignore counter-clockwise motion (user requested CCW to be ignored)
                    pass
            prev_angle = angle

        # Calculate revolutions per second (rev/s)
        raw_rev_per_sec = (total_cw_angle / (2 * math.pi)) / time_span

        # Exponential Moving Average for smooth output
        alpha = 0.35
        self.smoothed_rev_per_sec = (alpha * raw_rev_per_sec) + ((1 - alpha) * self.smoothed_rev_per_sec)

        # Classify state based on thresholds
        if self.smoothed_rev_per_sec >= self.fast_threshold:
            self.current_state = GestureState.FAST
        elif self.smoothed_rev_per_sec >= self.slow_threshold:
            self.current_state = GestureState.SLOW
        else:
            self.current_state = GestureState.INACTIVE

        return (
            self.current_state,
            self.smoothed_rev_per_sec,
            list(self.history),
            (int(cx), int(cy)),
            avg_radius
        )
