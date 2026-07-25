"""Keyboard controller and input automation module for VN Motion Reader."""

import time
import threading
from gesture import GestureState

# Import pynput as primary, pyautogui as fallback
try:
    from pynput.keyboard import Controller as PynputController, Key
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False


class KeyboardController:
    SUPPORTED_KEYS = {
        'enter': 'Enter',
        'space': 'Space',
        'down': 'Down Arrow',
        'right': 'Right Arrow'
    }

    def __init__(self, target_key='enter', use_pyautogui_fallback=False):
        self.target_key = target_key.lower()
        self.use_pyautogui = use_pyautogui_fallback or (not PYNPUT_AVAILABLE)

        if PYNPUT_AVAILABLE and not self.use_pyautogui:
            self.pynput_keyboard = PynputController()
        else:
            self.pynput_keyboard = None

        # Safety control toggles
        self.enabled = True  # Controlled by F8
        self.last_press_time = 0.0

        # Auto-click intervals (seconds per press)
        self.intervals = {
            GestureState.INACTIVE: float('inf'),
            GestureState.SLOW: 1.5,
            GestureState.FAST: 0.35
        }

        # Lock for thread safety
        self._lock = threading.Lock()

    def toggle_enabled(self):
        with self._lock:
            self.enabled = not self.enabled
            return self.enabled

    def set_enabled(self, value: bool):
        with self._lock:
            self.enabled = value

    def cycle_target_key(self):
        keys = list(self.SUPPORTED_KEYS.keys())
        idx = keys.index(self.target_key) if self.target_key in keys else 0
        self.target_key = keys[(idx + 1) % len(keys)]
        return self.target_key

    def send_key_press(self):
        """Dispatches the configured key press event using pynput or pyautogui."""
        try:
            if not self.use_pyautogui and self.pynput_keyboard:
                if self.target_key == 'enter':
                    self.pynput_keyboard.press(Key.enter)
                    self.pynput_keyboard.release(Key.enter)
                elif self.target_key == 'space':
                    self.pynput_keyboard.press(Key.space)
                    self.pynput_keyboard.release(Key.space)
                elif self.target_key == 'down':
                    self.pynput_keyboard.press(Key.down)
                    self.pynput_keyboard.release(Key.down)
                elif self.target_key == 'right':
                    self.pynput_keyboard.press(Key.right)
                    self.pynput_keyboard.release(Key.right)
            elif PYAUTOGUI_AVAILABLE:
                pyautogui.press(self.target_key)
        except Exception as e:
            print(f"[Controller Error] Failed to send key press: {e}")

    def update(self, state):
        """
        Called in the loop with the current GestureState.
        Triggers key press if enabled and interval has elapsed.
        """
        if not self.enabled or state == GestureState.INACTIVE:
            return False

        interval = self.intervals.get(state, float('inf'))
        now = time.time()

        if (now - self.last_press_time) >= interval:
            self.last_press_time = now
            # Execute key press in a light thread to prevent frame drops
            threading.Thread(target=self.send_key_press, daemon=True).start()
            return True

        return False
