from pynput.keyboard import Controller
import time


class KeyboardManager:
    def __init__(self):
        self.keyboard = Controller()
        self._active = False
        self._last_typed = 0.0
        self.min_interval = 1.2  # seconds

    def is_active(self):
        return self._active

    def toggle_active(self):
        self._active = not self._active
        return self._active

    def _cooldown_ok(self):
        return (time.time() - self._last_typed) >= self.min_interval

    def type_character(self, ch: str):
        if not self._cooldown_ok():
            return False
        try:
            self.keyboard.type(ch)
            self._last_typed = time.time()
            print(f"Typed: {ch}")
            return True
        except Exception as e:
            print(f"Typing error: {e}")
            return False
