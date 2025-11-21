from pynput.keyboard import Controller, Key
import time


class KeyboardManager:
    def __init__(self):
        self.keyboard = Controller()
        self._active = False
        self._last = 0.0
        self.min_interval = 1.0  # seconds

    def is_active(self):
        return self._active

    def toggle_active(self):
        self._active = not self._active
        return self._active

    def _ok(self):
        return (time.time() - self._last) >= self.min_interval

    def type_character(self, ch: str):
        """Type character including Norwegian letters Æ, Ø, Å"""
        if not self._active or not self._ok(): return False
        try:
            # Handle Norwegian letters
            if ch.upper() == 'Æ':
                self.keyboard.type('æ' if ch.islower() else 'Æ')
            elif ch.upper() == 'Ø':
                self.keyboard.type('ø' if ch.islower() else 'Ø')
            elif ch.upper() == 'Å':
                self.keyboard.type('å' if ch.islower() else 'Å')
            else:
                self.keyboard.type(ch)

            self._last = time.time()
            print(f"Typed: {ch}")
            return True
        except Exception as e:
            print(f"Typing error: {e}")
            return False

    def press_space(self):
        if not self._active or not self._ok(): return False
        try:
            self.keyboard.press(Key.space);
            self.keyboard.release(Key.space)
            self._last = time.time()
            print("Typed: <SPACE>")
            return True
        except Exception as e:
            print(f"Space error: {e}")
            return False

    def press_backspace(self):
        if not self._active or not self._ok(): return False
        try:
            self.keyboard.press(Key.backspace);
            self.keyboard.release(Key.backspace)
            self._last = time.time()
            print("Typed: <BACKSPACE>")
            return True
        except Exception as e:
            print(f"Backspace error: {e}")
            return False