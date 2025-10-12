from pynput.keyboard import Controller
import time


class KeyboardManager:
    """
    Manages keyboard simulation and application state
    """
    def __init__(self):
        # Initialize keyboard controller
        self.keyboard = Controller()

        # State management
        self.active = False
        self.last_typed_char = None
        self.last_typed_time = 0
        self.min_time_between_chars = 1.5  # TODO: find out best time interval


    def is_active(self):
        """Check if typing is active"""
        return self.active


    def toggle_active(self):
        """Toggle typing on/off"""
        self.active = not self.active
        return self.active


    def can_type_character(self, char):
        """
        Check if character can be typed based on timing and repetition
        """
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_typed_time < self.min_time_between_chars:
            return False

        return True


    def type_character(self, char):
        """
        Simulate keyboard input for the given character
        """
        if not self.can_type_character(char):
            return False

        try:
            self.keyboard.type(char)

            self.last_typed_char = char
            self.last_typed_time = time.time()
            print(f"Typed: {char}")
            return True

        except Exception as e:
            print(f"Error typing character: {e}")
            return False