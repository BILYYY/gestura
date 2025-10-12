import cv2

from hand_detector import HandDetector
from sign_recognizer import SignRecognizer
from keyboard_manager import KeyboardManager


class Gestura:
    """
    Main application class that coordinates all components
    """
    def __init__(self, model_path=None):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")

        # Initialize components
        self.hand_detector = HandDetector()
        self.sign_recognizer = SignRecognizer(model_path)
        self.keyboard_manager = KeyboardManager()

        # UI state
        self.show_help = True


    def draw_ui(self, frame, hand_data, predicted_char=None, confidence=None):
        """
        Draw UI elements on frame
        """
        h, w, c = frame.shape

        # Draw hand visualization if detected
        if hand_data:
            # Draw contour
            cv2.drawContours(frame, [hand_data['contour']], -1, (0, 255, 0), 2)

            # Draw bounding box
            x_min, y_min, x_max, y_max = hand_data['bbox']
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Status indicator
        status_color = (0, 255, 0) if self.keyboard_manager.is_active() else (0, 0, 255)
        status_text = "ACTIVE" if self.keyboard_manager.is_active() else "INACTIVE (Press 'A' to activate)"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Hand detection status
        hand_status = "Hand: DETECTED" if hand_data else "Hand: NOT DETECTED"
        cv2.putText(frame, hand_status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show prediction
        if predicted_char and confidence:
            pred_text = f"Predicted: {predicted_char} ({confidence:.2f})"
            cv2.putText(frame, pred_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        # Help text
        if self.show_help:
            help_lines = [
                "Controls:",
                "A - Toggle Active/Inactive",
                "H - Toggle Help",
                "Q - Quit"
            ]
            for i, line in enumerate(help_lines):
                cv2.putText(frame, line, (w - 280, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame


    def run(self):
        """
        Main application loop
        """
        print("\nControls:")
        print("  A - Toggle Active/Inactive")
        print("  H - Toggle Help Display")
        print("  Q - Quit")

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera disconnected.")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Detect hand
                hand_data = self.hand_detector.detect_hand(frame)

                predicted_char = None
                confidence = None

                # Process hand if detected
                if hand_data:
                    # Predict sign
                    predicted_char, confidence = self.sign_recognizer.predict(hand_data['roi'])

                    # Type character if conditions are met
                    if self.keyboard_manager.is_active() and predicted_char:
                        self.keyboard_manager.type_character(predicted_char)

                # Draw UI
                frame = self.draw_ui(frame, hand_data, predicted_char, confidence)

                # Show frame and mask for debugging
                cv2.imshow("Sign Language Keyboard", frame)
                if hand_data:
                    cv2.imshow("Hand Mask", hand_data['mask'])

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('a'):
                    active = self.keyboard_manager.toggle_active()
                    print(f"Typing {'ENABLED' if active else 'DISABLED'}")
                elif key == ord('h'):
                    self.show_help = not self.show_help

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            self.cleanup()


    def cleanup(self):
        """
        Release resources
        """
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    try:
        # Initialize application
        app = Gestura()
        app.run()
    except Exception as e:
        print(f"Error: {e}")