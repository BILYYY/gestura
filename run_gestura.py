import cv2

from gestura.hand_detector import HandDetector
from gestura.recognizer_orb import ORBSignRecognizer
from gestura.keyboard_manager import KeyboardManager
from gestura.ui import draw_ui


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    detector = HandDetector()
    recognizer = ORBSignRecognizer(references_path="resources/references")
    keyboard = KeyboardManager()

    show_help = True
    show_mask = False

    print("\nControls: A=Active  H=Help  M=Mask  R=Reset  Q=Quit\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera disconnected.")
                break

            frame = cv2.flip(frame, 1)
            hand = detector.detect_hand(frame)

            pred_letter, pred_conf, stable = None, 0.0, False

            if hand:
                letter, conf, is_stable, _ = recognizer.predict(hand["roi"], hand["roi_mask"])
                if is_stable:
                    pred_letter, pred_conf, stable = letter, conf, True
                    if keyboard.is_active() and letter:
                        keyboard.type_character(letter)
                else:
                    pred_letter, pred_conf = letter, conf

                if show_mask:
                    cv2.imshow("Mask", hand["roi_mask"])

            frame = draw_ui(
                frame=frame,
                hand_data=hand,
                active=keyboard.is_active(),
                pred_letter=pred_letter,
                pred_conf=pred_conf,
                stable=stable,
                show_help=show_help
            )
            cv2.imshow("Gestura — NSL Keyboard (Final Build)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                keyboard.toggle_active()
                print(f"Typing {'ENABLED' if keyboard.is_active() else 'DISABLED'}")
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    try:
                        cv2.destroyWindow("Mask")
                    except:
                        pass
            elif key == ord('r'):
                recognizer.reset_history()
                print("Stability reset.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    main()
