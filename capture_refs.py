import os
import cv2
from gestura.hand_detector import HandDetector
from gestura.recognizer_orb import ORBSignRecognizer

SAVE_DIR = "resources/references"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    ensure_dir(SAVE_DIR)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    hd = HandDetector()
    # Use recognizer only for the normalization routine
    rz = ORBSignRecognizer(references_path=SAVE_DIR)

    print("\nReference capture.")
    print("Hold the pose; press A–Z to save; Q to quit.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            d = hd.detect_hand(frame)

            vis = frame.copy()
            if d:
                x0, y0, x1, y1 = d["bbox"]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, "Hand detected", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(vis, "Press A–Z to capture, Q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Capture References", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Letter save
            letter = None
            if ord('a') <= key <= ord('z'):
                letter = chr(key).upper()
            elif ord('A') <= key <= ord('Z'):
                letter = chr(key)

            if letter and d:
                gray, _mask = rz._normalize_roi(d["roi"], d["roi_mask"], 200)
                out = os.path.join(SAVE_DIR, f"{letter}.png")
                cv2.imwrite(out, gray)
                print(f"Saved {out}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Capture finished.")


if __name__ == "__main__":
    main()
