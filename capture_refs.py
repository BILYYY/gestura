from pathlib import Path
import cv2

from gestura.hand_detector import HandDetector
from gestura.recognizer_orb import ORBSignRecognizer


def main():
    ROOT = Path(__file__).resolve().parent
    REFS = ROOT / "resources" / "references"
    REFS.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error"); return

    hd = HandDetector()
    rz = ORBSignRecognizer(references_path=str(REFS))  # for normalization util only

    print("\nReference capture tool")
    print("Hold pose; press A–Z to save; Q to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            d = hd.detect_hand(frame)

            vis = frame.copy()
            if d:
                x0, y0, x1, y1 = d["bbox"]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, "Hand detected", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(vis, "Press A–Z to capture, Q to quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Capture References", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

            letter = None
            if ord('a') <= key <= ord('z'):
                letter = chr(key).upper()
            elif ord('A') <= key <= ord('Z'):
                letter = chr(key)

            if letter and d:
                norm = rz.export_normalized_roi(d["roi"], d["roi_mask"])
                if norm is not None:
                    out = REFS / f"{letter}.png"
                    cv2.imwrite(str(out), norm)
                    print(f"Saved {out}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Capture finished.")


if __name__ == "__main__":
    main()
