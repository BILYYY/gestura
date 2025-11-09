# capture_refs.py
from pathlib import Path
import time
import string
import cv2

from gestura.hand_detector import HandDetector
from gestura.recognizer_orb import ORBSignRecognizer


def to_square(binary_img):
    """Pad a binary/grayscale image to a centered square canvas (black background)."""
    h, w = binary_img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(binary_img, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=0)


def normalize_with_fallback(roi_bgr, roi_mask):
    """
    Normalize ROI to a clean 200x200 single-channel silhouette.
    Expects:
      - roi_bgr: ROI in BGR (uint8)
      - roi_mask: 0/255 mask (uint8), same HxW as roi_bgr
    Returns:
      - 200x200 uint8 single-channel image (0=background, 255=hand)
    """
    if roi_bgr is None or roi_mask is None:
        return None

    # Ensure single channel through grayscale, then apply mask
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    fg = cv2.bitwise_and(gray, gray, mask=roi_mask)

    # Otsu binarization for a clean silhouette
    _, bin_img = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional polarity check: ensure hand is white on black background
    # Count white vs black pixels inside bbox; if mostly black, invert.
    white_ratio = (bin_img > 0).mean()
    if white_ratio < 0.5:
        bin_img = cv2.bitwise_not(bin_img)

    # Center on a square canvas, then resize
    sq = to_square(bin_img)
    norm = cv2.resize(sq, (200, 200), interpolation=cv2.INTER_AREA)

    return norm


def main():
    ROOT = Path(__file__).resolve().parent
    REFS = ROOT / "resources" / "references"
    REFS.mkdir(parents=True, exist_ok=True)

    # Camera init
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; safe elsewhere
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera error")
        return

    # Init models
    hd = HandDetector()
    rz = ORBSignRecognizer(references_path=str(REFS))  # used for built-in normalization if available

    print("\nReference capture tool")
    print("Hold pose; press A–Z to save; Q to quit.\n")

    # Camera warm-up (auto-exposure/white-balance settle)
    warmup_frames = 20
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.01)

    # Debounce key repeats
    last_key = None

    # Helpful HUD
    instructions = "Press A–Z to capture, Q to quit"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Detect hand
            d = hd.detect_hand(frame)

            vis = frame.copy()
            if d and "bbox" in d:
                x0, y0, x1, y1 = d["bbox"]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis, "Hand detected", (x0, max(20, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Show one hand inside the box area",
                            (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            # HUD
            cv2.putText(vis, instructions,
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Capture References", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Only handle a key once until it changes (avoid spamming)
            if key == last_key:
                continue
            last_key = key

            # Letter capture A–Z
            letter = None
            if ord('a') <= key <= ord('z'):
                letter = chr(key).upper()
            elif ord('A') <= key <= ord('Z'):
                letter = chr(key)

            if letter and d:
                # Prefer project’s normalizer for consistency with the recognizer
                norm = None
                try:
                    if "roi" in d and "roi_mask" in d:
                        norm = rz.export_normalized_roi(d["roi"], d["roi_mask"])
                except Exception as e:
                    print(f"export_normalized_roi failed: {e}")

                # Fallback normalization (robust & consistent)
                if norm is None and "roi" in d and "roi_mask" in d:
                    norm = normalize_with_fallback(d["roi"], d["roi_mask"])

                if norm is None:
                    print("No valid ROI/mask to save; hold steady and ensure the hand is inside the box.")
                    continue

                # Quality gate: reject blurry frames
                try:
                    lap_var = cv2.Laplacian(norm, cv2.CV_64F).var()
                except Exception:
                    lap_var = 1.0  # If something odd, don't block save
                if lap_var < 25:
                    print("Frame too blurry; hold steady and try again.")
                    continue

                out = REFS / f"{letter}.png"
                ok = cv2.imwrite(str(out), norm, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                if ok:
                    kb = Path(out).stat().st_size / 1024
                    print(f"Saved {out} ({kb:.1f} KB)")
                    # Brief visual confirmation on the window
                    cv2.putText(vis, f"Saved {letter}.png",
                                (12, vis.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Capture References", vis)
                    cv2.waitKey(100)  # tiny pause so user sees confirmation
                else:
                    print("cv2.imwrite failed")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Capture finished.")


if __name__ == "__main__":
    main()
