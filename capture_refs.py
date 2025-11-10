from pathlib import Path
import time
import cv2
import numpy as np

from gestura.hand_detector import HandDetector
try:
    from gestura.recognizer_orb import ORBSignRecognizer
except Exception:
    ORBSignRecognizer = None

# ===================== Config (KEEP SAME across ALL camera apps) =====================
CAM_INDEX = 0
CAM_BACKEND = cv2.CAP_MSMF        # or cv2.CAP_DSHOW — but use the SAME everywhere
FORCE_HFLIP = True                # same as run_gestura.py
FORCE_VFLIP = False
FORCE_ROT180 = False

FIXED_BOX_SIZE = 280
WARMUP_FRAMES = 15

# Saving mode (use normalized silhouettes!)
USE_NORMALIZATION = True          # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
RESIZE_TO = None                  # optional resize of raw crop (ignored when normalized)
PNG_COMPRESSION = 9               # max compression for tiny files

DEBOUNCE_KEYS = True
# =====================================================================================

def safe_crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w,     x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h,     y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]

def to_square(binary_img):
    h, w = binary_img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(binary_img, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=0)

def normalize_with_fallback(roi_bgr, roi_mask):
    if roi_bgr is None or roi_mask is None:
        return None
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    fg = cv2.bitwise_and(gray, gray, mask=roi_mask)
    _, bin_img = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure hand is white on black
    if (bin_img > 0).mean() < 0.5:
        bin_img = cv2.bitwise_not(bin_img)
    sq = to_square(bin_img)
    return cv2.resize(sq, (200, 200), interpolation=cv2.INTER_AREA)

def main():
    ROOT = Path(__file__).resolve().parent
    REFS = ROOT / "resources" / "references"
    REFS.mkdir(parents=True, exist_ok=True)

    # Camera (same backend everywhere)
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera error")
        return

    hd = HandDetector()
    rz = ORBSignRecognizer(references_path=str(REFS)) if (USE_NORMALIZATION and ORBSignRecognizer) else None

    print("\nReference capture tool (FIXED BOX, ORIENTATION LOCKED)")
    print("Saving: normalized 200x200 silhouettes (white hand on black)")
    print("Keys: A–Z save | Q quit\n")

    # Warm-up
    for _ in range(WARMUP_FRAMES):
        cap.read()
        time.sleep(0.01)

    last_key = None
    box = None  # (x0,y0,x1,y1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Orientation lock — MUST match main app
            if FORCE_HFLIP:
                frame = cv2.flip(frame, 1)
            if FORCE_VFLIP:
                frame = cv2.flip(frame, 0)
            if FORCE_ROT180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            H, W = frame.shape[:2]
            if box is None:
                S = min(FIXED_BOX_SIZE, W - 10, H - 10)
                x0 = (W - S)//2; y0 = (H - S)//2
                x1 = x0 + S;     y1 = y0 + S
                box = (x0, y0, x1, y1)

            # Detector only for mask/status (box is fixed)
            d = hd.detect_hand(frame)
            mask_full = d["mask"] if (d and "mask" in d) else None

            vis = frame.copy()
            # Draw fixed guide box (cyan)
            x0, y0, x1, y1 = box
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)

            status = "Hand detected" if d else "Align hand in the box"
            color = (0, 255, 0) if d else (0, 200, 255)
            cv2.putText(vis, status, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.putText(vis, "Press A–Z to capture (fixed box), Q to quit",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Capture References (Fixed Box)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if DEBOUNCE_KEYS and key == last_key:
                continue
            last_key = key

            # Map key to A–Z
            letter = None
            if ord('a') <= key <= ord('z'):
                letter = chr(key).upper()
            elif ord('A') <= key <= ord('Z'):
                letter = chr(key)
            if not letter:
                continue

            # Capture exactly the fixed box
            crop = safe_crop(frame, *box)
            if crop is None:
                print("Fixed box invalid; not saved.")
                continue

            out = REFS / f"{letter}.png"

            if USE_NORMALIZATION:
                roi_mask = safe_crop(mask_full, *box) if mask_full is not None else None
                if roi_mask is not None and roi_mask.ndim == 3:
                    roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)

                norm = None
                if rz is not None and roi_mask is not None:
                    try:
                        norm = rz.export_normalized_roi(crop, roi_mask)
                    except Exception as e:
                        print(f"export_normalized_roi failed: {e}")
                if norm is None and roi_mask is not None:
                    norm = normalize_with_fallback(crop, roi_mask)

                to_save = norm if norm is not None else crop
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION if norm is None else 9]
                mode = "normalized 200x200" if norm is not None else "raw fixed (no mask)"
            else:
                to_save = crop
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
                mode = "raw fixed"

            if RESIZE_TO and to_save is crop:
                to_save = cv2.resize(to_save, RESIZE_TO, interpolation=cv2.INTER_AREA)

            ok = cv2.imwrite(str(out), to_save, save_params)
            if ok:
                kb = Path(out).stat().st_size / 1024
                h, w = to_save.shape[:2]
                print(f"Saved {out} {w}x{h} ({kb:.1f} KB, {mode})")
                cv2.putText(vis, f"Saved {letter}.png", (12, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Capture References (Fixed Box)", vis)
                cv2.waitKey(120)
            else:
                print("cv2.imwrite failed")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Capture finished.")

if __name__ == "__main__":
    main()
