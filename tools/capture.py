from pathlib import Path
import sys
import cv2
import numpy as np
import time
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gestura.hand_detector import HandDetector
from _shared_utils import (
    detect_hand_in_box, calculate_mask_quality, normalize_to_silhouette,
    draw_fixed_box, get_quality_color_and_status, get_box_color
)
from _skeleton import SkeletonTracker, draw_skeleton_overlay, skeleton_to_json

CAM_INDEX = 0
CAM_BACKEND = cv2.CAP_DSHOW
FORCE_HFLIP = True
FIXED_BOX_SIZE = 280
WARMUP_FRAMES = 15
PNG_COMPRESSION = 9
QUALITY_THRESHOLD = 70.0
NORWEGIAN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ"
SPECIAL_GESTURES = ["SPACE", "DELETE"]


def draw_debug_panel(frame, phase, quality, skeleton, hand, captured=None, mode=None):
    h, w = frame.shape[:2]

    canvas = np.zeros((h + 250, w * 2 + 100, 3), dtype=np.uint8)

    canvas[0:h, 0:w] = frame

    if hand and "roi_mask" in hand:
        mask_display = cv2.cvtColor(hand["roi_mask"], cv2.COLOR_GRAY2BGR)
        mask_h, mask_w = mask_display.shape[:2]

        target_h = h
        target_w = int(mask_w * (h / mask_h))
        mask_resized = cv2.resize(mask_display, (target_w, target_h))

        if target_w > w:
            mask_resized = mask_resized[:, :w]

        canvas[0:target_h, w:w + mask_resized.shape[1]] = mask_resized

        q_color, _ = get_quality_color_and_status(quality)
        cv2.putText(canvas, f"MASK: {quality:.0f}%", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, q_color, 2)
    else:
        cv2.putText(canvas, "NO MASK", (w + w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    cv2.rectangle(canvas, (w, 0), (w * 2 - 1, h - 1), (100, 100, 100), 2)

    info_y = h + 20
    x = 20

    phase_names = {
        "calibrate": "1: CALIBRATE",
        "test": "2: TEST QUALITY",
        "choose": "3: CHOOSE MODE",
        "capture": "4: CAPTURE"
    }
    cv2.putText(canvas, f"PHASE: {phase_names.get(phase, phase)}", (x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
    info_y += 35

    if mode:
        mode_color = (0, 255, 255) if mode == "universal" else (0, 255, 0)
        mode_text = "UNIVERSAL" if mode == "universal" else "PERSONAL"
        cv2.putText(canvas, f"Mode: {mode_text}", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        info_y += 25

    q_color, q_status = get_quality_color_and_status(quality)
    cv2.putText(canvas, f"Quality: {quality:.0f}% - {q_status}", (x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 1)
    info_y += 30

    hand_status = "YES" if hand else "NO"
    hand_color = (0, 255, 0) if hand else (0, 0, 255)
    cv2.putText(canvas, f"Hand: {hand_status}", (x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    x += 150

    if skeleton:
        num_tips = len(skeleton.get("fingertips", []))
        extended = skeleton.get("extended_count", 0)
        hand_type = skeleton.get("hand_type", "?")

        cv2.putText(canvas, f"Skeleton: {num_tips} tips", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        x += 170
        cv2.putText(canvas, f"Extended: {extended}", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        x += 150
        cv2.putText(canvas, f"Hand: {hand_type.upper()}", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
    else:
        cv2.putText(canvas, "Skeleton: NONE", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    x = 20
    info_y += 30

    if hand and "roi_mask" in hand:
        mask = hand["roi_mask"]
        fill = (np.count_nonzero(mask) / mask.size) * 100
        cv2.putText(canvas, f"Mask fill: {fill:.0f}%", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(canvas, "Mask fill: 0%", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    if captured is not None:
        x = 20
        info_y += 40
        total_count = len(NORWEGIAN_LETTERS) + len(SPECIAL_GESTURES)
        cv2.putText(canvas, f"CAPTURED: {len(captured)}/{total_count}", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        info_y += 30

        for i, letter in enumerate(NORWEGIAN_LETTERS):
            row = i // 10
            col = i % 10

            lx = x + col * 60
            ly = info_y + row * 30

            color = (0, 255, 0) if letter in captured else (80, 80, 80)
            thickness = 2 if letter in captured else 1

            cv2.putText(canvas, letter, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        info_y += 90
        for i, gesture in enumerate(SPECIAL_GESTURES):
            lx = x + i * 120
            color = (0, 255, 0) if gesture in captured else (80, 80, 80)
            thickness = 2 if gesture in captured else 1
            cv2.putText(canvas, gesture, (lx, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return canvas


def phase_calibrate_test(cap, hd, box, skeleton_tracker):
    print("\n" + "=" * 70)
    print("  PHASE 1-2: CALIBRATE & TEST")
    print("=" * 70)
    print("\n1. Press C to calibrate (place hand in box)")
    print("2. Check quality > 70%")
    print("3. Press SPACE when ready to continue")
    print("\nESC: Quit")
    print("=" * 70 + "\n")

    quality_history = []
    calibrated = False

    while True:
        ret, frame = cap.read()
        if not ret:
            return None, False

        if FORCE_HFLIP:
            frame = cv2.flip(frame, 1)

        hand, _ = detect_hand_in_box(frame, box, hd)

        quality = 0.0
        skeleton = None

        if hand and "roi_mask" in hand:
            quality = calculate_mask_quality(hand["roi_mask"])
            quality_history.append(quality)
            if len(quality_history) > 60:
                quality_history.pop(0)

            skeleton = skeleton_tracker.extract_skeleton(hand["roi_mask"])

            cv2.drawContours(frame, [hand["contour"]], -1, (0, 255, 0), 2)

            if skeleton:
                hx0, hy0, hx1, hy1 = hand["bbox"]
                hand_w = hx1 - hx0
                hand_h = hy1 - hy0
                draw_skeleton_overlay(frame, skeleton, box_size=(hand_w, hand_h), offset=(hx0, hy0))

        box_color = get_box_color(hand, quality)
        draw_fixed_box(frame, box, box_color)

        phase = "calibrate" if not calibrated else "test"
        canvas = draw_debug_panel(frame, phase, quality, skeleton, hand)

        H, W = frame.shape[:2]
        if not calibrated:
            cv2.putText(canvas, "PRESS C TO CALIBRATE", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, "Then press SPACE to continue", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            if quality >= QUALITY_THRESHOLD:
                cv2.putText(canvas, "QUALITY GOOD!", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(canvas, "Press SPACE to continue", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(canvas, "TESTING QUALITY...", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
                cv2.putText(canvas, f"Need quality > {QUALITY_THRESHOLD}%", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(canvas, "C: Calibrate | SPACE: Continue | ESC: Quit", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Gestura - Complete Capture", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return None, False
        elif key == ord(' '):  # SPACE
            if calibrated and quality >= QUALITY_THRESHOLD:
                avg_quality = np.mean(quality_history[-30:]) if len(quality_history) >= 30 else quality
                print(f"\n✅ Quality good! ({avg_quality:.1f}%)")
                return avg_quality, True
            elif not calibrated:
                print("\n⚠️  Please calibrate first (press C)")
            else:
                print(f"\n⚠️  Quality too low ({quality:.0f}%), need >{QUALITY_THRESHOLD}%")
        elif key == ord('c'):
            print("📍 Calibrating...")
            time.sleep(0.3)
            ret, cal_frame = cap.read()
            if ret:
                if FORCE_HFLIP:
                    cal_frame = cv2.flip(cal_frame, 1)
                cal_hand, _ = detect_hand_in_box(cal_frame, box, hd)
                if cal_hand and "roi" in cal_hand:
                    success = hd.calibrate_from_roi(cal_hand["roi"], cal_hand["roi_mask"],
                                                    pad_h=20, pad_s=40, pad_v=50)
                    if success:
                        print("✅ Calibration successful!")
                        calibrated = True
                        quality_history.clear()
                    else:
                        print("❌ Calibration failed")


def phase_choose_mode(cap, hd, box, quality, skeleton_tracker):
    print("\n" + "=" * 70)
    print("  PHASE 3: CHOOSE MODE")
    print("=" * 70)
    print("\n1 - UNIVERSAL (for everyone)")
    print("2 - PERSONAL (your hand)")
    print("\nESC: Quit")
    print("=" * 70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        if FORCE_HFLIP:
            frame = cv2.flip(frame, 1)

        hand, _ = detect_hand_in_box(frame, box, hd)
        skeleton = None

        if hand and "roi_mask" in hand:
            skeleton = skeleton_tracker.extract_skeleton(hand["roi_mask"])
            cv2.drawContours(frame, [hand["contour"]], -1, (0, 255, 0), 2)

            if skeleton:
                hx0, hy0, hx1, hy1 = hand["bbox"]
                hand_w = hx1 - hx0
                hand_h = hy1 - hy0
                draw_skeleton_overlay(frame, skeleton, box_size=(hand_w, hand_h), offset=(hx0, hy0))

        box_color = get_box_color(hand, quality)
        draw_fixed_box(frame, box, box_color)

        canvas = draw_debug_panel(frame, "choose", quality, skeleton, hand)

        H, W = frame.shape[:2]
        y = 80
        cv2.putText(canvas, "CHOOSE MODE", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
        y += 60

        cv2.putText(canvas, "1 - UNIVERSAL", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 25
        cv2.putText(canvas, "    For everyone", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 50

        cv2.putText(canvas, "2 - PERSONAL", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(canvas, "    Your hand only", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.putText(canvas, "Press 1 or 2 | ESC: Quit", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Gestura - Complete Capture", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return None
        elif key == ord('1'):
            print("✅ Selected: UNIVERSAL")
            return "universal"
        elif key == ord('2'):
            print("✅ Selected: PERSONAL")
            return "personal"


def phase_capture(cap, hd, box, mode, output_dir, skeleton_tracker):
    print("\n" + "=" * 70)
    print(f"  PHASE 4: CAPTURE {mode.upper()}")
    print("=" * 70)
    print("\nPress A-Z, Æ, Ø, Å to capture letters")
    print("Press F1 for SPACE gesture, F2 for DELETE gesture")
    print("ESC: Quit")
    print("=" * 70 + "\n")

    captured = set()
    last_key = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FORCE_HFLIP:
            frame = cv2.flip(frame, 1)

        hand, box_region = detect_hand_in_box(frame, box, hd)

        quality = 0.0
        skeleton = None

        if hand and "roi_mask" in hand:
            quality = calculate_mask_quality(hand["roi_mask"])
            skeleton = skeleton_tracker.extract_skeleton(hand["roi_mask"])

            cv2.drawContours(frame, [hand["contour"]], -1, (0, 255, 0), 2)

            if skeleton:
                hx0, hy0, hx1, hy1 = hand["bbox"]
                hand_w = hx1 - hx0
                hand_h = hy1 - hy0
                draw_skeleton_overlay(frame, skeleton, box_size=(hand_w, hand_h), offset=(hx0, hy0))

        box_color = get_box_color(hand, quality)
        draw_fixed_box(frame, box, box_color)

        canvas = draw_debug_panel(frame, "capture", quality, skeleton, hand, captured=captured, mode=mode)

        H, W = frame.shape[:2]
        cv2.putText(canvas, f"CAPTURE {mode.upper()}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        status = "Press A-Z, F1(SPACE), F2(DELETE)" if hand else "Place hand in box"
        status_color = (0, 255, 0) if hand else (0, 200, 255)
        cv2.putText(canvas, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

        cv2.putText(canvas, "ESC: Quit", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Gestura - Complete Capture", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == last_key:
            continue
        last_key = key

        letter = None

        if ord('a') <= key <= ord('z'):
            letter = chr(key).upper()
        elif ord('A') <= key <= ord('Z'):
            letter = chr(key)
        elif key == 0:  # F1
            letter = "SPACE"
        elif key == 1:  # F2
            letter = "DELETE"

        if not letter:
            continue

        if not hand or box_region is None:
            print(f"❌ {letter}: No hand detected")
            continue

        if skeleton is None:
            print(f"⚠️  {letter}: No skeleton")

        normalized = normalize_to_silhouette(box_region, hand["roi_mask"])
        if normalized is None:
            print(f"❌ {letter}: Normalization failed")
            continue

        img_path = output_dir / f"{letter}.png"
        try:
            success = cv2.imwrite(str(img_path), normalized, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
            if not success:
                print(f"❌ {letter}: Image save failed")
                continue
        except Exception as e:
            print(f"❌ {letter}: Save error - {e}")
            continue

        skeleton_path = output_dir / f"{letter}_skeleton.json"
        try:
            if skeleton:
                with open(skeleton_path, 'w', encoding='utf-8') as f:
                    json.dump(skeleton_to_json(skeleton), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  {letter}: Skeleton save error - {e}")

        captured.add(letter)

        try:
            size_kb = img_path.stat().st_size / 1024
        except:
            size_kb = 0

        skel_info = ""
        if skeleton:
            hand_type = skeleton.get("hand_type", "?")
            extended = skeleton.get("extended_count", 0)
            skel_info = f"[{hand_type[0].upper()} {extended}ext]"

        print(f"✅ {letter}.png ({size_kb:.1f} KB) {skel_info}")

        flash = canvas.copy()
        cv2.putText(flash, f"SAVED {letter}!", (W // 2 - 100, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Gestura - Complete Capture", flash)
        cv2.waitKey(300)

    return captured


def main():
    ROOT = Path(__file__).resolve().parent.parent

    print("\n" + "=" * 70)
    print("  GESTURA - COMPLETE CAPTURE FLOW")
    print("=" * 70)
    print("\n🇳🇴 Norwegian Sign Language (29 letters + 2 gestures)")
    print("\n📋 FLOW:")
    print("  1. Calibrate (press C)")
    print("  2. Test quality (press SPACE when ready)")
    print("  3. Choose mode (1 or 2)")
    print("  4. Capture A-Å + SPACE + DELETE")
    print("=" * 70 + "\n")

    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera error")
        return

    print("✅ Camera opened")

    for _ in range(WARMUP_FRAMES):
        cap.read()
        time.sleep(0.01)

    print("✅ Ready!\n")

    hd = HandDetector()
    hd.enable_guide_box(False)

    skeleton_tracker = SkeletonTracker(smoothing_frames=15)

    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        return

    if FORCE_HFLIP:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]
    S = min(FIXED_BOX_SIZE, W - 10, H - 10)
    box = ((W - S) // 2, (H - S) // 2, (W - S) // 2 + S, (H - S) // 2 + S)

    try:
        quality, success = phase_calibrate_test(cap, hd, box, skeleton_tracker)
        if not success:
            print("❌ Cancelled")
            return

        mode = phase_choose_mode(cap, hd, box, quality, skeleton_tracker)
        if mode is None:
            print("❌ Cancelled")
            return

        if mode == "universal":
            output_dir = ROOT / "resources" / "references"
        else:
            output_dir = ROOT / "resources" / "references_personal"

        output_dir.mkdir(parents=True, exist_ok=True)

        captured = phase_capture(cap, hd, box, mode, output_dir, skeleton_tracker)

        print("\n" + "=" * 70)
        print("  COMPLETE!")
        print("=" * 70)
        print(f"\n  ✅ Captured: {len(captured)}/{len(NORWEGIAN_LETTERS) + len(SPECIAL_GESTURES)}")
        print(f"  📂 {output_dir}")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()