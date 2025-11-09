from pathlib import Path
import cv2
import numpy as np
import time

from gestura.hand_detector import HandDetector
from gestura.recognizer_orb import ORBSignRecognizer
from gestura.keyboard_manager import KeyboardManager
from gestura.calibration import CalibrationWizard
from gestura.subtitle_manager import SubtitleManager

SPECIAL_TOKENS = {"SPACE", "BACK", "CLEAR"}


def draw_fps(frame, fps):
    """Draw FPS counter"""
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps}", (w - 100, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_confidence_bar(frame, conf, x=12, y=88, width=200, height=15):
    """Draw confidence bar visualization"""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    fill_width = int(width * conf)
    if conf > 0.7:
        color = (0, 255, 0)
    elif conf > 0.5:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    cv2.putText(frame, f"{int(conf * 100)}%", (x + width + 10, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_hand_guide_box(frame, hand=None):
    """Draw guide box showing where to place hand"""
    h, w = frame.shape[:2]

    # Calculate box position (center-right area)
    box_size = 280
    box_x = w - box_size - 80
    box_y = (h - box_size) // 2

    # Determine box color
    if hand:
        x0, y0, x1, y1 = hand["bbox"]
        hand_center_x = (x0 + x1) // 2
        hand_center_y = (y0 + y1) // 2

        # Check if hand is inside guide box
        in_box_x = box_x < hand_center_x < box_x + box_size
        in_box_y = box_y < hand_center_y < box_y + box_size

        if in_box_x and in_box_y:
            color = (0, 255, 0)  # Green - good position
            text = "GOOD POSITION!"
            text_color = (0, 255, 0)
        else:
            color = (0, 165, 255)  # Orange - detected but not in box
            text = "Move hand into box"
            text_color = (0, 165, 255)
    else:
        color = (0, 0, 255)  # Red - no hand detected
        text = "Place hand here"
        text_color = (0, 0, 255)

    # Draw dashed box outline
    dash_length = 20
    gap_length = 10
    thickness = 3

    # Top line
    for x in range(box_x, box_x + box_size, dash_length + gap_length):
        cv2.line(frame, (x, box_y), (min(x + dash_length, box_x + box_size), box_y), color, thickness)
    # Bottom line
    for x in range(box_x, box_x + box_size, dash_length + gap_length):
        cv2.line(frame, (x, box_y + box_size), (min(x + dash_length, box_x + box_size), box_y + box_size), color,
                 thickness)
    # Left line
    for y in range(box_y, box_y + box_size, dash_length + gap_length):
        cv2.line(frame, (box_x, y), (box_x, min(y + dash_length, box_y + box_size)), color, thickness)
    # Right line
    for y in range(box_y, box_y + box_size, dash_length + gap_length):
        cv2.line(frame, (box_x + box_size, y), (box_x + box_size, min(y + dash_length, box_y + box_size)), color,
                 thickness)

    # Draw hand icon in center
    icon_size = 60
    icon_x = box_x + (box_size - icon_size) // 2
    icon_y = box_y + (box_size - icon_size) // 2

    # Simple hand icon (palm + fingers)
    cv2.circle(frame, (icon_x + icon_size // 2, icon_y + icon_size // 2), icon_size // 3, color, 2)
    # Fingers
    for angle in [0, 30, 60, -30, -60]:
        rad = np.radians(angle)
        end_x = int(icon_x + icon_size // 2 + np.cos(rad) * icon_size // 2)
        end_y = int(icon_y + icon_size // 2 - np.sin(rad) * icon_size // 2)
        cv2.line(frame, (icon_x + icon_size // 2, icon_y + icon_size // 2), (end_x, end_y), color, 2)

    # Draw instruction text
    text_x = box_x + box_size // 2 - 80
    text_y = box_y + box_size + 30
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


def main():
    ROOT = Path(__file__).resolve().parent
    REFS = ROOT / "resources" / "references"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    hd = HandDetector()
    rec = ORBSignRecognizer(references_path=str(REFS))
    kbd = KeyboardManager()
    subs = SubtitleManager()

    show_help = True
    show_mask = False
    capture_mode = False
    show_guide_box = True  # NEW: Show hand placement guide

    # FPS tracking
    fps_counter = 0
    fps_start = time.time()
    current_fps = 0

    # Error recovery tracking
    no_hand_counter = 0
    MAX_NO_HAND = 150

    # ---- Calibration menu WITH GUIDE BOX ----
    wiz = CalibrationWizard(cap, hd, rec, resources_path=str(REFS))
    # IMPORTANT: Enable box for startup calibration menu
    hd.enable_guide_box(True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # Draw guide box in calibration menu
        hand_preview = hd.detect_hand(frame)
        draw_hand_guide_box(frame, hand_preview)

        cv2.putText(frame, "Calibration", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
        cv2.putText(frame, "Press 1: Simple (A)   2: Advanced (A,E,O)   S: Skip",
                    (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            wiz.run_simple(target_letter="A", conf_thresh=0.50)
            break
        elif key == ord('2'):
            res = wiz.run_advanced(letters=("A", "E", "O"), consec=3, base_conf=0.50)
            wiz.show_summary(res)
            break
        elif key == ord('s'):
            break
    try:
        cv2.destroyWindow("Calibration")
    except:
        pass

    print("\nGestura — Hybrid Enhanced Build (Like Your Friend's!)")
    print("Controls: A=Active  H=Help  M=Mask  G=Guide  C=Capture  K=Calibration  Q=Quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera disconnected.")
                break
            frame = cv2.flip(frame, 1)

            # FPS calculation
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            hand = hd.detect_hand(frame)
            candidate, conf, ready = None, 0.0, False

            # Error recovery
            if not hand:
                no_hand_counter += 1
                if no_hand_counter >= MAX_NO_HAND:
                    print("[Auto-reset] No hand detected for 5s")
                    hd = HandDetector()
                    hd.enable_guide_box(show_guide_box)  # Restore box state
                    rec.reset_history()
                    no_hand_counter = 0
            else:
                no_hand_counter = 0

            if hand:
                label, c, is_ready, _ = rec.predict(hand["roi"], hand["roi_mask"])
                candidate, conf, ready = label, c, is_ready

                if is_ready and label:
                    lab = label.upper()
                    if lab in SPECIAL_TOKENS:
                        if lab == "SPACE":
                            subs.add_space()
                            if kbd.is_active():
                                kbd.press_space()
                        elif lab == "BACK":
                            subs.backspace()
                            if kbd.is_active():
                                kbd.press_backspace()
                        elif lab == "CLEAR":
                            subs.clear()
                    elif len(lab) == 1 and lab.isalpha():
                        subs.add_letter(lab)
                        if kbd.is_active():
                            kbd.type_character(lab)

            # ---- UI overlay ----
            disp = frame.copy()

            # Draw hand guide box
            if show_guide_box:
                draw_hand_guide_box(disp, hand)

            if hand:
                cv2.drawContours(disp, [hand["contour"]], -1, (0, 255, 0), 2)
                x0, y0, x1, y1 = hand["bbox"]
                cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 0, 0), 2)

            status_color = (0, 255, 0) if kbd.is_active() else (0, 0, 255)
            status_text = "ACTIVE" if kbd.is_active() else "INACTIVE (A)"
            cv2.putText(disp, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

            if candidate:
                cv2.putText(disp, f"Pred: {candidate}", (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                draw_confidence_bar(disp, conf)

            if ready and candidate:
                cv2.putText(disp, "READY", (disp.shape[1] - 120, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            elif hand:
                cv2.putText(disp, "Hold steady…", (disp.shape[1] - 210, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if show_help:
                help_lines = [
                    "Controls:",
                    "A - Toggle typing",
                    "H - Toggle help",
                    "M - Toggle mask",
                    "G - Toggle guide box",
                    "C - Capture mode",
                    "K - Calibration",
                    "Q - Quit"
                ]
                xh, yh = disp.shape[1] - 340, 26
                for i, line in enumerate(help_lines):
                    cv2.putText(disp, line, (xh, yh + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (230, 230, 230), 1)

            # Draw FPS
            draw_fps(disp, current_fps)

            # Movie-style subtitles
            disp = subs.draw(disp)
            cv2.imshow("Gestura — Hybrid (Like Your Friend!)", disp)

            if show_mask:
                if hand:
                    cv2.imshow("Hand Mask", hand["roi_mask"])
                else:
                    cv2.imshow("Hand Mask", np.zeros(frame.shape[:2], dtype=np.uint8))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('g'):
                show_guide_box = not show_guide_box
                hd.enable_guide_box(show_guide_box)  # ← FIX: Sync with detector!
                print(f"Guide box: {'ON' if show_guide_box else 'OFF'}")
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    try:
                        cv2.destroyWindow("Hand Mask")
                    except:
                        pass
            elif key == ord('a'):
                st = kbd.toggle_active()
                print(f"Typing {'ENABLED' if st else 'DISABLED'}")
            elif key == ord('k'):
                # Re-open calibration WITH guide box
                hd.enable_guide_box(True)  # Force enable for calibration
                while True:
                    ok2, fr2 = cap.read()
                    if not ok2:
                        break
                    fr2 = cv2.flip(fr2, 1)
                    # Draw guide box in recalibration menu too
                    hand_preview2 = hd.detect_hand(fr2)
                    draw_hand_guide_box(fr2, hand_preview2)
                    cv2.putText(fr2, "Calibration", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
                    cv2.putText(fr2, "Press 1: Simple (A)   2: Advanced (A,E,O)   S: Skip",
                                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
                    cv2.imshow("Calibration", fr2)
                    k2 = cv2.waitKey(1) & 0xFF
                    if k2 == ord('1'):
                        wiz.run_simple(target_letter="A", conf_thresh=0.50)
                        break
                    elif k2 == ord('2'):
                        res = wiz.run_advanced(letters=("A", "E", "O"), consec=3, base_conf=0.50)
                        wiz.show_summary(res)
                        break
                    elif k2 == ord('s'):
                        break
                hd.enable_guide_box(show_guide_box)
                try:
                    cv2.destroyWindow("Calibration")
                except:
                    pass
            elif key == ord('c'):
                capture_mode = not capture_mode
                print(f"Capture mode: {'ON' if capture_mode else 'OFF'}")
            elif capture_mode and key not in (255,):
                try:
                    ch = chr(key)
                    if hand and ch.isalpha():
                        norm = rec.export_normalized_roi(hand["roi"], hand["roi_mask"])
                        if norm is not None:
                            REFS.mkdir(parents=True, exist_ok=True)
                            out = REFS / f"{ch.upper()}.png"
                            cv2.imwrite(str(out), norm)
                            print(f"[Capture] Saved {out}")
                except:
                    pass

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    main()