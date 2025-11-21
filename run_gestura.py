from pathlib import Path
import cv2
import numpy as np
import time
import sys

from gestura.hand_detector import HandDetector
from gestura.recognizer_orb import ORBSignRecognizer
from gestura.keyboard_manager import KeyboardManager
from gestura.calibration import CalibrationWizard
from gestura.subtitle_manager import SubtitleManager

# Add tools folder for skeleton
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
try:
    from tools._skeleton import SkeletonTracker, draw_skeleton_overlay

    SKELETON_AVAILABLE = True
except Exception:
    SKELETON_AVAILABLE = False
    print("[RunGestura] Skeleton module not available")

SPECIAL_TOKENS = {"SPACE", "BACK", "CLEAR"}


def create_3panel_display(frame, hand, skeleton, quality, status_info):
    """Create 3-panel layout: Camera | Mask | Info Panel"""
    h, w = frame.shape[:2]

    # Slightly taller canvas for info
    canvas = np.zeros((h + 300, w * 2 + 100, 3), dtype=np.uint8)

    # LEFT: camera view
    canvas[0:h, 0:w] = frame

    # RIGHT: mask + skeleton on mask
    if hand and "roi_mask" in hand:
        mask_display = cv2.cvtColor(hand["roi_mask"], cv2.COLOR_GRAY2BGR)
        mask_h, mask_w = mask_display.shape[:2]

        target_h = h
        target_w = int(mask_w * (h / mask_h))
        mask_resized = cv2.resize(mask_display, (target_w, target_h))

        if target_w > w:
            mask_resized = mask_resized[:, :w]

        mask_panel = mask_resized.copy()

        # Draw skeleton on mask panel
        if skeleton:
            mask_w_skel, mask_h_skel = skeleton.get("mask_size", (mask_w, mask_h))
            scale_x = mask_panel.shape[1] / float(mask_w_skel)
            scale_y = mask_panel.shape[0] / float(mask_h_skel)

            # Palm
            palm_x = int(skeleton["palm_center"][0] * mask_w_skel * scale_x)
            palm_y = int(skeleton["palm_center"][1] * mask_h_skel * scale_y)
            cv2.circle(mask_panel, (palm_x, palm_y), 8, (0, 255, 255), -1)

            # Fingertips
            for fx_rel, fy_rel in skeleton["fingertips"]:
                fx = int(fx_rel * mask_w_skel * scale_x)
                fy = int(fy_rel * mask_h_skel * scale_y)
                cv2.line(mask_panel, (palm_x, palm_y), (fx, fy), (0, 255, 0), 2)
                cv2.circle(mask_panel, (fx, fy), 6, (255, 0, 255), -1)

        canvas[0:target_h, w:w + mask_panel.shape[1]] = mask_panel

        # Quality text
        q_text = f"MASK: {quality:.0f}%"
        if quality >= 70:
            q_color = (0, 255, 0)
        elif quality >= 50:
            q_color = (0, 165, 255)
        else:
            q_color = (0, 0, 255)
        cv2.putText(canvas, q_text, (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, q_color, 2)
    else:
        cv2.putText(canvas, "NO MASK", (w + w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    cv2.rectangle(canvas, (w, 0), (w * 2 - 1, h - 1), (100, 100, 100), 2)

    # === INFO PANEL (bottom) ===
    info_y = h + 20
    x = 20

    status_text = status_info.get("status", "INACTIVE")
    status_color = (0, 255, 0) if status_text == "ACTIVE" else (0, 0, 255)
    cv2.putText(canvas, f"STATUS: {status_text}", (x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    info_y += 35

    # Prediction + confidence bar
    if status_info.get("prediction"):
        pred = status_info["prediction"]
        conf = status_info.get("confidence", 0.0)
        cv2.putText(canvas, f"Prediction: {pred} ({conf:.2f})", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        bar_x = x + 250
        bar_y = info_y - 15
        bar_w = 200
        bar_h = 15
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * conf)
        bar_color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

        info_y += 25

        # Matching info
        match_info = status_info.get("match_info", {})

        if match_info.get("best_match"):
            best_match = match_info["best_match"]
            cv2.putText(canvas, f"Best Match: {best_match}", (x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1)
            info_y += 20

        if match_info.get("match_count"):
            match_count = match_info["match_count"]
            cv2.putText(canvas, f"Good Matches: {match_count}", (x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            info_y += 20

        if match_info.get("shape_similarity") is not None:
            shape_sim = match_info["shape_similarity"]
            cv2.putText(canvas, f"Shape Similarity: {shape_sim:.2f}", (x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
            info_y += 20

        if match_info.get("closeness_score") is not None:
            closeness = match_info["closeness_score"]
            cv2.putText(canvas, f"Closeness: {closeness:.2f}", (x, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 255), 1)
            info_y += 20
    else:
        info_y += 30

    # Skeleton info
    if skeleton:
        num_tips = len(skeleton.get("fingertips", []))
        extended = skeleton.get("extended_count", 0)
        hand_type = skeleton.get("hand_type", "?")

        cv2.putText(canvas, f"Skeleton: {num_tips} tips", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        x2 = x + 170
        cv2.putText(canvas, f"Extended: {extended}", (x2, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        x3 = x2 + 150
        cv2.putText(canvas, f"Hand: {hand_type.upper()}", (x3, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
    else:
        cv2.putText(canvas, "Skeleton: NONE", (x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    info_y += 30
    x = 20

    # FPS
    fps = status_info.get("fps", 0)
    cv2.putText(canvas, f"FPS: {fps}", (x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # READY indicator
    if status_info.get("ready"):
        cv2.putText(canvas, "READY TO TYPE!", (canvas.shape[1] - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Help text
    if status_info.get("show_help"):
        help_lines = [
            "CONTROLS:",
            "A - Toggle typing",
            "H - Toggle help",
            "G - Toggle guide",
            "K - Calibration",
            "ESC - Quit"
        ]
        help_x = canvas.shape[1] - 250
        help_y = 100
        for line in help_lines:
            cv2.putText(canvas, line, (help_x, help_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            help_y += 22

    # Reference thumbnail (closest reference image)
    ref_thumb = status_info.get("ref_thumb")
    if ref_thumb is not None:
        th, tw = ref_thumb.shape[:2]
        y0 = canvas.shape[0] - th - 20
        x0 = canvas.shape[1] - tw - 20
        canvas[y0:y0 + th, x0:x0 + tw] = ref_thumb
        cv2.putText(canvas, "Reference", (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas


def draw_hand_guide_box(frame, hand=None):
    """Draw dashed guide box for hand placement"""
    h, w = frame.shape[:2]

    box_size = 280
    box_x = w - box_size - 80
    box_y = (h - box_size) // 2

    if hand:
        x0, y0, x1, y1 = hand["bbox"]
        hand_center_x = (x0 + x1) // 2
        hand_center_y = (y0 + y1) // 2

        in_box_x = box_x < hand_center_x < box_x + box_size
        in_box_y = box_y < hand_center_y < box_y + box_size

        if in_box_x and in_box_y:
            color = (0, 255, 0)
            text = "GOOD POSITION!"
            text_color = (0, 255, 0)
        else:
            color = (0, 165, 255)
            text = "Move hand into box"
            text_color = (0, 165, 255)
    else:
        color = (0, 0, 255)
        text = "Place hand here"
        text_color = (0, 0, 255)

    dash_length = 20
    gap_length = 10
    thickness = 3

    for x in range(box_x, box_x + box_size, dash_length + gap_length):
        cv2.line(frame, (x, box_y), (min(x + dash_length, box_x + box_size), box_y), color, thickness)
    for x in range(box_x, box_x + box_size, dash_length + gap_length):
        cv2.line(frame, (x, box_y + box_size), (min(x + dash_length, box_x + box_size), box_y + box_size), color,
                 thickness)
    for y in range(box_y, box_y + box_size, dash_length + gap_length):
        cv2.line(frame, (box_x, y), (box_x, min(y + dash_length, box_y + box_size)), color, thickness)
    for y in range(box_y, box_y + box_size, dash_length + gap_length):
        cv2.line(frame, (box_x + box_size, y), (box_x + box_size, min(y + dash_length, box_y + box_size)), color,
                 thickness)

    icon_size = 60
    icon_x = box_x + (box_size - icon_size) // 2
    icon_y = box_y + (box_size - icon_size) // 2

    cv2.circle(frame, (icon_x + icon_size // 2, icon_y + icon_size // 2), icon_size // 3, color, 2)
    for angle in [0, 30, 60, -30, -60]:
        rad = np.radians(angle)
        end_x = int(icon_x + icon_size // 2 + np.cos(rad) * icon_size // 2)
        end_y = int(icon_y + icon_size // 2 - np.sin(rad) * icon_size // 2)
        cv2.line(frame, (icon_x + icon_size // 2, icon_y + icon_size // 2), (end_x, end_y), color, 2)

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

    # Skeleton mode selection
    use_mediapipe = False  # default
    if SKELETON_AVAILABLE:
        print("\n" + "=" * 70)
        print("  SKELETON TRACKING MODE")
        print("=" * 70)
        print("\n1 - CV MODE (your algorithm, teacher-approved)")
        print("    - Works well, sometimes merges close fingers")
        print("\n2 - MEDIAPIPE MODE (AI-enhanced, more accurate)")
        print("    - Better accuracy, always finds 5 fingers")
        print("    - Requires: pip install mediapipe")
        print("\nWhich mode? (1/2, default=1): ", end="")

        choice = input().strip()
        use_mediapipe = (choice == "2")

        if use_mediapipe:
            print("✅ Using MediaPipe mode\n")
        else:
            print("✅ Using CV mode (your algorithm)\n")

        skeleton_tracker = SkeletonTracker(smoothing_frames=15, use_mediapipe=use_mediapipe)
    else:
        skeleton_tracker = None

    show_help = True
    show_mask = False  # currently unused, kept for future
    capture_mode = False  # currently unused
    show_guide_box = True

    fps_counter = 0
    fps_start = time.time()
    current_fps = 0

    no_hand_counter = 0
    MAX_NO_HAND = 150  # ~5 seconds at 30fps

    # IMPORTANT: prevent "bbbbbb" spam
    last_typed_label = None

    # Calibration wizard
    wiz = CalibrationWizard(cap, hd, rec, resources_path=str(REFS))
    hd.enable_guide_box(True)

    # Calibration screen
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hand_preview = hd.detect_hand(frame)
        draw_hand_guide_box(frame, hand_preview)

        cv2.putText(frame, "Calibration", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
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
    except Exception:
        pass

    print("\nGestura — Enhanced with Skeleton!")
    print("Controls: A=Active  H=Help  G=Guide  K=Calibration  ESC=Quit\n")

    # Simple cache for reference thumbnails
    ref_thumb_cache = {}

    # MAIN LOOP
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera disconnected.")
                break
            frame = cv2.flip(frame, 1)

            # FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            hand = hd.detect_hand(frame)
            candidate, conf, ready = None, 0.0, False
            skeleton = None
            quality = 0.0
            match_info = {}

            # No hand timeout
            if not hand:
                no_hand_counter += 1
                if no_hand_counter >= MAX_NO_HAND:
                    print("[Auto-reset] No hand detected for a while, resetting.")
                    hd = HandDetector()
                    hd.enable_guide_box(show_guide_box)
                    rec.reset_history()
                    if skeleton_tracker and SKELETON_AVAILABLE:
                        skeleton_tracker = SkeletonTracker(smoothing_frames=15, use_mediapipe=use_mediapipe)
                    no_hand_counter = 0
            else:
                no_hand_counter = 0

            # Process hand
            if hand:
                if "roi_mask" in hand:
                    quality = np.count_nonzero(hand["roi_mask"]) / hand["roi_mask"].size * 100

                    # Skeleton (CV or MediaPipe – handled inside tracker)
                    if skeleton_tracker and SKELETON_AVAILABLE:
                        skeleton = skeleton_tracker.extract_skeleton(
                            hand["roi_mask"], frame, hand["bbox"]
                        )

                        if skeleton:
                            hx0, hy0, hx1, hy1 = hand["bbox"]
                            hand_w = hx1 - hx0
                            hand_h = hy1 - hy0
                            draw_skeleton_overlay(frame, skeleton,
                                                  box_size=(hand_w, hand_h),
                                                  offset=(hx0, hy0))

                # Recognize gesture
                label, c, is_ready, debug_info = rec.predict(hand["roi"], hand["roi_mask"], skeleton)
                candidate, conf, ready = label, c, is_ready

                # Matching info for display
                if debug_info:
                    match_info = {
                        "best_match": debug_info.get("best_match", ""),
                        "match_count": debug_info.get("good_matches", 0),
                        "shape_similarity": debug_info.get("shape_score", 0.0),
                        "closeness_score": debug_info.get("closeness", 0.0),
                    }

                # Handle tokens with anti-spam (no "bbbbbb")
                if is_ready and label:
                    lab = label.upper()
                    if lab != last_typed_label:
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
                        last_typed_label = lab
                elif not is_ready:
                    last_typed_label = None

            # Build display frame
            disp = frame.copy()
            if show_guide_box:
                draw_hand_guide_box(disp, hand)
            if hand:
                cv2.drawContours(disp, [hand["contour"]], -1, (0, 255, 0), 2)

            # Reference thumbnail (closest symbol image)
            ref_thumb = None
            if candidate:
                lab = candidate.upper()
                ref_path = REFS / f"{lab}.png"
                if ref_path.exists():
                    if lab in ref_thumb_cache:
                        ref_thumb = ref_thumb_cache[lab]
                    else:
                        img = cv2.imread(str(ref_path))
                        if img is not None:
                            ref_thumb = cv2.resize(img, (120, 120))
                            ref_thumb_cache[lab] = ref_thumb

            status_info = {
                "status": "ACTIVE" if kbd.is_active() else "INACTIVE",
                "prediction": candidate,
                "confidence": conf,
                "ready": ready,
                "fps": current_fps,
                "show_help": show_help,
                "match_info": match_info,
                "ref_thumb": ref_thumb,
            }

            canvas = create_3panel_display(disp, hand, skeleton, quality, status_info)
            canvas = subs.draw(canvas)

            cv2.imshow("Gestura — Enhanced", canvas)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('g'):
                show_guide_box = not show_guide_box
                hd.enable_guide_box(show_guide_box)
                print(f"Guide box: {'ON' if show_guide_box else 'OFF'}")
            elif key == ord('a'):
                st = kbd.toggle_active()
                print(f"Typing {'ENABLED' if st else 'DISABLED'}")
            elif key == ord('k'):
                # Re-run calibration
                hd.enable_guide_box(True)
                while True:
                    ok2, fr2 = cap.read()
                    if not ok2:
                        break
                    fr2 = cv2.flip(fr2, 1)
                    hand_preview2 = hd.detect_hand(fr2)
                    draw_hand_guide_box(fr2, hand_preview2)
                    cv2.putText(fr2, "Calibration", (16, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
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
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    main()
