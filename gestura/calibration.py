import time
import os
from pathlib import Path
import cv2
import numpy as np


class CalibrationWizard:
    """
    Enhanced calibration with SIMILARITY SCORES showing how close you are!
    """

    def __init__(self, cap, hand_detector, recognizer, resources_path):
        self.cap = cap
        self.hd = hand_detector
        self.rec = recognizer
        self.refs_path = Path(resources_path)

    def _draw_calibration_guide_box(self, frame, hand=None):
        """Draw guide box for calibration"""
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
                color = (0, 255, 0)  # Green
                text = "GOOD!"
                text_color = (0, 255, 0)
            else:
                color = (0, 165, 255)  # Orange
                text = "Move into box"
                text_color = (0, 165, 255)
        else:
            color = (0, 0, 255)  # Red
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
        icon_size = 50
        icon_x = box_x + (box_size - icon_size) // 2
        icon_y = box_y + (box_size - icon_size) // 2

        # Simple hand icon
        cv2.circle(frame, (icon_x + icon_size // 2, icon_y + icon_size // 2), icon_size // 3, color, 2)
        # Fingers
        for angle in [0, 30, 60, -30, -60]:
            rad = np.radians(angle)
            end_x = int(icon_x + icon_size // 2 + np.cos(rad) * icon_size // 2)
            end_y = int(icon_y + icon_size // 2 - np.sin(rad) * icon_size // 2)
            cv2.line(frame, (icon_x + icon_size // 2, icon_y + icon_size // 2), (end_x, end_y), color, 2)

        # Draw instruction text
        text_x = box_x + box_size // 2 - 60
        text_y = box_y + box_size + 35
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def _draw_similarity_meter(self, frame, target_letter, current_conf, all_scores):
        """Draw similarity meter showing how close you are to target"""
        h, w = frame.shape[:2]

        # Position on left side
        x_start = 12
        y_start = 200
        bar_width = 250
        bar_height = 25

        # Title
        cv2.putText(frame, f"Target: {target_letter}", (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        # Similarity bar for TARGET letter
        target_score = all_scores.get(target_letter, 0.0)
        cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height),
                      (60, 60, 60), -1)

        fill_width = int(bar_width * target_score)
        if target_score > 0.7:
            color = (0, 255, 0)  # Green - very close!
        elif target_score > 0.5:
            color = (0, 165, 255)  # Orange - getting there
        else:
            color = (0, 0, 255)  # Red - not close

        cv2.rectangle(frame, (x_start, y_start), (x_start + fill_width, y_start + bar_height),
                      color, -1)
        cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_start + bar_height),
                      (255, 255, 255), 2)

        # Percentage text
        cv2.putText(frame, f"{int(target_score * 100)}% match",
                    (x_start + bar_width + 10, y_start + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show top 3 closest letters
        y_offset = y_start + bar_height + 40
        cv2.putText(frame, "Closest letters:", (x_start, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 25

        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (letter, score) in enumerate(sorted_scores):
            color = (0, 255, 0) if letter == target_letter else (180, 180, 180)
            text = f"{i + 1}. {letter}: {score:.2f}"
            cv2.putText(frame, text, (x_start, y_offset + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2 if letter == target_letter else 1)

    # ---------- Simple ----------
    def run_simple(self, target_letter="A", conf_thresh=0.50):
        # Enable guide box for calibration
        self.hd.enable_guide_box(True)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                return {"passed": False, "skipped": True}
            frame = cv2.flip(frame, 1)

            self._banner(frame, f"Calibration: Show letter {target_letter}")
            hand = self.hd.detect_hand(frame)

            # Draw guide box
            self._draw_calibration_guide_box(frame, hand)

            passed = False
            all_scores = {}

            if hand:
                # Get prediction with scores for ALL letters
                letter, conf, _, _ = self.rec.predict(hand["roi"], hand["roi_mask"])

                # Get detailed scores (we need to expose this from recognizer)
                gray = cv2.cvtColor(hand["roi"], cv2.COLOR_BGR2GRAY)
                mask = hand["roi_mask"]

                # Extract features
                query_geometric = self.rec._extract_geometric_features(mask)

                if query_geometric:
                    # Calculate scores for ALL references
                    for label, data in self.rec.refs.items():
                        geo_score = self.rec._compare_geometric_features(query_geometric, data["geometric"])
                        all_scores[label] = geo_score

                # Draw similarity meter
                self._draw_similarity_meter(frame, target_letter, conf, all_scores)

                # Display current prediction
                cv2.putText(frame, f"Detected: {letter or '-'} ({conf:.2f})",
                            (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if letter == target_letter and conf >= conf_thresh:
                    passed = True

            # Environment hints
            env_msgs = self._env_msgs(frame, hand)
            self._hints_below_box(frame, env_msgs)

            if passed:
                cv2.putText(frame, "SUCCESS! Press SPACE to continue", (12, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Try again... (S=Skip, Q=Quit)", (12, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return {"passed": False, "skipped": True}
            if key == ord('s'):
                return {"passed": False, "skipped": True}
            if passed and key == 32:  # SPACE
                if hand:
                    self.hd.calibrate_from_roi(hand["roi"], hand["roi_mask"])
                return {"passed": True, "skipped": False}

    # ---------- Advanced ----------
    def run_advanced(self, letters=("A", "E", "O"), consec=3, base_conf=0.50):
        # Enable guide box for calibration
        self.hd.enable_guide_box(True)

        per_letter_conf = []
        for L in letters:
            streak, conf_sum = 0, 0.0
            t0 = time.time()

            ref = self._load_ref(L)
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    return {"passed": False, "skipped": True}
                frame = cv2.flip(frame, 1)

                self._banner(frame, f"Calibration (Advanced): Show '{L}'")

                # Show reference in top-left instead of overlapping with box
                if ref is not None:
                    self._show_ref_topleft(frame, ref, label=L)

                hand = self.hd.detect_hand(frame)

                # Draw guide box
                self._draw_calibration_guide_box(frame, hand)

                all_scores = {}

                if hand:
                    letter, conf, _, _ = self.rec.predict(hand["roi"], hand["roi_mask"])

                    # Get detailed scores
                    gray = cv2.cvtColor(hand["roi"], cv2.COLOR_BGR2GRAY)
                    mask = hand["roi_mask"]
                    query_geometric = self.rec._extract_geometric_features(mask)

                    if query_geometric:
                        for label, data in self.rec.refs.items():
                            geo_score = self.rec._compare_geometric_features(query_geometric, data["geometric"])
                            all_scores[label] = geo_score

                    # Draw similarity meter
                    self._draw_similarity_meter(frame, L, conf, all_scores)

                    cv2.putText(frame, f"Detected: {letter or '-'} ({conf:.2f})",
                                (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    if letter == L and conf >= base_conf:
                        streak += 1
                        conf_sum += conf
                        self.hd.calibrate_from_roi(hand["roi"], hand["roi_mask"])
                    else:
                        streak, conf_sum = 0, 0.0

                # Environment hints below box
                env_msgs = self._env_msgs(frame, hand)
                self._hints_below_box(frame, env_msgs)

                cv2.putText(frame, f"{streak}/{consec} correct   S=Skip  R=Retry  Q=Quit",
                            (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
                cv2.imshow("Calibration", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return {"passed": False, "skipped": True}
                if key == ord('s'):
                    return {"passed": False, "skipped": True}
                if key == ord('r'):
                    streak, conf_sum = 0, 0.0

                if time.time() - t0 > 60:
                    return {"passed": False, "skipped": False, "reason": "timeout"}

                if streak >= consec:
                    per_letter_conf.append(conf_sum / consec)
                    break

        # Tune stability from observed mean confidence
        if per_letter_conf:
            avg_conf = float(np.mean(per_letter_conf))
        else:
            avg_conf = 0.65

        new_window = 12 if avg_conf < 0.72 else 10
        new_conf = max(0.45, min(0.85, avg_conf - 0.05))
        self.rec.set_stability(window=new_window, conf=new_conf)
        return {"passed": True, "skipped": False, "avg_conf": avg_conf, "stability": self.rec.get_stability()}

    # ---------- Summary screen ----------
    def show_summary(self, res):
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        title = "Calibration complete!" if res.get("passed") else "Calibration skipped/failed"
        color = (0, 255, 0) if res.get("passed") else (0, 0, 255)
        cv2.putText(frame, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if res.get("passed"):
            if "avg_conf" in res:
                cv2.putText(frame, f"Avg confidence: {res['avg_conf'] * 100:.1f}%",
                            (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 255, 220), 2)
            st = res.get("stability", {})
            cv2.putText(frame, f"Window={st.get('window')}  freq>=0.51  conf>=0.45",
                        (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 2)
        cv2.putText(frame, "Press any key...", (16, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(0)
        try:
            cv2.destroyWindow("Calibration")
        except:
            pass

    # ---------- helpers ----------
    def _banner(self, frame, text):
        cv2.putText(frame, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

    def _load_ref(self, letter):
        p = self.refs_path / f"{letter}.png"
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (100, 100)) if img is not None else None

    def _show_ref_topleft(self, frame, ref_img, label):
        """Show reference in top-left corner to avoid box overlap"""
        y0 = 70
        x0 = 10
        ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        h, w = ref_img.shape[:2]
        frame[y0:y0 + h, x0:x0 + w] = ref_bgr
        cv2.putText(frame, f"REF: {label}", (x0, y0 + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _env_msgs(self, frame_bgr, hand):
        msgs = []
        H, W = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[..., 2].astype(np.float32)
        v_mean = float(np.mean(V))
        v_std = float(np.std(V))

        if v_mean < 70:
            msgs.append("Lighting too dark")
        elif v_mean > 235:
            msgs.append("Lighting too bright")
        if v_std > 80:
            msgs.append("Uneven lighting")
        if W < 640 or H < 480:
            msgs.append(f"Low resolution ({W}x{H})")

        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0), 50, 150)
        mask_out = np.ones_like(edges, np.uint8) * 255
        if hand:
            x0, y0, x1, y1 = hand['bbox']
            mask_out[y0:y1, x0:x1] = 0
        edges_bg = cv2.bitwise_and(edges, edges, mask=mask_out)
        dens = float(np.count_nonzero(edges_bg)) / float(edges_bg.size)
        if dens > 0.05:
            msgs.append("Background cluttered")

        if hand:
            ar = hand['area'] / float(H * W)
            if ar < 0.01:
                msgs.append("Hand too small")
            elif ar > 0.35:
                msgs.append("Hand too close")
        return msgs

    def _hints_below_box(self, frame, msgs):
        """Display hints below the guide box area"""
        h, w = frame.shape[:2]
        # Position hints in bottom-left area
        y = h - 100
        for m in msgs[:3]:
            cv2.putText(frame, m, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2)
            y += 25