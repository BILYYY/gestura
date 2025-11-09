import time
import os
from pathlib import Path
import cv2
import numpy as np


class CalibrationWizard:
    """
    Simple & Advanced calibration:
      - Env checks (lighting, background clutter, resolution, hand size)
      - Auto-tune skin HSV from ROI
      - Tune recognizer stability (window / conf) from observed confidence
    """

    def __init__(self, cap, hand_detector, recognizer, resources_path):
        self.cap = cap
        self.hd = hand_detector
        self.rec = recognizer
        self.refs_path = Path(resources_path)

    # ---------- Simple ----------
    def run_simple(self, target_letter="A", conf_thresh=0.70):
        while True:
            ok, frame = self.cap.read()
            if not ok: return {"passed": False, "skipped": True}
            frame = cv2.flip(frame, 1)

            self._banner(frame, f"Calibration: Show letter {target_letter}")
            hand = self.hd.detect_hand(frame)
            self._hints(frame, self._env_msgs(frame, hand))

            passed = False
            if hand:
                letter, conf, _, _ = self.rec.predict(hand["roi"], hand["roi_mask"])
                cv2.putText(frame, f"Pred: {letter or '-'} ({conf:.2f})",
                            (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if letter == target_letter and conf >= conf_thresh:
                    passed = True

            if passed:
                cv2.putText(frame, "Success! Press SPACE to continue", (12, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Try again… (S=Skip, Q=Quit)", (12, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): return {"passed": False, "skipped": True}
            if key == ord('s'): return {"passed": False, "skipped": True}
            if passed and key == 32:  # SPACE
                if hand: self.hd.calibrate_from_roi(hand["roi"], hand["roi_mask"])
                return {"passed": True, "skipped": False}

    # ---------- Advanced ----------
    def run_advanced(self, letters=("A", "E", "O"), consec=3, base_conf=0.70):
        per_letter_conf = []
        for L in letters:
            streak, conf_sum = 0, 0.0
            t0 = time.time()

            ref = self._load_ref(L)  # tiny preview if available
            while True:
                ok, frame = self.cap.read()
                if not ok: return {"passed": False, "skipped": True}
                frame = cv2.flip(frame, 1)

                self._banner(frame, f"Calibration (Advanced): Show '{L}'")
                if ref is not None:
                    self._show_ref(frame, ref, label=L)

                hand = self.hd.detect_hand(frame)
                self._hints(frame, self._env_msgs(frame, hand))

                if hand:
                    letter, conf, _, _ = self.rec.predict(hand["roi"], hand["roi_mask"])
                    cv2.putText(frame, f"Pred: {letter or '-'} ({conf:.2f})",
                                (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if letter == L and conf >= base_conf:
                        streak += 1
                        conf_sum += conf
                        self.hd.calibrate_from_roi(hand["roi"], hand["roi_mask"])
                    else:
                        streak, conf_sum = 0, 0.0

                cv2.putText(frame, f"{streak}/{consec} correct   S=Skip  R=Retry  Q=Quit",
                            (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 180), 2)
                cv2.imshow("Calibration", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): return {"passed": False, "skipped": True}
                if key == ord('s'): return {"passed": False, "skipped": True}
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
        new_conf = max(0.55, min(0.85, avg_conf - 0.05))
        self.rec.set_stability(window=new_window, conf=new_conf)
        return {"passed": True, "skipped": False, "avg_conf": avg_conf, "stability": self.rec.get_stability()}

    # ---------- Summary screen ----------
    def show_summary(self, res):
        ok, frame = self.cap.read()
        if not ok: return
        frame = cv2.flip(frame, 1)
        title = "Calibration complete!" if res.get("passed") else "Calibration skipped/failed"
        color = (0, 255, 0) if res.get("passed") else (0, 0, 255)
        cv2.putText(frame, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if res.get("passed"):
            if "avg_conf" in res:
                cv2.putText(frame, f"Avg confidence: {res['avg_conf']*100:.1f}%",
                            (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 255, 220), 2)
            st = res.get("stability", {})
            cv2.putText(frame, f"Window={st.get('window')}  freq>={st.get('freq',0):.2f}  conf>={st.get('conf',0):.2f}",
                        (16, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 2)
        cv2.putText(frame, "Press any key…", (16, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(0)
        try: cv2.destroyWindow("Calibration")
        except: pass

    # ---------- helpers ----------
    def _banner(self, frame, text):
        cv2.putText(frame, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

    def _load_ref(self, letter):
        p = self.refs_path / f"{letter}.png"
        if not p.exists(): return None
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, (120, 120)) if img is not None else None

    def _show_ref(self, frame, ref_img, label):
        y0 = 10; x0 = frame.shape[1] - 10 - ref_img.shape[1]
        ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        frame[y0:y0 + ref_img.shape[0], x0:x0 + ref_img.shape[1]] = ref_bgr
        cv2.putText(frame, f"REF {label}", (x0, y0 + ref_img.shape[0] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _env_msgs(self, frame_bgr, hand):
        msgs = []
        H, W = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[..., 2].astype(np.float32)
        v_mean = float(np.mean(V)); v_std = float(np.std(V))
        if v_mean < 70: msgs.append("Lighting too dark - increase brightness")
        elif v_mean > 235: msgs.append("Lighting too bright - reduce exposure")
        if v_std > 80: msgs.append("Uneven lighting - avoid harsh shadows")
        if W < 640 or H < 480: msgs.append(f"Low camera resolution ({W}x{H})")

        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3,3), 0), 50, 150)
        mask_out = np.ones_like(edges, np.uint8) * 255
        if hand:
            x0, y0, x1, y1 = hand['bbox']; mask_out[y0:y1, x0:x1] = 0
        edges_bg = cv2.bitwise_and(edges, edges, mask=mask_out)
        dens = float(np.count_nonzero(edges_bg)) / float(edges_bg.size)
        if dens > 0.05: msgs.append("Background too cluttered - use a plain wall")

        if hand:
            ar = hand['area'] / float(H*W)
            if ar < 0.01: msgs.append("Hand too small - move closer")
            elif ar > 0.35: msgs.append("Hand too close - move back")
        return msgs

    def _hints(self, frame, msgs):
        y = 72
        for m in msgs[:3]:
            cv2.putText(frame, m, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)
            y += 24
