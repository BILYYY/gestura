import cv2
import numpy as np


class HandDetector:


    def __init__(self):
        # WIDER ranges - more tolerant
        self.lower_skin = np.array([0, 20, 50], dtype=np.uint8)  # Wider!
        self.upper_skin = np.array([30, 255, 255], dtype=np.uint8)  # Wider!

        self.min_area_frac = 0.006
        self.max_area_frac = 0.55
        self.kernel = np.ones((3, 3), np.uint8)

        self.guide_box_size = 280
        self.guide_box_enabled = True

        # NEW: Store if we've calibrated
        self.is_calibrated = False
        self.calibrated_ranges = None

    def set_skin_hsv(self, lower_hsv, upper_hsv):
        self.lower_skin = np.array(lower_hsv, dtype=np.uint8).clip(0, 255)
        self.upper_skin = np.array(upper_hsv, dtype=np.uint8).clip(0, 255)

    def enable_guide_box(self, enabled):
        self.guide_box_enabled = enabled

    def get_guide_box_coords(self, frame_width, frame_height):
        box_size = self.guide_box_size
        box_x = frame_width - box_size - 80
        box_y = (frame_height - box_size) // 2
        return box_x, box_y, box_x + box_size, box_y + box_size

    def calibrate_from_roi(self, roi_bgr, roi_mask, pad_h=15, pad_s=30, pad_v=40):
        """WIDER calibration tolerances for lighting changes"""
        if roi_bgr is None or roi_mask is None:
            return False
        if roi_bgr.size == 0 or roi_mask.size == 0:
            return False

        # Apply auto-adjust BEFORE calibrating (so we calibrate on adjusted image)
        roi_bgr = self._auto_adjust_brightness(roi_bgr)

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m = (roi_mask > 0)
        if np.count_nonzero(m) < 50:
            return False

        H = hsv[..., 0][m].astype(np.int32)
        S = hsv[..., 1][m].astype(np.int32)
        V = hsv[..., 2][m].astype(np.int32)

        h_lo, h_hi = np.percentile(H, [10, 90]).astype(int)  # Less extreme
        s_lo, s_hi = np.percentile(S, [10, 90]).astype(int)
        v_lo, v_hi = np.percentile(V, [10, 90]).astype(int)

        # MUCH WIDER padding for lighting tolerance
        lower = [max(0, h_lo - pad_h), max(0, s_lo - pad_s), max(0, v_lo - pad_v)]
        upper = [min(179, h_hi + pad_h), min(255, s_hi + pad_s), min(255, v_hi + pad_v)]

        # Store calibrated ranges
        self.calibrated_ranges = (lower, upper)
        self.is_calibrated = True

        self.set_skin_hsv(lower, upper)
        print(f"[Calibration] HSV range: {lower} to {upper}")
        return True

    def _auto_adjust_brightness(self, frame_bgr):
        """CONSISTENT auto-adjustment for all frames"""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Stronger adjustment for more lighting tolerance
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _create_box_mask(self, frame_shape):
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.guide_box_enabled:
            box_x1, box_y1, box_x2, box_y2 = self.get_guide_box_coords(w, h)
            mask[box_y1:box_y2, box_x1:box_x2] = 255
        else:
            mask[:, :] = 255

        return mask

    def _skin_mask(self, frame_bgr):
        # ALWAYS auto-adjust (consistent with calibration)
        frame_bgr = self._auto_adjust_brightness(frame_bgr)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Use calibrated ranges if available, otherwise use defaults
        if self.is_calibrated and self.calibrated_ranges:
            lower, upper = self.calibrated_ranges
            # Add even MORE tolerance at runtime
            lower_adjusted = [max(0, lower[0] - 5), max(0, lower[1] - 10), max(0, lower[2] - 15)]
            upper_adjusted = [min(179, upper[0] + 5), min(255, upper[1] + 10), min(255, upper[2] + 15)]
            mask = cv2.inRange(hsv, np.array(lower_adjusted, dtype=np.uint8),
                               np.array(upper_adjusted, dtype=np.uint8))
        else:
            # Uncalibrated - use very wide defaults
            mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Aggressive morphology for robustness
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Apply box mask
        box_mask = self._create_box_mask(frame_bgr.shape)
        mask = cv2.bitwise_and(mask, mask, mask=box_mask)

        return mask

    def _largest_valid_contour(self, mask, frame_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < self.min_area_frac * frame_area or area > self.max_area_frac * frame_area:
            return None, 0.0
        return c, area

    def _crop_with_pad(self, frame_bgr, mask, contour, pad=20):
        x, y, w, h = cv2.boundingRect(contour)
        H, W = frame_bgr.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        roi = frame_bgr[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1]
        return roi, roi_mask, (x0, y0, x1, y1)

    def detect_hand(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        mask = self._skin_mask(frame_bgr)
        contour, area = self._largest_valid_contour(mask, H * W)
        if contour is None:
            return None
        roi, roi_mask, bbox = self._crop_with_pad(frame_bgr, mask, contour)
        if roi.size == 0 or roi_mask.size == 0:
            return None
        return {
            "mask": mask, "contour": contour,
            "roi": roi, "roi_mask": roi_mask,
            "bbox": bbox, "area": area
        }