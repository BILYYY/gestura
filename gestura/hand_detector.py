import cv2
import numpy as np


class HandDetector:
    """
    HSV skin mask + morphology + largest-contour gating + ROI crop.
    Includes adaptive skin tuning + auto brightness/contrast adjustment.
    """

    def __init__(self):
        # Broad defaults; calibration will tighten these.
        self.lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        self.min_area_frac = 0.006  # ≥0.6% of frame
        self.max_area_frac = 0.55  # ≤55% of frame
        self.kernel = np.ones((3, 3), np.uint8)

    def set_skin_hsv(self, lower_hsv, upper_hsv):
        self.lower_skin = np.array(lower_hsv, dtype=np.uint8).clip(0, 255)
        self.upper_skin = np.array(upper_hsv, dtype=np.uint8).clip(0, 255)

    def calibrate_from_roi(self, roi_bgr, roi_mask, pad_h=10, pad_s=20, pad_v=25):
        """Tighten HSV thresholds from the current user's ROI."""
        if roi_bgr is None or roi_mask is None:
            return False
        if roi_bgr.size == 0 or roi_mask.size == 0:
            return False
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m = (roi_mask > 0)
        if np.count_nonzero(m) < 50:
            return False

        H = hsv[..., 0][m].astype(np.int32)
        S = hsv[..., 1][m].astype(np.int32)
        V = hsv[..., 2][m].astype(np.int32)
        h_lo, h_hi = np.percentile(H, [5, 95]).astype(int)
        s_lo, s_hi = np.percentile(S, [5, 95]).astype(int)
        v_lo, v_hi = np.percentile(V, [5, 95]).astype(int)

        lower = [max(0, h_lo - pad_h), max(0, s_lo - pad_s), max(0, v_lo - pad_v)]
        upper = [min(179, h_hi + pad_h), min(255, s_hi + pad_s), min(255, v_hi + pad_v)]
        self.set_skin_hsv(lower, upper)
        return True

    def _auto_adjust_brightness(self, frame_bgr):
        """Auto-adjust brightness and contrast using CLAHE in LAB color space"""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ---- internals ----
    def _skin_mask(self, frame_bgr):
        # Auto-adjust brightness first
        frame_bgr = self._auto_adjust_brightness(frame_bgr)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
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

    # ---- public ----
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