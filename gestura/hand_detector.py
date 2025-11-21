import cv2
import numpy as np
from collections import deque


class HandDetector:
    def __init__(self):
        # ---- Base HSV skin range (used before calibration) ----
        self.lower_skin = np.array([0, 20, 50], dtype=np.uint8)
        self.upper_skin = np.array([30, 255, 255], dtype=np.uint8)

        # ---- Calibration padding around percentiles (tunable) ----
        self.pad_h = 15
        self.pad_s = 30
        self.pad_v = 40

        # ---- Morphology (tunable) ----
        self.kernel_size = 3          # base kernel size (will be made odd)
        self.open_iter = 3            # remove noise
        self.close_iter = 3           # fill holes
        self.blur_ksize = 7           # Gaussian blur (must be odd)

        # ---- Area filtering (fraction of full frame) ----
        self.min_area_frac = 0.006
        self.max_area_frac = 0.55

        # ---- Guide box ----
        self.guide_box_size = 280
        self.guide_box_enabled = True

        # ---- Calibration state ----
        self.is_calibrated = False
        self.calibrated_ranges = None

        # ---- Morphology kernel (ellipse is better for hands) ----
        k = self._make_odd(self.kernel_size)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # ---- Temporal smoothing for mask ----
        # maxlen=3 means we average last 3 frames → less flicker
        self.mask_history = deque(maxlen=3)

    # ===================== PUBLIC TUNING METHODS =====================

    def set_skin_hsv(self, lower_hsv, upper_hsv):
        """Directly set global HSV skin range (used when not calibrated)."""
        self.lower_skin = np.array(lower_hsv, dtype=np.uint8).clip(0, 255)
        self.upper_skin = np.array(upper_hsv, dtype=np.uint8).clip(0, 255)

    def set_calibration_padding(self, pad_h=None, pad_s=None, pad_v=None):
        """Change how 'loose' calibration is around measured HSV percentiles."""
        if pad_h is not None:
            self.pad_h = int(pad_h)
        if pad_s is not None:
            self.pad_s = int(pad_s)
        if pad_v is not None:
            self.pad_v = int(pad_v)

    def set_morphology(self, kernel_size=None, open_iter=None, close_iter=None, blur_ksize=None):
        """Adjust morphology strength (useful later for sliders)."""
        if kernel_size is not None and kernel_size > 0:
            kernel_size = int(kernel_size)
            k = self._make_odd(kernel_size)
            self.kernel_size = k
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if open_iter is not None:
            self.open_iter = int(max(0, open_iter))
        if close_iter is not None:
            self.close_iter = int(max(0, close_iter))
        if blur_ksize is not None:
            blur_ksize = int(blur_ksize)
            self.blur_ksize = self._make_odd(max(1, blur_ksize))

    def set_area_fraction(self, min_area=None, max_area=None):
        """Tweak minimal/maximal hand size as fraction of full frame."""
        if min_area is not None:
            self.min_area_frac = float(min_area)
        if max_area is not None:
            self.max_area_frac = float(max_area)

    def enable_guide_box(self, enabled):
        self.guide_box_enabled = bool(enabled)

    def get_guide_box_coords(self, frame_width, frame_height):
        box_size = self.guide_box_size
        box_x = frame_width - box_size - 80
        box_y = (frame_height - box_size) // 2
        return box_x, box_y, box_x + box_size, box_y + box_size

    # ===================== CALIBRATION =====================

    def calibrate_from_roi(self, roi_bgr, roi_mask, pad_h=None, pad_s=None, pad_v=None):
        """
        Calibrate skin HSV range from ROI (your hand inside guide box).
        pad_* override the default padding only for this call if given.
        """
        if roi_bgr is None or roi_mask is None:
            return False
        if roi_bgr.size == 0 or roi_mask.size == 0:
            return False

        roi_bgr = self._auto_adjust_brightness(roi_bgr)

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        m = (roi_mask > 0)
        if np.count_nonzero(m) < 50:
            return False

        H = hsv[..., 0][m].astype(np.int32)
        S = hsv[..., 1][m].astype(np.int32)
        V = hsv[..., 2][m].astype(np.int32)

        h_lo, h_hi = np.percentile(H, [10, 90]).astype(int)
        s_lo, s_hi = np.percentile(S, [10, 90]).astype(int)
        v_lo, v_hi = np.percentile(V, [10, 90]).astype(int)

        # Use provided pads or stored ones
        pad_h = self.pad_h if pad_h is None else int(pad_h)
        pad_s = self.pad_s if pad_s is None else int(pad_s)
        pad_v = self.pad_v if pad_v is None else int(pad_v)

        lower = [max(0,  h_lo - pad_h), max(0,   s_lo - pad_s), max(0,   v_lo - pad_v)]
        upper = [min(179, h_hi + pad_h), min(255, s_hi + pad_s), min(255, v_hi + pad_v)]

        self.calibrated_ranges = (lower, upper)
        self.is_calibrated = True

        # Also update base HSV so other code can read it
        self.set_skin_hsv(lower, upper)
        print(f"[HandDetector] Calibrated HSV: {lower} to {upper}")
        return True

    # ===================== INTERNAL HELPERS =====================

    @staticmethod
    def _make_odd(k):
        """Ensure kernel size is odd (required by GaussianBlur)."""
        k = int(k)
        return k if k % 2 == 1 else k + 1

    def _auto_adjust_brightness(self, frame_bgr):
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

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

    # ===================== MASK CREATION =====================

    def _skin_mask(self, frame_bgr):
        # Lighting normalization
        frame_bgr = self._auto_adjust_brightness(frame_bgr)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Use calibrated HSV range if available, else default
        if self.is_calibrated and self.calibrated_ranges:
            lower, upper = self.calibrated_ranges
            lower_arr = np.array(lower, dtype=np.uint8)
            upper_arr = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_arr, upper_arr)
        else:
            mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Morphology: open → close
        if self.open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=self.open_iter)
        if self.close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iter)

        # Smooth edges
        if self.blur_ksize > 1:
            k = self._make_odd(self.blur_ksize)
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        # Binarize again (in case blur softened it)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Limit to guide box region
        box_mask = self._create_box_mask(frame_bgr.shape)
        mask = cv2.bitwise_and(mask, mask, mask=box_mask)

        # ---- Temporal smoothing over last few masks ----
        self.mask_history.append(mask)
        if len(self.mask_history) > 1:
            avg = np.mean(self.mask_history, axis=0)
            mask = np.where(avg >= 127, 255, 0).astype(np.uint8)

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

    # ===================== MAIN API =====================

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
            "mask": mask,
            "contour": contour,
            "roi": roi,
            "roi_mask": roi_mask,
            "bbox": bbox,
            "area": area
        }
