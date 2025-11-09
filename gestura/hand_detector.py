import cv2
import numpy as np


class HandDetector:
    """
    Dual color-space (HSV ∩ YCrCb) skin segmentation + morphology.
    Relative area gating (robust to resolution). Returns:
    - full-frame mask, largest contour
    - cropped ROI + ROI mask, bbox
    """

    def __init__(self):
        # HSV bounds
        self.lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        self.upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

        # YCrCb bounds
        self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        # Relative area thresholds
        self.min_area_ratio = 0.010   # 1% of frame
        self.max_area_ratio = 0.600   # 60% of frame

        self.kernel = np.ones((3, 3), np.uint8)

    def _skin_mask(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

        m_hsv = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        m_ycc = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)
        mask = cv2.bitwise_and(m_hsv, m_ycc)

        # Morphology + threshold
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        mask = cv2.erode(mask, self.kernel, iterations=1)
        return mask

    @staticmethod
    def _largest_contour(mask):
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None, 0.0
        c = max(cs, key=cv2.contourArea)
        return c, float(cv2.contourArea(c))

    @staticmethod
    def _crop_with_pad(frame_bgr, mask, contour, pad=20):
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
        contour, area = self._largest_contour(mask)
        if contour is None:
            return None

        min_area = self.min_area_ratio * (H * W)
        max_area = self.max_area_ratio * (H * W)
        if area < min_area or area > max_area:
            return None

        roi, roi_mask, bbox = self._crop_with_pad(frame_bgr, mask, contour)
        if roi.size == 0:
            return None

        return {
            "mask": mask,
            "contour": contour,
            "area": area,
            "roi": roi,
            "roi_mask": roi_mask,
            "bbox": bbox
        }
