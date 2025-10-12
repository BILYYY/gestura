import cv2
import numpy as np


class HandDetector:
    """
    Handles hand detection using computer vision techniques
    """
    def __init__(self):
        # Skin color detection parameters (HSV)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Hand detection parameters
        self.min_hand_area = 5000
        self.max_hand_area = 100000


    def detect_skin_color(self, frame): # TODO: find better hand detection method
        """
        Detect hand using skin color segmentation in HSV color space
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create skin mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        # Clean up mask (remove noise)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Dilate to fill gaps
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask


    def find_largest_contour(self, mask):
        """
        Find the largest contour (hand) in the mask
        Returns contour and its area if valid
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0

        # Find largest contour
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        # Check if area is reasonable for a hand
        if area < self.min_hand_area or area > self.max_hand_area:
            return None, 0

        return max_contour, area


    def extract_roi(self, frame, contour):
        """
        Extract hand region of interest with padding
        Returns ROI image and bounding box coordinates
        """
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding
        padding = 20
        h_frame, w_frame = frame.shape[:2]

        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(w_frame, x + w + padding)
        y_max = min(h_frame, y + h + padding)

        roi = frame[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)

        return roi, bbox


    def detect_hand(self, frame):
        """
        Main detection method - returns hand data or None
        Returns: dict with 'mask', 'contour', 'roi', 'bbox', 'area' or None
        """
        mask = self.detect_skin_color(frame)

        # Find hand contour
        contour, area = self.find_largest_contour(mask)

        if contour is None:
            return None

        # Extract ROI
        roi, bbox = self.extract_roi(frame, contour)

        if roi is None or roi.size == 0:
            return None

        return {
            'mask': mask,
            'contour': contour,
            'roi': roi,
            'bbox': bbox,
            'area': area
        }

