
import cv2
import numpy as np


def safe_crop(img, x0, y0, x1, y1):
    """
    Safely crop image with bounds checking

    Args:
        img: Input image
        x0, y0, x1, y1: Crop coordinates

    Returns:
        Cropped image or None if invalid
    """
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))

    if x1 <= x0 or y1 <= y0:
        return None

    return img[y0:y1, x0:x1]


def detect_hand_in_box(frame, box_coords, hand_detector):
    """
    Detect hand ONLY inside the fixed box

    Args:
        frame: Input frame
        box_coords: (x0, y0, x1, y1) box coordinates
        hand_detector: HandDetector instance

    Returns:
        hand dict or None
    """
    x0, y0, x1, y1 = box_coords

    # Crop to box region
    box_region = safe_crop(frame, x0, y0, x1, y1)
    if box_region is None:
        return None, None

    # Detect hand in cropped region
    hand = hand_detector.detect_hand(box_region)
    if hand is None:
        return None, box_region

    # Adjust bbox coordinates back to full frame
    bx0, by0, bx1, by1 = hand["bbox"]
    hand["bbox"] = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)

    # Adjust contour coordinates back to full frame
    adjusted_contour = hand["contour"].copy()
    adjusted_contour[:, 0, 0] += x0
    adjusted_contour[:, 0, 1] += y0
    hand["contour"] = adjusted_contour

    return hand, box_region


def calculate_mask_quality(mask):
    """
    Calculate mask quality score (0-100)

    Args:
        mask: Binary mask (grayscale)

    Returns:
        Quality score (0-100), higher is better
    """
    if mask is None or mask.size == 0:
        return 0.0

    # Calculate fill ratio
    total_pixels = mask.size
    white_pixels = np.count_nonzero(mask)
    fill_ratio = white_pixels / total_pixels

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    # Get largest contour (hand)
    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)

    # Calculate solidity (no holes = 1.0)
    solidity = (largest_area / white_pixels) if white_pixels > 0 else 0.0

    # Combined score: 30% fill + 70% solidity
    quality = (fill_ratio * 0.3 + solidity * 0.7) * 100

    return min(quality, 100.0)


def to_square(binary_img):
    """
    Pad image to square maintaining aspect ratio

    Args:
        binary_img: Binary image (grayscale)

    Returns:
        Square image with black padding
    """
    h, w = binary_img.shape[:2]
    side = max(h, w)

    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left

    return cv2.copyMakeBorder(
        binary_img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )


def normalize_to_silhouette(roi_bgr, roi_mask):
    """
    Convert ROI to 200x200 binary silhouette
    White hand on black background

    Args:
        roi_bgr: Color ROI image
        roi_mask: Binary mask

    Returns:
        200x200 binary silhouette or None
    """
    if roi_bgr is None or roi_mask is None:
        return None

    # Ensure mask is grayscale
    if len(roi_mask.shape) == 3:
        roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)

    # Resize mask to match ROI if needed
    if roi_bgr.shape[:2] != roi_mask.shape[:2]:
        roi_mask = cv2.resize(roi_mask, (roi_bgr.shape[1], roi_bgr.shape[0]))

    # Convert to grayscale
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Apply mask to isolate hand
    fg = cv2.bitwise_and(gray, gray, mask=roi_mask)

    # Threshold to binary
    _, bin_img = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure hand is white on black
    if (bin_img > 0).mean() < 0.5:
        bin_img = cv2.bitwise_not(bin_img)

    # Make square and resize to 200x200
    sq = to_square(bin_img)
    normalized = cv2.resize(sq, (200, 200), interpolation=cv2.INTER_AREA)

    return normalized


def draw_fixed_box(frame, box_coords, color):
    """
    Draw the fixed capture box with corner markers

    Args:
        frame: Frame to draw on (modified in-place)
        box_coords: (x0, y0, x1, y1)
        color: BGR color tuple
    """
    x0, y0, x1, y1 = box_coords
    thickness = 3

    # Main rectangle
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)

    # Corner markers
    corner_len = 20

    # Top-left
    cv2.line(frame, (x0, y0), (x0 + corner_len, y0), color, thickness + 1)
    cv2.line(frame, (x0, y0), (x0, y0 + corner_len), color, thickness + 1)

    # Top-right
    cv2.line(frame, (x1, y0), (x1 - corner_len, y0), color, thickness + 1)
    cv2.line(frame, (x1, y0), (x1, y0 + corner_len), color, thickness + 1)

    # Bottom-left
    cv2.line(frame, (x0, y1), (x0 + corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x0, y1), (x0, y1 - corner_len), color, thickness + 1)

    # Bottom-right
    cv2.line(frame, (x1, y1), (x1 - corner_len, y1), color, thickness + 1)
    cv2.line(frame, (x1, y1), (x1, y1 - corner_len), color, thickness + 1)


def get_quality_color_and_status(quality):
    """
    Get color and status text based on quality score

    Args:
        quality: Quality score (0-100)

    Returns:
        (color_bgr, status_text) tuple
    """
    if quality >= 80:
        return (0, 255, 0), "EXCELLENT"
    elif quality >= 70:
        return (0, 255, 255), "GOOD"
    elif quality >= 50:
        return (0, 165, 255), "FAIR"
    else:
        return (0, 0, 255), "POOR"


def get_box_color(hand, quality):
    """
    Determine box color based on hand detection and quality

    Args:
        hand: Hand dict or None
        quality: Quality score (0-100)

    Returns:
        BGR color tuple
    """
    if not hand:
        return (0, 0, 255)  # Red - no hand

    if quality >= 70:
        return (0, 255, 0)  # Green - good
    elif quality >= 50:
        return (0, 255, 255)  # Yellow - fair
    else:
        return (0, 165, 255)  # Orange - poor