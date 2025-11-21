import cv2
import numpy as np
from collections import deque


class SkeletonTracker:


    def __init__(self, smoothing_frames=15, use_mediapipe=False):
        self.smoothing_frames = smoothing_frames
        self.palm_history = deque(maxlen=smoothing_frames)
        self.fingertips_history = deque(maxlen=smoothing_frames)

        # Mode selection
        self.use_mediapipe = use_mediapipe
        self.mp_hands = None

        # Try to import MediaPipe if requested
        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✅ MediaPipe mode enabled")
            except ImportError:
                print("⚠️  MediaPipe not installed. Install with: pip install mediapipe")
                print("   Falling back to CV mode")
                self.use_mediapipe = False

    def extract_skeleton(self, mask, frame_bgr=None, bbox=None):
        """
        Extract hand skeleton from mask.

        Args:
            mask: Binary hand mask (required for both modes)
            frame_bgr: Full color frame (required for MediaPipe mode)
            bbox: Hand bounding box (x0, y0, x1, y1) in frame coordinates (required for MediaPipe)

        Returns:
            Skeleton dict or None
        """
        if self.use_mediapipe and self.mp_hands is not None and frame_bgr is not None and bbox is not None:
            # Try MediaPipe first
            skeleton = self._extract_mediapipe(frame_bgr, mask, bbox)
            if skeleton is not None:
                return skeleton
            # If MediaPipe fails, fall back to CV
            print("⚠️  MediaPipe failed, using CV fallback")

        # Use CV mode (your original algorithm)
        return self._extract_cv(mask)

    # ==================== CV MODE (YOUR ORIGINAL ALGORITHM) ====================

    def _extract_cv(self, mask):
        """Original CV-based skeleton extraction (teacher-approved)"""
        if mask is None or mask.size == 0:
            return None

        h, w = mask.shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 10:
            return None

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        palm_cx = M["m10"] / M["m00"]
        palm_cy = M["m01"] / M["m00"]

        palm_x_px = int(palm_cx)
        palm_y_px = int(palm_cy)
        if palm_x_px < 0 or palm_x_px >= w or palm_y_px < 0 or palm_y_px >= h:
            return None
        if mask[palm_y_px, palm_x_px] == 0:
            return None

        try:
            hull_points = cv2.convexHull(contour, returnPoints=True)

            fingertip_candidates = []
            for point in hull_points:
                px, py = point[0]
                if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
                    dist_to_palm = np.linalg.norm(np.array([px, py]) - np.array([palm_cx, palm_cy]))

                    if dist_to_palm > h * 0.25:
                        is_local_max = True
                        for other_point in hull_points:
                            ox, oy = other_point[0]
                            if (ox, oy) != (px, py):
                                dist_between = np.linalg.norm(np.array([px, py]) - np.array([ox, oy]))
                                if dist_between < 30:
                                    other_dist = np.linalg.norm(np.array([ox, oy]) - np.array([palm_cx, palm_cy]))
                                    if other_dist > dist_to_palm:
                                        is_local_max = False
                                        break

                        if is_local_max:
                            fingertip_candidates.append((px, py))
        except:
            fingertip_candidates = []

        if len(fingertip_candidates) < 3:
            all_white_points = []
            for pt in contour:
                x, y = pt[0]
                if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                    all_white_points.append((x, y))

            all_white_points.sort(key=lambda p: p[1])
            top_points = all_white_points[:min(30, len(all_white_points))]

            for px, py in top_points:
                dist = np.linalg.norm(np.array([px, py]) - np.array([palm_cx, palm_cy]))
                if dist > h * 0.25:
                    fingertip_candidates.append((px, py))

        fingertips = self._merge_close_fingertips(fingertip_candidates, min_distance=max(25, w // 12))

        fingertips.sort(key=lambda p: p[0])

        fingertips = fingertips[:5]

        fingertips_clipped = []
        for fx, fy in fingertips:
            fx_clip = max(0, min(w - 1, fx))
            fy_clip = max(0, min(h - 1, fy))
            if mask[fy_clip, fx_clip] > 0:
                fingertips_clipped.append((fx_clip, fy_clip))

        skeleton = {
            "palm_center": (palm_cx / w, palm_cy / h),
            "fingertips": [(fx / w, fy / h) for fx, fy in fingertips_clipped],
            "mask_size": (w, h),
            "mode": "cv"  # NEW: Track which mode was used
        }

        skeleton = self._add_temporal_smoothing(skeleton)
        skeleton = self._analyze_hand(skeleton, mask)

        return skeleton

    # ==================== MEDIAPIPE MODE (OPTIONAL ENHANCEMENT) ====================

    def _extract_mediapipe(self, frame_bgr, mask, bbox):
        """MediaPipe-based skeleton extraction (optional, more accurate)"""
        if self.mp_hands is None:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        # Bbox in full frame coordinates
        x0, y0, x1, y1 = bbox
        hand_w = x1 - x0
        hand_h = y1 - y0

        if hand_w <= 0 or hand_h <= 0:
            return None

        # Convert full frame to RGB for MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0]

        # MediaPipe landmarks are normalized (0-1) relative to FULL FRAME
        # We need to convert them to be relative to the HAND ROI for consistency

        # Palm center (landmark 9 = middle finger base)
        palm_lm = landmarks.landmark[9]
        palm_x_frame = palm_lm.x * frame_w  # Pixel in full frame
        palm_y_frame = palm_lm.y * frame_h

        # Convert to ROI-relative normalized coordinates
        palm_x_roi = (palm_x_frame - x0) / hand_w
        palm_y_roi = (palm_y_frame - y0) / hand_h

        # Fingertip landmarks: 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        fingertip_ids = [4, 8, 12, 16, 20]
        fingertips_roi = []

        for tip_id in fingertip_ids:
            lm = landmarks.landmark[tip_id]
            fx_frame = lm.x * frame_w  # Pixel in full frame
            fy_frame = lm.y * frame_h

            # Convert to ROI-relative normalized coordinates
            fx_roi = (fx_frame - x0) / hand_w
            fy_roi = (fy_frame - y0) / hand_h

            fingertips_roi.append((fx_roi, fy_roi))

        # Must have all 5 fingertips
        if len(fingertips_roi) < 5:
            return None

        skeleton = {
            "palm_center": (palm_x_roi, palm_y_roi),
            "fingertips": fingertips_roi,
            "mask_size": (hand_w, hand_h),  # Use hand ROI size for drawing
            "mode": "mediapipe"  # NEW: Track which mode was used
        }

        skeleton = self._add_temporal_smoothing(skeleton)
        skeleton = self._analyze_hand(skeleton, mask)

        return skeleton

    # ==================== SHARED HELPER METHODS ====================

    def _merge_close_fingertips(self, points, min_distance=20):
        """Merge fingertips that are too close together"""
        if not points:
            return []

        merged = []
        used = set()

        for i, p1 in enumerate(points):
            if i in used:
                continue

            cluster = [p1]
            for j, p2 in enumerate(points[i + 1:], start=i + 1):
                if j in used:
                    continue
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist < min_distance:
                    cluster.append(p2)
                    used.add(j)

            avg_x = int(np.mean([p[0] for p in cluster]))
            avg_y = int(np.mean([p[1] for p in cluster]))
            merged.append((avg_x, avg_y))

        return merged

    def _add_temporal_smoothing(self, skeleton):
        """Temporal smoothing for both modes"""
        palm_rel = skeleton["palm_center"]
        tips_rel = skeleton["fingertips"]

        self.palm_history.append(palm_rel)
        self.fingertips_history.append(tips_rel)

        if len(self.palm_history) >= 3:
            palm_smooth = (
                np.mean([p[0] for p in self.palm_history]),
                np.mean([p[1] for p in self.palm_history])
            )
            skeleton["palm_center"] = palm_smooth

        if len(self.fingertips_history) >= 3 and tips_rel:
            num_tips = len(tips_rel)
            smoothed_tips = []

            for tip_idx in range(num_tips):
                x_vals = []
                y_vals = []
                for frame_tips in self.fingertips_history:
                    if tip_idx < len(frame_tips):
                        x_vals.append(frame_tips[tip_idx][0])
                        y_vals.append(frame_tips[tip_idx][1])

                if x_vals and y_vals:
                    smoothed_tips.append((np.mean(x_vals), np.mean(y_vals)))

            if smoothed_tips:
                skeleton["fingertips"] = smoothed_tips

        return skeleton

    def _analyze_hand(self, skeleton, mask):
        """Analyze finger states for both modes"""
        palm = np.array(skeleton["palm_center"])
        fingertips = [np.array(f) for f in skeleton["fingertips"]]

        if len(fingertips) == 0:
            skeleton["hand_type"] = "unknown"
            skeleton["finger_states"] = []
            skeleton["extended_count"] = 0
            return skeleton

        distances = [np.linalg.norm(ft - palm) for ft in fingertips]
        avg_dist = np.mean(distances) if distances else 0

        finger_states = []
        extended_count = 0

        for i, (ft, dist) in enumerate(zip(fingertips, distances)):
            ratio = dist / avg_dist if avg_dist > 0 else 0

            if ratio > 1.15:
                state = "extended"
                extended_count += 1
            elif ratio > 0.85:
                state = "half-bent"
            else:
                state = "folded"

            finger_states.append({
                "index": i,
                "state": state,
                "distance_ratio": ratio
            })

        skeleton["finger_states"] = finger_states
        skeleton["extended_count"] = extended_count

        # Hand type detection (left/right) - only for CV mode
        if skeleton.get("mode") != "mediapipe":
            if len(fingertips) >= 2:
                sorted_by_x = sorted(enumerate(fingertips), key=lambda x: x[1][0])
                leftmost_idx, leftmost = sorted_by_x[0]
                rightmost_idx, rightmost = sorted_by_x[-1]

                thumb_candidate = leftmost if abs(leftmost[0] - palm[0]) > abs(rightmost[0] - palm[0]) else rightmost

                if thumb_candidate[0] < palm[0]:
                    skeleton["hand_type"] = "right"
                else:
                    skeleton["hand_type"] = "left"
            else:
                skeleton["hand_type"] = "unknown"
        else:
            # MediaPipe: don't use left/right (can be inaccurate with flipped camera)
            skeleton["hand_type"] = "unknown"

        return skeleton


# ==================== DRAWING & EXPORT (UNCHANGED) ====================

def draw_skeleton_overlay(frame, skeleton, box_size, offset=(0, 0)):
    """Draw skeleton overlay on frame"""
    if skeleton is None:
        return

    mask_w, mask_h = skeleton.get("mask_size", box_size)
    ox, oy = offset

    palm_x = int(skeleton["palm_center"][0] * mask_w) + ox
    palm_y = int(skeleton["palm_center"][1] * mask_h) + oy

    cv2.circle(frame, (palm_x, palm_y), 8, (0, 255, 255), -1)
    cv2.circle(frame, (palm_x, palm_y), 10, (255, 255, 255), 2)

    finger_states = skeleton.get("finger_states", [])

    for i, (fx_rel, fy_rel) in enumerate(skeleton["fingertips"]):
        fx = int(fx_rel * mask_w) + ox
        fy = int(fy_rel * mask_h) + oy

        if i < len(finger_states):
            state = finger_states[i]["state"]
            if state == "extended":
                color = (0, 255, 0)
            elif state == "half-bent":
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
        else:
            color = (255, 255, 0)

        cv2.line(frame, (palm_x, palm_y), (fx, fy), color, 2)
        cv2.circle(frame, (fx, fy), 6, (255, 0, 255), -1)
        cv2.circle(frame, (fx, fy), 8, (255, 255, 255), 2)

        label = f"{i + 1}"
        if i < len(finger_states):
            state_short = {"extended": "E", "half-bent": "H", "folded": "F"}[finger_states[i]["state"]]
            label = f"{i + 1}:{state_short}"

        cv2.putText(frame, label, (fx + 5, fy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def skeleton_to_json(skeleton):
    """Convert skeleton to JSON"""
    if skeleton is None:
        return None

    return {
        "palm_center": skeleton["palm_center"],
        "fingertips": skeleton["fingertips"],
        "hand_type": skeleton.get("hand_type", "unknown"),
        "extended_count": skeleton.get("extended_count", 0),
        "finger_states": skeleton.get("finger_states", []),
        "mode": skeleton.get("mode", "cv")  # NEW: Include mode in JSON
    }