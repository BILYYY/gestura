import os
import cv2
import numpy as np
from collections import deque, Counter
import math
import json


def _clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _resize_pad(img, target=200):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target, target), dtype=img.dtype)
    s = float(target) / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((target, target), dtype=img.dtype)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def _pca_upright(gray, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return gray, mask
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = math.degrees(math.atan2(major[1], major[0]))
    rot = 90 - angle
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), rot, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(mask > 0)
    if len(xs) >= 10:
        x0, x1 = max(0, xs.min() - 2), min(w, xs.max() + 3)
        y0, y1 = max(0, ys.min() - 2), min(h, ys.max() + 3)
        gray = gray[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
    return gray, mask


class ORBSignRecognizer:
    def __init__(self, references_path):
        self.references_path = references_path

        self.orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
                                  firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                                  patchSize=31, fastThreshold=20)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ratio = 0.75

        self.prediction_buffer = deque(maxlen=30)
        self.confidence_buffer = deque(maxlen=30)
        self.geometric_buffer = deque(maxlen=30)
        self.orb_buffer = deque(maxlen=30)

        self.current_prediction = None
        self.current_confidence = 0.0
        self.frames_stable = 0
        self.min_stable_frames = 15

        self.refs = {}
        self._load_refs()

    def set_stability(self, window=None, freq=None, conf=None):
        if window is not None:
            new_len = int(window)
            self.prediction_buffer = deque(self.prediction_buffer, maxlen=new_len)
            self.confidence_buffer = deque(self.confidence_buffer, maxlen=new_len)

    def get_stability(self):
        return {"window": len(self.prediction_buffer), "freq": 0.51, "conf": 0.45}

    def reset_history(self):
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.geometric_buffer.clear()
        self.orb_buffer.clear()
        self.current_prediction = None
        self.current_confidence = 0.0
        self.frames_stable = 0

    def export_normalized_roi(self, roi_bgr, roi_mask=None):
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        m = roi_mask if roi_mask is not None else np.ones_like(gray, np.uint8) * 255
        g, m = self._normalize(gray, m)
        out = g.copy()
        out[m == 0] = 0
        return _resize_pad(out, 200)

    def _extract_geometric_features(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 10:
            return None

        features = {}

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features['area'] = area
        features['perimeter'] = perimeter
        features['compactness'] = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

        x, y, w, h = cv2.boundingRect(contour)
        features['aspect_ratio'] = float(w) / h if h > 0 else 0
        features['extent'] = area / (w * h) if (w * h) > 0 else 0

        try:
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments)
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            for i, hu in enumerate(hu_moments.flatten()):
                features[f'hu_{i}'] = hu
        except:
            for i in range(7):
                features[f'hu_{i}'] = 0.0

        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) > 3 and len(contour) > hull.max():
                try:
                    defects = cv2.convexityDefects(contour, hull)
                except cv2.error:
                    defects = None

                if defects is not None:
                    finger_count = 0
                    defect_depths = []

                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])

                        a = np.linalg.norm(np.array(start) - np.array(far))
                        b = np.linalg.norm(np.array(end) - np.array(far))
                        c = np.linalg.norm(np.array(start) - np.array(end))

                        if a > 0 and b > 0:
                            cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle = np.arccos(cos_angle)

                            if angle <= np.pi / 2 and d > 1000:
                                finger_count += 1
                                defect_depths.append(d)

                    features['finger_count'] = finger_count + 1 if finger_count > 0 else 0
                    features['avg_defect_depth'] = np.mean(defect_depths) if defect_depths else 0
                else:
                    features['finger_count'] = 0
                    features['avg_defect_depth'] = 0
            else:
                features['finger_count'] = 0
                features['avg_defect_depth'] = 0
        except Exception as e:
            features['finger_count'] = 0
            features['avg_defect_depth'] = 0

        try:
            hull_points = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull_points)
            features['solidity'] = area / hull_area if hull_area > 0 else 0
        except:
            features['solidity'] = 0.0

        try:
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            features['vertices'] = len(approx)
        except:
            features['vertices'] = 0

        return features

    def _compare_geometric_features(self, features1, features2):
        if features1 is None or features2 is None:
            return 0.0

        weights = {
            'finger_count': 4.0,
            'solidity': 2.5,
            'aspect_ratio': 2.0,
            'compactness': 1.5,
            'vertices': 1.5,
            'extent': 1.0,
        }

        for i in range(7):
            weights[f'hu_{i}'] = 0.6

        total_score = 0.0
        total_weight = 0.0

        for key in features1.keys():
            if key in features2 and key in weights:
                val1 = features1[key]
                val2 = features2[key]

                if key == 'finger_count':
                    diff = 0.0 if val1 == val2 else 1.0
                elif key.startswith('hu_'):
                    max_val = max(abs(val1), abs(val2), 1e-10)
                    diff = abs(val1 - val2) / max_val
                else:
                    diff = abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-10)

                similarity = 1.0 - min(diff, 1.0)
                total_score += similarity * weights[key]
                total_weight += weights[key]

        return total_score / total_weight if total_weight > 0 else 0.0

    def _normalize(self, gray, mask):
        gray = _clahe(gray)
        gray, mask = _pca_upright(gray, mask)
        gray = _resize_pad(gray, 200)
        mask = _resize_pad(mask, 200)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        return gray, mask

    def _orb_score(self, des_q, des_r):
        if des_q is None or des_r is None or len(des_q) < 2 or len(des_r) < 2:
            return 0.0
        try:
            matches = self.matcher.match(des_r, des_q)
        except cv2.error:
            return 0.0
        if not matches:
            return 0.0

        distances = [m.distance for m in matches]
        avg_distance = np.mean(distances)
        score = 1 / (avg_distance + 1e-6)
        confidence = min(score / 0.05, 1.0)
        return confidence

    def _load_refs(self):
        if not os.path.isdir(self.references_path):
            print(f"[Recognizer] references path not found: {self.references_path}")
            return

        exts = (".png", ".jpg", ".jpeg", ".bmp")
        loaded = 0

        for f in os.listdir(self.references_path):
            if os.path.splitext(f)[1].lower() not in exts:
                continue

            label = os.path.splitext(f)[0].upper()
            img = cv2.imread(os.path.join(self.references_path, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            m = (img > 0).astype(np.uint8) * 255 if np.count_nonzero(img) > 0 else \
                cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            geometric_features = self._extract_geometric_features(m)

            img, m = _pca_upright(img, m)
            img = _resize_pad(_clahe(img), 200)
            m = _resize_pad(m, 200)
            kps, des = self.orb.detectAndCompute(img, m)

            skeleton_data = None
            skeleton_path = os.path.join(self.references_path, f"{label}_skeleton.json")
            if os.path.exists(skeleton_path):
                try:
                    with open(skeleton_path, 'r', encoding='utf-8') as sf:
                        skeleton_data = json.load(sf)
                except:
                    pass

            if geometric_features and des is not None and len(kps) > 10:
                self.refs[label] = {
                    "tmpl": img,
                    "des": des,
                    "geometric": geometric_features,
                    "skeleton": skeleton_data
                }
                finger_count = geometric_features.get('finger_count', 0)
                skel_info = f", skeleton: {skeleton_data.get('extended_count', '?')}ext" if skeleton_data else ""
                print(f"[Recognizer] Loaded: {label} (fingers: {finger_count}, kps: {len(kps)}{skel_info})")
                loaded += 1
            else:
                print(f"[Recognizer] Skipped {label}: insufficient features")

        print(f"[Recognizer] Total loaded: {loaded} reference(s)")

    def predict(self, roi_bgr, roi_mask):
        if roi_bgr is None or roi_bgr.size == 0 or not self.refs:
            return self.current_prediction, self.current_confidence, False, {}

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mask = roi_mask if roi_mask is not None else np.ones_like(gray, np.uint8) * 255

        query_geometric = self._extract_geometric_features(mask)
        g, m = self._normalize(gray, mask)

        if g.size == 0 or np.count_nonzero(m) < 30:
            return self.current_prediction, self.current_confidence, False, {}

        kps, des = self.orb.detectAndCompute(g, m)

        geometric_scores = {}
        if query_geometric:
            for label, data in self.refs.items():
                score = self._compare_geometric_features(query_geometric, data["geometric"])
                geometric_scores[label] = score

        orb_scores = {}
        if des is not None and len(kps) >= 10:
            for label, data in self.refs.items():
                score = self._orb_score(des, data["des"])
                orb_scores[label] = score

        combined_scores = {}
        for label in self.refs.keys():
            geo_score = geometric_scores.get(label, 0.0)
            orb_score = orb_scores.get(label, 0.0)
            combined_scores[label] = 0.5 * geo_score + 0.5 * orb_score

        if combined_scores:
            best_label = max(combined_scores, key=combined_scores.get)
            best_score = combined_scores[best_label]
        else:
            best_label = None
            best_score = 0.0

        if best_score > 0.45:
            self.prediction_buffer.append(best_label)
            self.confidence_buffer.append(best_score)
            self.geometric_buffer.append(geometric_scores.get(best_label, 0.0))
            self.orb_buffer.append(orb_scores.get(best_label, 0.0))

        if len(self.prediction_buffer) >= 10:
            counts = Counter(self.prediction_buffer)
            most_common_label, most_common_count = counts.most_common(1)[0]
            stability = most_common_count / len(self.prediction_buffer)

            if stability >= 0.51:
                avg_confidence = np.mean([self.confidence_buffer[i] for i, pred in
                                          enumerate(self.prediction_buffer) if pred == most_common_label])

                if most_common_label == self.current_prediction:
                    self.frames_stable += 1
                else:
                    if avg_confidence > 0.45 or self.frames_stable >= self.min_stable_frames or self.current_prediction is None:
                        self.current_prediction = most_common_label
                        self.current_confidence = avg_confidence
                        self.frames_stable = 0
            else:
                self.frames_stable = max(0, self.frames_stable - 1)

        is_stable = self.frames_stable >= self.min_stable_frames

        return self.current_prediction, self.current_confidence, is_stable, {}