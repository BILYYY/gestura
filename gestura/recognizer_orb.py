import os
import cv2
import numpy as np
from collections import deque
import math


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
    y0 = (target - nh) // 2; x0 = (target - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = r
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
    M = cv2.getRotationMatrix2D((w//2, h//2), rot, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(mask > 0)
    if len(xs) >= 10:
        x0, x1 = max(0, xs.min()-2), min(w, xs.max()+3)
        y0, y1 = max(0, ys.min()-2), min(h, ys.max()+3)
        gray = gray[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
    return gray, mask


class ORBSignRecognizer:
    """
    ORB (primary) + template correlation + edge/corner similarity + temporal stability.
    Loads every image in references_path as a label (filename stem).
    """

    def __init__(self, references_path):
        self.references_path = references_path

        # ORB (no contrib)
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
                                  firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                                  patchSize=31, fastThreshold=20)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio = 0.75

        # Stability (auto-tuned via calibration)
        self.history = deque(maxlen=10)
        self.stability_freq = 0.70
        self.stability_conf = 0.60

        # refs: label -> features
        self.refs = {}
        self._load_refs()

    # ----- calibration tuning -----
    def set_stability(self, window=None, freq=None, conf=None):
        if window is not None and window != self.history.maxlen:
            self.history = deque(self.history, maxlen=int(window))
        if freq is not None:
            self.stability_freq = float(freq)
        if conf is not None:
            self.stability_conf = float(conf)

    def get_stability(self):
        return {"window": self.history.maxlen, "freq": self.stability_freq, "conf": self.stability_conf}

    # ----- public -----
    def reset_history(self):
        self.history.clear()

    def export_normalized_roi(self, roi_bgr, roi_mask=None):
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        m = roi_mask if roi_mask is not None else np.ones_like(gray, np.uint8) * 255
        g, m = self._normalize(gray, m)
        out = g.copy(); out[m == 0] = 0
        return _resize_pad(out, 200)

    def predict(self, roi_bgr, roi_mask):
        if roi_bgr is None or roi_bgr.size == 0 or not self.refs:
            self._push(None, 0.0); return None, 0.0, False, {}
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mask = roi_mask if roi_mask is not None else np.ones_like(gray, np.uint8) * 255
        g, m = self._normalize(gray, mask)
        if g.size == 0 or np.count_nonzero(m) < 30:
            self._push(None, 0.0); return None, 0.0, False, {}

        # Features
        kps, des = self.orb.detectAndCompute(g, m)

        # Scores against all references
        best_label, best_score = None, -1.0
        second_score = -1.0

        for label, R in self.refs.items():
            s_orb = self._orb_score(des, R['des'])
            s_tm = self._template_score(g, R['tmpl'])
            s_edge = self._count_similarity(self._edge_count(g, m), R['edge_count'])
            s_corner = self._count_similarity(self._corner_count(g, m), R['corner_count'])
            combined = 0.40*s_orb + 0.25*s_tm + 0.20*s_edge + 0.15*s_corner

            if combined > best_score:
                second_score = best_score
                best_score = combined
                best_label = label
            elif combined > second_score:
                second_score = combined

        margin = max(0.0, best_score - max(0.0, second_score))
        conf = float(np.clip(0.75*best_score + 0.25*min(1.0, margin/0.2), 0.0, 1.0))

        self._push(best_label, conf)
        ready = self._stable(best_label)
        return best_label, conf, ready, {}

    # ----- refs -----
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
            # normalize ref
            m = (img > 0).astype(np.uint8) * 255 if np.count_nonzero(img) > 0 else \
                cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img, m = _pca_upright(img, m)
            img = _resize_pad(_clahe(img), 200); m = _resize_pad(m, 200)
            kps, des = self.orb.detectAndCompute(img, m)
            R = {
                "tmpl": img,
                "des": des,
                "edge_count": self._edge_count(img, m),
                "corner_count": self._corner_count(img, m),
            }
            self.refs[label] = R
            loaded += 1
        print(f"[Recognizer] loaded {loaded} reference(s) from {self.references_path}")

    # ----- internals -----
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
            matches = self.matcher.knnMatch(des_q, des_r, k=2)
        except cv2.error:
            return 0.0
        good = 0
        for pair in matches:
            if len(pair) != 2: continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good += 1
        # cap/normalize
        return min(good / 50.0, 1.0)

    @staticmethod
    def _template_score(g1, g2):
        if g1.shape != g2.shape:
            return 0.0
        r = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)
        v = float(r[0, 0])
        return max(0.0, (v + 1.0) / 2.0)

    @staticmethod
    def _edge_count(gray, mask):
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        return int(np.sum(edges > 0))

    @staticmethod
    def _corner_count(gray, mask):
        if np.count_nonzero(mask) == 0:
            return 0
        g32 = np.float32(gray)
        dst = cv2.cornerHarris(g32, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        return int(np.sum(dst > 0.01 * max(dst.max(), 1e-6)))

    def _push(self, letter, conf):
        self.history.append((letter, float(conf)))

    def _stable(self, letter):
        if len(self.history) < self.history.maxlen:
            return False
        counts = {}
        conf_sum = {}
        for ltr, c in self.history:
            if ltr is None: continue
            counts[ltr] = counts.get(ltr, 0) + 1
            conf_sum[ltr] = conf_sum.get(ltr, 0.0) + c
        if letter not in counts: return False
        freq = counts[letter] / float(len(self.history))
        mean_conf = conf_sum[letter] / max(1, counts[letter])
        return (freq >= self.stability_freq) and (mean_conf >= self.stability_conf)
