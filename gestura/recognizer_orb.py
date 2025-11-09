import os
import cv2
import numpy as np
from collections import deque


class ORBSignRecognizer:
    """
    Final program recognizer:
      - Mask-aware ROI normalization (PCA upright, resize to 200, CLAHE)
      - ORB keypoints + BFMatcher ratio test (primary signal)
      - Template correlation (aux)
      - Edge/corner similarity (aux)
      - Temporal stability: N=10, freq>=0.70, conf>=0.60
    """

    def __init__(self, references_path="resources/references",
                 history_size=10, stability_freq=0.70, stability_conf=0.60):
        self.ref_path = references_path
        self.history = deque(maxlen=history_size)
        self.conf_hist = deque(maxlen=history_size)
        self.history_size = history_size
        self.stability_freq = stability_freq
        self.stability_conf = stability_conf

        # ORB + BFMatcher (no contrib)
        self.orb = cv2.ORB_create(
            nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15,
            firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31, fastThreshold=20
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.norm_size = 200
        # Fusion weights
        self.w_orb = 0.40
        self.w_tm = 0.25
        self.w_edge = 0.20
        self.w_corner = 0.15

        # Preload references
        self.refs = {}  # label -> features
        self.labels = []
        self._load_refs()

    # -------- public --------
    def reset_history(self):
        self.history.clear()
        self.conf_hist.clear()

    def predict(self, roi_bgr, roi_mask):
        if roi_bgr is None or roi_bgr.size == 0 or roi_mask is None or roi_mask.size == 0:
            self._push(None, 0.0)
            return None, 0.0, False, {}

        gray, mask = self._normalize_roi(roi_bgr, roi_mask, self.norm_size)
        feats = self._extract_feats(gray, mask)
        letter, conf, _ = self._classify(feats)

        self._push(letter, conf)
        stable_letter, stable_conf = self._stable()

        if stable_letter is not None:
            return stable_letter, stable_conf, True, {}
        return letter, conf, False, {}

    # -------- references --------
    def _load_refs(self):
        if not os.path.isdir(self.ref_path):
            print(f"[Recognizer] references path not found: {self.ref_path}")
            return
        loaded = 0
        for f in os.listdir(self.ref_path):
            if os.path.splitext(f)[1].lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            label = os.path.splitext(f)[0].upper()
            img = cv2.imread(os.path.join(self.ref_path, f), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # Basic mask via Otsu
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (3, 3), 0)
            _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

            gn, mn = self._normalize_roi(img, m, self.norm_size)
            feats = self._extract_feats(gn, mn)
            self.refs[label] = feats
            self.labels.append(label)
            loaded += 1
        print(f"[Recognizer] loaded {loaded} reference(s) from {self.ref_path}")

    # -------- normalization --------
    def _normalize_roi(self, roi_bgr, roi_mask, out_size):
        mask = (roi_mask > 0).astype(np.uint8) * 255
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            if len(c) >= 5:
                pts = c.reshape(-1, 2).astype(np.float32)
                mean = np.mean(pts, axis=0)
                pts_c = pts - mean
                cov = np.cov(pts_c.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                major = eigvecs[:, np.argmax(eigvals)]
                angle = np.degrees(np.arctan2(major[1], major[0]))
                rot = 90 - angle
                M = cv2.getRotationMatrix2D(tuple(mean), rot, 1.0)
                H, W = gray.shape[:2]
                gray = cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
                mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

        ys, xs = np.where(mask > 0)
        if len(xs) and len(ys):
            x0, x1 = max(0, xs.min() - 10), min(mask.shape[1], xs.max() + 11)
            y0, y1 = max(0, ys.min() - 10), min(mask.shape[0], ys.max() + 11)
            gray = gray[y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]

        gray = self._resize_square(gray, out_size)
        mask = self._resize_square(mask, out_size, value=0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray, mask

    @staticmethod
    def _resize_square(img, size, value=0):
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((size, size), np.uint8)
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size), r.dtype) + value
        y0 = (size - nh) // 2
        x0 = (size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = r
        return canvas

    # -------- features --------
    def _extract_feats(self, gray, mask):
        kps, des = self.orb.detectAndCompute(gray, mask)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edge_count = int(np.sum(edges > 0))

        g32 = np.float32(gray)
        harris = cv2.cornerHarris(g32, 2, 3, 0.04)
        harris = cv2.dilate(harris, None)
        corners = np.argwhere(harris > 0.01 * harris.max())
        corner_count = int(len(corners))

        return {
            "gray": gray,
            "mask": mask,
            "kps": kps if kps else [],
            "des": des,
            "edge_count": edge_count,
            "corner_count": corner_count
        }

    # -------- scoring & stability --------
    def _classify(self, feats):
        if not self.refs:
            return None, 0.0, {}

        best_label, best_score = None, -1.0
        for lbl, ref in self.refs.items():
            s_orb = self._match_orb(feats["des"], ref["des"])
            s_tm = self._template_sim(feats["gray"], ref["gray"])
            s_edge = self._count_sim(feats["edge_count"], ref["edge_count"])
            s_corner = self._count_sim(feats["corner_count"], ref["corner_count"])

            combined = (self.w_orb * s_orb +
                        self.w_tm * s_tm +
                        self.w_edge * s_edge +
                        self.w_corner * s_corner)

            if combined > best_score:
                best_score = combined
                best_label = lbl

        return best_label, float(best_score), {}

    def _match_orb(self, des_q, des_r, ratio=0.75):
        if des_q is None or des_r is None:
            return 0.0
        if len(des_q) < 2 or len(des_r) < 2:
            return 0.0
        try:
            matches = self.matcher.knnMatch(des_q, des_r, k=2)
            good = 0
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ratio * n.distance:
                        good += 1
            return min(good / 50.0, 1.0)
        except:
            return 0.0

    @staticmethod
    def _template_sim(g1, g2):
        if g1.shape != g2.shape:
            return 0.0
        r = cv2.matchTemplate(g1, g2, cv2.TM_CCOEFF_NORMED)
        v = float(r[0, 0])
        return max(0.0, (v + 1.0) / 2.0)

    @staticmethod
    def _count_sim(c1, c2):
        diff = abs(c1 - c2)
        mx = max(c1, c2, 1)
        return 1.0 - min(diff / mx, 1.0)

    def _push(self, letter, conf):
        self.history.append(letter)
        self.conf_hist.append(conf)

    def _stable(self):
        if len(self.history) < self.history_size:
            return None, 0.0
        counts, conf_sum = {}, {}
        for ltr, c in zip(self.history, self.conf_hist):
            if ltr is None:
                continue
            counts[ltr] = counts.get(ltr, 0) + 1
            conf_sum[ltr] = conf_sum.get(ltr, 0.0) + float(c)
        if not counts:
            return None, 0.0
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        freq = counts[best] / float(len(self.history))
        mean_conf = conf_sum[best] / max(1, counts[best])
        if freq >= self.stability_freq and mean_conf >= self.stability_conf:
            return best, mean_conf
        return None, 0.0
