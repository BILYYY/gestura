import cv2


class SubtitleManager:
    """
    Two-line, movie-style subtitles at bottom with auto-wrap.
    """

    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.8, thickness=2):
        self.buffer = ""  # full text
        self.font = font
        self.scale = scale
        self.thickness = thickness
        self.max_width_ratio = 0.92
        self.bottom_margin = 12

    # text ops
    def add_letter(self, ch: str):
        if ch and len(ch) == 1:
            self.buffer += ch

    def add_space(self): self.buffer += " "
    def backspace(self):
        if self.buffer: self.buffer = self.buffer[:-1]
    def clear(self): self.buffer = ""

    # rendering
    def _wrap(self, frame_width):
        max_w = int(self.max_width_ratio * frame_width)
        words = self.buffer.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip() if cur else w
            (tw, _), _ = cv2.getTextSize(trial, self.font, self.scale, self.thickness)
            if tw <= max_w:
                cur = trial
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines[-2:] if len(lines) > 2 else lines

    def draw(self, frame):
        h, w = frame.shape[:2]
        lines = self._wrap(w)
        if not lines: return frame

        band_h = 60 if len(lines) == 2 else 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - band_h), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        y = h - self.bottom_margin
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, self.font, self.scale, self.thickness)
            x = (w - tw) // 2
            cv2.putText(frame, line, (x, y), self.font, self.scale, (0, 0, 0), self.thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), self.font, self.scale, (255, 255, 255), self.thickness, cv2.LINE_AA)
            y -= (th + 10)
        return frame
