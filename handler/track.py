import cv2
import math
import colorsys
import random
from collections import defaultdict, deque
from ultralytics.engine.results import Results

from .base import HandlerBase


def id_to_color(idx: int) -> tuple[int, int, int]:
    random.seed(idx)
    h = random.random()
    s, v = 0.9, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


def smooth_path(
    points: list[tuple[float, float]], k: int = 5
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    smoothed = []
    for i in range(len(points)):
        lo = max(0, i - k)
        hi = min(len(points), i + k + 1)
        xs, ys = zip(*points[lo:hi])
        smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return smoothed


class Tracker(HandlerBase):
    def __init__(self, *args, line_thickness: int = 2, smooth_tracks: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        default_len = 150
        self.history: defaultdict[int, deque[tuple[float, float]]] = (
            defaultdict(lambda: deque(maxlen=self.lines_history or default_len))
        )
        self.line_thickness = line_thickness
        self.smooth_tracks = smooth_tracks

    def annotate_frame(
        self, frame: cv2.typing.MatLike
    ) -> tuple[cv2.typing.MatLike, int]:
        results = self.model.track(
            frame,
            persist=True,
            conf=getattr(self, "conf", 0.2),
            imgsz=getattr(self, "imgsz", 640),
            device=self.get_device(),
            tracker="models/custom_tracker.yaml",
        )
        try:
            ids = results[0].boxes.id.int().cpu().tolist()
            self.counter.update(ids)
            frame_out = (
                self.custom_box(results, frame)
                if self.hide_labels
                else results[0].plot()
            )
            return frame_out, len(ids)
        except Exception:
            return frame, 0

    def draw_history(
        self, results: Results, frame: cv2.typing.MatLike
    ) -> cv2.typing.MatLike:
        boxes = results[0].boxes.xywh.cpu()
        ids = results[0].boxes.id.int().cpu().tolist()

        for (x, y, w, h), tid in zip(boxes, ids):
            cx, cy = float(x), float(y)
            self.history[tid].append((cx, cy))

        active = set(ids)
        for tid in list(self.history):
            if tid not in active and len(self.history[tid]) < 2:
                self.history.pop(tid, None)

        for tid, pts in self.history.items():
            if len(pts) < 2:
                continue

            path = list(pts)
            if self.smooth_tracks:
                path = smooth_path(path)

            base_color = id_to_color(tid)
            n = len(path) - 1

            for i, (p1, p2) in enumerate(zip(path[:-1], path[1:])):
                alpha = (i + 1) / n
                col = tuple(int(c * alpha) for c in base_color)
                cv2.line(
                    frame,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    col,
                    self.line_thickness,
                    lineType=cv2.LINE_AA,
                )

            p1, p2 = path[-2], path[-1]
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 5:
                cv2.arrowedLine(
                    frame,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    base_color,
                    self.line_thickness + 1,
                    tipLength=0.3,
                    line_type=cv2.LINE_AA,
                )

        return frame
