import cv2
import numpy as np

from .base import HandlerBase


class HeatmapGenerator(HandlerBase):
    def __init__(self, *args, alpha: float = 0.4, radius: int = 15, blur: bool = True, **kwargs,):
        super().__init__(*args, **kwargs)
        self.heatmap: np.ndarray | None = None
        self.alpha = alpha
        self.radius = radius
        self.blur = blur
        self.coverage: float = 0.0

    def prepare_model(self):
        self.model.conf = getattr(self, "conf", 0.2)

    def annotate_frame(
        self, frame: cv2.typing.MatLike
    ) -> tuple[cv2.typing.MatLike, int]:
        h, w = frame.shape[:2]
        if self.heatmap is None or self.heatmap.shape != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        results = self.model.predict(frame, imgsz=self.imgsz, device=self.get_device())

        dets = results[0]
        boxes = dets.boxes.xywh.cpu().numpy()
        curr_count = len(boxes)

        for x, y, *_ in boxes:
            cx, cy = int(x), int(y)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(self.heatmap, (cx, cy), self.radius, 1.0, thickness=-1)

        self.coverage = np.count_nonzero(self.heatmap) / (h * w) * 100

        heat_norm = cv2.normalize(
            self.heatmap, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        if self.blur:
            heat_norm = cv2.GaussianBlur(heat_norm, (0, 0), self.radius / 2)
        heat_vis = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(
            heat_vis, self.alpha, frame, 1 - self.alpha, 0
        )

        return blended, curr_count

    def draw_history(self, results, frame):
        return frame

    def counter_box(self, frame, frame_obj_cnt):
        return frame
