import cv2

from .base import HandlerBase


class Predictor(HandlerBase):
    def prepare_model(self):
        self.model.conf = getattr(self, "conf", 0.2)

    def annotate_frame(
        self, frame: cv2.typing.MatLike
    ) -> tuple[cv2.typing.MatLike, int]:
        results = self.model.predict(
            frame, imgsz=getattr(self, "imgsz", 640), device=self.get_device()
        )

        current_cnt = len(results[0].boxes.xywh)

        if self.hide_labels:
            return self.custom_box(results, frame), current_cnt
        return results[0].plot(), current_cnt

    def draw_history(
        self, results, frame: cv2.typing.MatLike
    ) -> cv2.typing.MatLike:
        return frame
