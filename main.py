import os

from handler.config import configure_argument_parser
from handler.constants import SIZE
from handler.heatmap import HeatmapGenerator
from handler.predict import Predictor
from handler.track import Tracker

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    arg_parser = configure_argument_parser()
    args = arg_parser.parse_args()
    attrs = {
        key: value for key, value in args.__dict__.items() if value is not None
    }
    attrs["imgsz"] = SIZE.get(args.__dict__.get("imgsz"), SIZE["s1K"])

    detector = (
        Tracker(**attrs)
        if args.mode == "tracking"
        else (
            Predictor(**attrs)
            if args.mode == "predict"
            else HeatmapGenerator(**attrs)
        )
    )

    detector.process_video(**attrs)

    if args.save:
        detector.save_video(**attrs)


if __name__ == "__main__":
    main()
