from __future__ import annotations
import bentoml
import numpy as np
import torch


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 600},
)
class ImageDetectionService:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5x")

    @bentoml.api
    def detect(self, image: np.ndarray) -> np.ndarray:
        results_array = []

        result = self.model(image)
        df = result.pandas().xyxy[0]

        for i in range(len(df)):
            results_array.append(
                [
                    df["xmin"][i],
                    df["ymin"][i],
                    df["xmax"][i],
                    df["ymax"][i],
                    df["confidence"][i],
                    df["class"][i],
                ]
            )
        return np.array(results_array)
