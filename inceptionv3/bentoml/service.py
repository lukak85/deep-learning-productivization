from __future__ import annotations
import bentoml
import numpy as np
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
    decode_predictions,
)
import tensorflow as tf
from PIL.Image import Image as PILImage
from PIL.Image import Resampling

from alibi.datasets import load_cats
from alibi.explainers import AnchorImage


IMG_SIZE = (299, 299)
IMG_SIZE_EXPLAIN = (299, 299, 3)


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 600},
)
class ImageAnalysisService:
    def __init__(self):
        self.model = InceptionV3(weights="imagenet")
        print(
            "Using GPU"
            if len(tf.config.list_physical_devices("GPU")) > 0
            else "Using CPU"
        )

    @bentoml.api
    def detect(self, image: PILImage):
        image = self.preprocess(image)
        image = preprocess_input(image)
        preds = self.model.predict(image)
        label = decode_predictions(preds, top=3)
        return {label[0][1][1]}

    @bentoml.api
    def explain(self, image: PILImage) -> np.ndarray:
        image = image.resize(IMG_SIZE, resample=Resampling.BILINEAR)
        image = np.array(image)

        predict_fn = lambda x: self.model.predict(x)

        segmentation_fn = "slic"
        kwargs = {"n_segments": 15, "compactness": 20, "sigma": 0.5, "start_label": 0}
        explainer = AnchorImage(
            predict_fn,
            IMG_SIZE_EXPLAIN,
            segmentation_fn=segmentation_fn,
            segmentation_kwargs=kwargs,
            images_background=None,
        )

        np.random.seed(0)
        explanation = explainer.explain(image, threshold=0.95, p_sample=0.5, tau=0.25)

        return explanation.anchor

    def preprocess(self, image):
        image = image.resize(IMG_SIZE, resample=Resampling.BILINEAR)
        image = np.array(image)
        return np.expand_dims(image, axis=0)
