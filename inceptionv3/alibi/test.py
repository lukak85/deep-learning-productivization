import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
    decode_predictions,
)
from alibi.datasets import load_cats
from alibi.explainers import AnchorImage
import os
import numpy as np
from PIL import Image as PILImage

model = InceptionV3(weights="imagenet")

# Dataset folder
DATASET_FOLDER = "../dataset"

# Select a random category folder, then a random image from that category
category = np.random.choice(os.listdir(DATASET_FOLDER))
image = np.random.choice(os.listdir(os.path.join(DATASET_FOLDER, category)))

print(f"Category: {category}")
print(f"Image: {image}")

image = PILImage.open(DATASET_FOLDER + "/" + category + "/" + image)
image = image.resize((299, 299), resample=PILImage.Resampling.BILINEAR)

image = np.array(image)

predict_fn = lambda x: model.predict(x)

segmentation_fn = "slic"
kwargs = {"n_segments": 15, "compactness": 20, "sigma": 0.5, "start_label": 0}
explainer = AnchorImage(
    predict_fn,
    (299, 299, 3),
    segmentation_fn=segmentation_fn,
    segmentation_kwargs=kwargs,
    images_background=None,
)

np.random.seed(0)
explanation = explainer.explain(image, threshold=0.95, p_sample=0.5, tau=0.25)

plt.imshow(explanation.anchor)

plt.show()

plt.imshow(explanation.segments)

print(explanation.segments.shape)
print(type(explanation.segments))

plt.show()
