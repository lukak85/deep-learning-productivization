import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
    decode_predictions,
)
from PIL import Image as PILImage

IMG_SIZE = (299, 299)

model = InceptionV3(weights="imagenet")
# Dataset folder
DATASET_FOLDER = "../dataset"

# Select a random category folder, then a random image from that category
category = np.random.choice(os.listdir(DATASET_FOLDER))
image = np.random.choice(os.listdir(os.path.join(DATASET_FOLDER, category)))

print(f"Category: {category}")
print(f"Image: {image}")

image = PILImage.open(DATASET_FOLDER + "/" + category + "/" + image)
image = image.resize(IMG_SIZE, resample=PILImage.Resampling.BILINEAR)
image = np.array(image)
image = np.expand_dims(image, axis=0)

images = preprocess_input(image)
preds = model.predict(images)
label = decode_predictions(preds, top=3)
print(label[0])
