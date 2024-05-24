import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import (  # type: ignore
    InceptionV3,
)

model = InceptionV3(weights="imagenet")

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Save the model
model.save("tmp/inceptionv3/1/", save_format="tf")
