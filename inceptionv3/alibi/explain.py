import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input,
    decode_predictions,
)
from alibi.datasets import load_cats
from alibi.explainers import AnchorImage

model = InceptionV3(weights="imagenet")

image_shape = (299, 299, 3)
data, labels = load_cats(target_size=image_shape[:2], return_X_y=True)
print(f"Images shape: {data.shape}")

images = preprocess_input(data)
preds = model.predict(images)
label = decode_predictions(preds, top=3)
print(label[0])

predict_fn = lambda x: model.predict(x)

segmentation_fn = "slic"
kwargs = {"n_segments": 15, "compactness": 20, "sigma": 0.5, "start_label": 0}
explainer = AnchorImage(
    predict_fn,
    image_shape,
    segmentation_fn=segmentation_fn,
    segmentation_kwargs=kwargs,
    images_background=None,
)

i = 0
plt.imshow(data[i])

image = images[i]
np.random.seed(0)
explanation = explainer.explain(image, threshold=0.95, p_sample=0.5, tau=0.25)

plt.imshow(explanation.anchor)

plt.show()

plt.imshow(explanation.segments)

print(explanation.segments.shape)
print(type(explanation.segments))

plt.show()
