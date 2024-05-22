import os

import bentoml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

client = bentoml.SyncHTTPClient("http://localhost:3000", timeout=600)

# Dataset folder
DATASET_FOLDER = "../dataset"

# Select a random category folder, then a random image from that category
category = np.random.choice(os.listdir(DATASET_FOLDER))
image = np.random.choice(os.listdir(os.path.join(DATASET_FOLDER, category)))

print(f"Category: {category}")
print(f"Image: {image}")

prediction = client.detect(image=Path(DATASET_FOLDER + "/" + category + "/" + image))[0]
print(prediction)
explanation = client.explain(image=Path(DATASET_FOLDER + "/" + category + "/" + image))
plt.imshow(explanation)
plt.show()

# Close the client to release resources
client.close()
