import os

import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow.keras.applications.inception_v3 import (  # type: ignore
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image  # type: ignore


channel = grpc.insecure_channel("localhost:8500")
service_stub = PredictionServiceStub(channel)

# Dataset folder
DATASET_FOLDER = "../dataset"
IMG_SIZE = (299, 299)

# Select a random category folder, then a random image from that category
category = np.random.choice(os.listdir(DATASET_FOLDER))
image_name = np.random.choice(os.listdir(os.path.join(DATASET_FOLDER, category)))

print(f"Category: {category}")
print(f"Image: {image_name}")

img = image.load_img(
    DATASET_FOLDER + "/" + category + "/" + image_name, target_size=(299, 299)
)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

grpc_request = PredictRequest()
grpc_request.model_spec.name = "inceptionv3"
grpc_request.model_spec.signature_name = "serving_default"
grpc_request.inputs["keras_tensor"].CopyFrom(tf.make_tensor_proto(x, shape=x.shape))

result = service_stub.Predict(grpc_request, 600.0)  # 600 sec timeout
predictions = tf.make_ndarray(result.outputs["output_0"])
label = decode_predictions(predictions, top=3)
print(label)
