from flask import Flask, request, jsonify
import numpy as np
import torch

app = Flask(__name__)

model = torch.hub.load("ultralytics/yolov5", "yolov5x")


@app.route("/echo", methods=["GET"])
def echo():
    print("Endpoint: /echo")
    return {"message": "Hello, World!"}


@app.route("/predict", methods=["POST"])
def predict():
    print("Endpoint: /predict")

    img = request.json["image"]

    results_array = []

    result = model(np.array(img))
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

    data = [
        [float(x) if isinstance(x, (float, np.float32)) else int(x) for x in sublist]
        for sublist in results_array
    ]

    return {"results": data}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
