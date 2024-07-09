from utils import non_max_suppression

import cv2
import numpy as np
import torch
import tritonclient.grpc as grpcclient


NUMBER_OF_CLASSES = 80
IMAGE_SIZE = (640, 640)
LABELS = [line.rstrip("\n") for line in open("coco.txt")]


"""
Modified postprocess helper functions obtained at YOLOV5's GitHub repository and used under AGPL license:
https://github.com/ultralytics/yolov5
"""


def preprocess(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype(np.float32)
    img /= 255.0  # Normalize to 0.0 - 1.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(predictions, conf_thresh=0.25, iou_thresh=0.45):
    predictions = non_max_suppression(
        torch.from_numpy(predictions), conf_thresh, iou_thresh
    )
    return predictions


def detect(client, img):
    preprocessed_img = preprocess(img)

    inputs = [grpcclient.InferInput("images", [*preprocessed_img.shape], "FP32")]

    outputs = [grpcclient.InferRequestedOutput("output0")]

    inputs[0].set_data_from_numpy(preprocessed_img)

    results = client.infer(model_name="yolov5", inputs=inputs, outputs=outputs)

    return postprocess(results.as_numpy("output0"))


def main(host, port):
    url = f"{host}:{port}"
    triton_client = grpcclient.InferenceServerClient(url=url)

    # cap = cv2.VideoCapture("../../../datasets/traffic/cctv052x2004080620x00108.avi")
    cap = cv2.VideoCapture("../../testing/test.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect(triton_client, frame)

        # Draw boxes
        img = cv2.resize(frame, (640, 640))
        if results[0] is not None and isinstance(results[0], torch.Tensor):
            results = results[0].numpy()
            for result in results:
                x1, y1 = int(result[0]), int(result[1])
                x2, y2 = int(result[2]), int(result[3])
                conf = result[4]
                label = LABELS[int(result[5])]
                text = f"{label} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    img,
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 0),
                    2,
                )

        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="yolov5-cli")
    parser.add_argument("-s", "--server", required=True, help="Inference server host")
    parser.add_argument(
        "-p",
        "--port",
        required=False,
        default="8001",
        help="Inference server port",
    )
    args = parser.parse_args()
    main(args.server, args.port)
