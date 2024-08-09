import time

import cv2
import numpy as np
import torch
import tritonclient.grpc as grpcclient

from utils import non_max_suppression


NUMBER_OF_CLASSES = 80
IMAGE_SIZE = (640, 640)
LABELS = [line.rstrip("\n") for line in open("coco.txt")]

DEBUG_FRAME_BREAK = 100


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


def main(host, port):
    url = f"{host}:{port}"

    print(url)

    # cap = cv2.VideoCapture("../../../datasets/traffic/cctv052x2004080620x00108.avi")
    cap = cv2.VideoCapture("../../testing/test.mp4")

    ratio = None
    
    out = None # Saving video

    count = 0

    frame_list = []
    preprocessed_frame_list = []
    fp_result_list = []

    while True:
        print(count)

        ret, frame = cap.read()

        if not ret:
            break

        frame_list.append(frame)

        if ratio is None:
            ratio = (frame.shape[0] / IMAGE_SIZE[0], frame.shape[1] / IMAGE_SIZE[1])
            print(frame.shape)
            print(IMAGE_SIZE)
            print(ratio)

        preprocessed_img = preprocess(frame)
        preprocessed_frame_list.append(preprocessed_img)

        if count >= DEBUG_FRAME_BREAK:
            break

        count = count + 1

    # Timing the inference of the video
    start_time = time.time()

    with grpcclient.InferenceServerClient(url=url) as triton_client:
        for preprocessed_img in preprocessed_frame_list:
            inputs = [grpcclient.InferInput("images", [*preprocessed_img.shape], "FP32")]

            outputs = [grpcclient.InferRequestedOutput("output0")]

            inputs[0].set_data_from_numpy(preprocessed_img)

            fp_result_list.append(triton_client.infer(model_name="yolov5", inputs=inputs, outputs=outputs).as_numpy("output0"))

    print(f"Time taken: {time.time() - start_time}")

    for results in fp_result_list:
        results = postprocess(results)
        # Draw boxes
        if results[0] is not None and isinstance(results[0], torch.Tensor):
            results = results[0].numpy()
            for result in results:
                x1, y1 = int(result[0] * ratio[1]), int(result[1] * ratio[0])
                x2, y2 = int(result[2] * ratio[1]), int(result[3] * ratio[0])
                conf = result[4]
                label = LABELS[int(result[5])]
                text = f"{label} {conf:.2f}"
                cv2.rectangle(frame_list[0], (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    frame_list[0],
                    text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 0),
                    2,
                )

        if out is None:
            out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_list[0].shape[1], frame_list[0].shape[0]))

        out.write(frame_list[0])

        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        frame_list.pop(0)

    out.release()


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
