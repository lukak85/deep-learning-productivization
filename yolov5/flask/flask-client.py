import json
import requests
import sys
import time

import cv2


NUMBER_OF_CLASSES = 80
IMAGE_SIZE = (640, 640)
LABELS = [line.rstrip("\n") for line in open("coco.txt")]

DEBUG_FRAME_BREAK = 100


def main(host, port, debug):
    server_url = f"http://{host}:{port}"
    print(server_url)

    # Dataset folder
    cap = cv2.VideoCapture("../testing/test.mp4")

    ratio = None

    out = None  # Saving video

    count = 0

    frame_list = []
    preprocessed_frame_list = []
    result_list = []

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

        preprocessed_img = cv2.resize(frame, IMAGE_SIZE)
        preprocessed_frame_list.append(preprocessed_img)

        if count >= DEBUG_FRAME_BREAK:
            break

        count = count + 1

    # Timing the inference of the video
    start_time = time.time()

    for preprocessed_img in preprocessed_frame_list:
        image = {"image": preprocessed_img.tolist()}

        response = requests.post(f"{server_url}/predict", json=image)

        # Check if the response is valid and exit if not
        if not response:
            print(response.status_code)
            print(response.reason)
            sys.exit(1)
        else:
            # Print the response
            if debug:
                print(response.text)

        result_list.append(json.loads(response.text)["results"])

    print(f"Time taken: {time.time() - start_time}")

    for results in result_list:
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
            out = cv2.VideoWriter(
                "result.avi",
                cv2.VideoWriter_fourcc(*"XVID"),
                30,
                (frame_list[0].shape[1], frame_list[0].shape[0]),
            )

        out.write(frame_list[0])

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
        default="5000",
        help="Inference server port",
    )
    parser.add_argument(
        "-d",
        "--debug",
        required=False,
        default=False,
        help="Debug mode",
    )
    args = parser.parse_args()
    main(args.server, args.port, args.debug)
