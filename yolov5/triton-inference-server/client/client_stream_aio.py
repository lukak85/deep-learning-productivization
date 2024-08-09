import asyncio
from functools import partial
import queue
import sys
import time

from utils import non_max_suppression

import cv2
import numpy as np
import torch
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


NUMBER_OF_CLASSES = 80
IMAGE_SIZE = (640, 640)
LABELS = [line.rstrip("\n") for line in open("coco.txt")]

BATCH_SIZE = 1
DEBUG_FRAME_BREAK = 100


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


async def async_stream_yield(
    values
):
    # Stream specific variables
    sequence_id = 1000

    count = 0

    for value in values:
        # Create the tensor for input
        value_data = np.full(shape=value.shape, fill_value=value, dtype=np.float32)

        inputs = []
        inputs.append(grpcclient.InferInput("images", [*value.shape], "FP32"))

        # Initialize the data
        inputs[0].set_data_from_numpy(value_data)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("output0"))

        # Issue the asynchronous sequence inference.
        yield {
            "model_name": "yolov5",
            "inputs": inputs,
            "outputs": outputs,
            "request_id": "{}_{}".format(sequence_id, count),
            "sequence_id": sequence_id,
            "sequence_start": (count == 1),
            "sequence_end": (count == len(values)),
        }

        count = count + 1


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


async def main(host, port, sequence="../../testing/test.mp4"):
    url = f"{host}:{port}"

    ratio = None

    out = None  # Saving video

    count = 0

    # Stream specific variables
    frame_list = []
    preprocessed_frame_list = []

    # Reading all the data first
    cap = cv2.VideoCapture(sequence)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        # Temporary break for testing purposes
        if count >= DEBUG_FRAME_BREAK:
            break

        count = count + 1

    # Timing the inference of the video
    start_time = time.time()

    async with grpcclient.InferenceServerClient(
        url=url
    ) as triton_client:
        fp_result_list = []

         # Request iterator that yields the next request
        async def async_request_iterator():
            async for request in async_stream_yield(
                preprocessed_frame_list
            ):
                yield request

        try:
            # Start streaming
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_request_iterator(),
                stream_timeout=None,
            )
            # Read response from the stream
            user_data = UserData()
            async for response in response_iterator:
                result, error = response
                if error:
                    user_data._completed_requests.put(error)
                else:
                    user_data._completed_requests.put(result)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

        # results = postprocess(results.as_numpy("output0"))

        # Retrieve results...
        recv_count = 0
        while recv_count < len(preprocessed_frame_list):
            print(recv_count)
            data_item = user_data._completed_requests.get()

            if type(data_item) == InferenceServerException:
                print(data_item)
                sys.exit(1)
            else:
                try:
                    this_id = data_item.get_response().id.split("_")[0]
                    # if int(this_id) == sequence_id:
                    if True:
                        fp_result_list.append(data_item.as_numpy("output0"))
                    else:
                        print(
                            "unexpected sequence id returned by the server: {}".format(
                                this_id
                            )
                        )
                        sys.exit(1)
                except ValueError:
                    fp_result_list.append(data_item.as_numpy("output0"))
            
            # Temporary break for testing purposes
            if recv_count >= DEBUG_FRAME_BREAK:
                break

            recv_count = recv_count + 1

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
            out = cv2.VideoWriter('result_stream_aio.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_list[0].shape[1], frame_list[0].shape[0]))

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
    asyncio.run(main(args.server, args.port))
