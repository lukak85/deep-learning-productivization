import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5x")

cap = cv2.VideoCapture("../../datasets/traffic/cctv052x2004080620x00108.avi")

while True:
    img = cap.read()[1]

    img = cv2.resize(img, (640, 640))

    print(img.shape)

    if img is None:
        break

    result = model(img)

    df = result.pandas().xyxy[0]

    print(df)

    results_array = []

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

    for ind in df.index:
        x1, y1 = int(df["xmin"][ind]), int(df["ymin"][ind])
        x2, y2 = int(df["xmax"][ind]), int(df["ymax"][ind])
        label = df["name"][ind]
        conf = df["confidence"][ind]
        text = f"{label} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2
        )

    cv2.imshow("VIDEO", img)
    cv2.waitKey(10)
