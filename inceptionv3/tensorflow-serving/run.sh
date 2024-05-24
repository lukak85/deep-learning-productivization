docker run --gpus all -p 8500:8500 --name inceptionv3_serving \
    --mount type=bind,source=./tmp/inceptionv3,target=/models/inceptionv3 \
    -e MODEL_NAME=inceptionv3 -t tensorflow/serving:latest-gpu