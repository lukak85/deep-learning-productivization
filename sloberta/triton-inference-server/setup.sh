docker run --gpus all \
    --name sloberta-triton \
    --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver \
    --model-repository=/models