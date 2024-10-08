docker run --gpus=all \
    --name sloberta-triton \
    -it --shm-size=256m --rm \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${PWD}:/workspace/ \
    -v ${PWD}/model_repository:/model_repository nvcr.io/nvidia/tritonserver:23.03-py3 bash

# pip install torch torchvision
# pip install transformers
# tritonserver --model-repository=/model_repository --log-verbose=1