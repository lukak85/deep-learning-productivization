docker run -it --gpus all \
    --name sloberta-triton \
    -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:22.07-py3
# python export.py
