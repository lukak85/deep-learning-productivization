# Deployment of YOLOv5 <!-- omit in toc -->

YOLOv5 is a state-of-the-art, real-time object detection system that is capable of detecting objects in images and videos. In this repository, YOLOv5 is used for object detection in videos, such as traffic surveillance.

Currently, as of our knowledge, the only deployment option for YOLOv5 with inference input and output streaming is with NVIDIA Triton Inference Server. This repository provides a guide on how to deploy YOLOv5.

We provide the following deployment options for IVA using YOLOv5:

- [Locally](#locally)
- [With NVIDIA Triton Inference Server](#with-nvidia-triton-inference-server)

## Requirements <!-- omit in toc -->

Create a `yolov5` Anaconda environment:

```bash
conda create -n yolov5 python=3.8
```

Activate the environment:

```bash
conda activate yolov5
```

Install the required packages:

```bash
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

## Locally

TODO

## With NVIDIA Triton Inference Server

In order to deploy YOLOv5 on NVIDIA Triton Inference Server, we export the model to ONNX format. Luckily the Ultralytics team have already provided a way to export the model for us. To export to ONNX, either:

- export by hand. This is done by cloning the [YOLOv5 repository](https://github.com/ultralytics/yolov5):

  ```bash
  git clone https://github.com/ultralytics/yolov5.git
  ```
  
  Then inside [yolov5](/yolov5/yolov5/) folder, run the following command to export the model to ONNX format:

  ```bash
  python export.py --weights yolov5s.pt --include torchscript onnx
  ```

- download the already exported model:

  ```bash
  wget https://drive.google.com/file/d/1qrvxjhgo8nMYOthgPxHl4TOiY7IS2IFn/view?usp=drive_link
  ```

Move the model to the [model_repository/yolov5/1](./triton-inference-server/model_repository/yolov5/1/) directory.

Inside the [triton-inference-server](/yolov5/triton-inference-server/) folder, run the following command to start the model on Triton Inference Server:

```bash
./setup.sh
```

This will start the Triton Inference Server with the YOLOv5 model. The server will be running on `localhost:8001` for gRPC.

### SLURM with Apptainer adaptation

In order to run our model on HPC, we need to adapt it to the Apptainer format. To do that, we first pull the NVIDIA Triton Inference Server container using this command:

```bash
apptainer pull tritonserver.sif docker://nvcr.io/nvidia/tritonserver:23.03-py3
```

We then run the model using the following command:

```bash
srun -G1 --partition=gpu apptainer run --nv --bind ${PWD}/model_repository:/mnt/model_repository ${PWD}/tritonserver.sif tritonserver --model-repository=/mnt/model_repository --log-verbose 1
```