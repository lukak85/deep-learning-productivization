# Deployment of SloBERTa <!-- omit in toc -->

The [SloBERTa](http://hdl.handle.net/11356/1778) model is a Transformer-based solving the downstream task of question answering. The model was trained and evaluated using the [Slovene translation of the SQuAD2.0 dataset](http://hdl.handle.net/11356/1756).

We provide the following deployment options for the SloBERTa model:

- [Locally](#locally)
- [With Flask](#with-flask)
- [With TorchServe](#with-torchserve)
- [With Docker](#with-docker)
  - [GPU Support](#gpu-support)
- [With NVIDIA Triton Inference Server](#with-nvidia-triton-inference-server)
- [With KServe](#with-kserve)
- [With Seldon Core](#with-seldon-core)

### Requirements <!-- omit in toc -->

Before running, install the required Python packages (preferably in a Conda environment):

```bash
pip install -r requirements.txt
```

**TODO**: Check if all of the packages (or their explicit specifications) are needed, check their versions and split them into separate files (for deployment specific requirements).

## Locally

Before starting, either:

- dowload the following files:

  ```bash
  wget https://drive.google.com/file/d/1clOno-RzAhOizCzVuvmP3orgYLGEQ_1-/view?usp=sharing
  wget https://drive.google.com/file/d/1iJ8RvFknig0WLwopCjhOaPuMWQQLNI1x/view?usp=sharing
  ```

- download the model from [CLARIN.SI](http://hdl.handle.net/11356/1778).

Move the downloaded files to the [model-dir](./model-dir/) directory.

To run an inference, simply run the following Python script inside the [local](./local/) directory:

```bash
python inference.py
```

Expected result:

```bash
Ljubljanica
```

## With Flask

We deploy a Flask server exposing an endpoint POST taking in a JSON example and returning the answer to the question. Run the Flask server by moving inside the [flask](./flask/) directory and running the following command:

```bash
python flask-server.py
```

Run an inference on the model by running the following inside [examples](./test/examples/) directory:

```bash
curl -v -H "Content-Type: application/json" http://localhost:5000/predict -d @./example1.json
```

## With TorchServe

Inside [torchserve](./torchserve/) directory a `.mar` (Model ARchive) file by running the following command:

```bash
./build.sh
```

Run TorchServe deployed model with the following command:

```bash
./run.sh
```

Running inference on the model:

- using a bash script:

  ```bash
  ./inference.sh
  ```

  Expected result:

  ```bash
  Ljubljanica
  ```

- using a Python script:
  ```bash
  python inference.py
  ```

Note: TorchServe is configured in such a way that the same files can be used when deploying the model on [KServe](#with-kserve); this is why the model resides in the [pv](./pv/) directory.

## With Docker

Build the Docker image by running the following command:

```bash
docker build -t sloberta .
```

Running a docker container with the Sloberta model and GPU support

```bash
docker run -dp 8085:8085 -p 8081:8081 --name sloberta sloberta
```

Running inference on the model (same as with [TorchServe](#with-torchserve)):

- using a bash script:

  ```bash
  ./inference.sh
  ```

  Expected result:

  ```bash
  Ljubljanica
  ```

- using a Python script:
  ```bash
  python inference.py
  ```

Note: As mentioned before, the model and the configuration files are inside the [pv](./pv/) directory for easy deployment on [KServe](#with-kserve) later on.

### GPU Support

If NVIDIA GPU is present, add `--gpus all` to the `docker run` command:

```bash
docker run --gpus all -dp 8085:8085 -p 8081:8081 --name sloberta sloberta
```

## With NVIDIA Triton Inference Server

In order to deploy the model on NVIDIA Triton Inference Server, we first need to export the model to TorchScript. To do that, either:

- move into the [triton-inference-server](./triton-inference-server/) directory and run the following command to export a TorchScript model:

  ```bash
  python export_model.py
  ```

- download the already exported model:

  ```bash
  wget https://drive.google.com/file/d/1qrvxjhgo8nMYOthgPxHl4TOiY7IS2IFn/view?usp=drive_link
  ```

Move the model to the [model_repository/sloberta/1](./triton-inference-server/model_repository/sloberta/1/) directory.

Test the inference on the exported TorchScript model by running the following command:

```bash
python test-saved-model.py
```

Run the following command to start the model on Triton Inference Server:

```bash
./setup.sh
```

### Testing the Model <!-- omit in toc -->

#### Running a Basic Inference <!-- omit in toc -->

Use Triton Client to call Triton Inference Server and get the inference:

```bash
python client.py -s localhost
```

Expected result:

```bash
Answer: Ljubljanica
```

## With KServe

For deploying a PyTorch model on KServe, read: [PyTorch - KServe Documentation Website](https://kserve.github.io/website/0.11/modelserving/v1beta1/torchserve/).

For KServe installation, run (when in the correct context):

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
```

### Moving the Model to PVC <!-- omit in toc -->

Create a SC (Storage Class) by running the following command inside the [k8s](./k8s/) directory:

```bash
kubectl apply -f ./k8s/sc.yaml
```

Create PV (Persistent Volume) and PVC (Persistent Volume Claim) by running the following command inside the [k8s](./k8s/) directory:

```bash
kubectl apply -f ./k8s/pv-pvc.yaml
```

Create a `model-store` pod by running the following command inside the [k8s](./k8s/) directory:

```bash
kubectl apply -f ./k8s/pv-model-store.yaml
```

Move the model from [pv](/pv/) on your local machine to the model-store-pod by running the following command:

```bash
kubectl cp ./pv/ model-store-pod:/ -c model-store
```

### Deploying the Model <!-- omit in toc -->

Apply the [SloBERTa model](/k8s/sloberta/torchserve.yaml) to the cluster:

```bash
kubectl apply -f ./k8s/sloberta/torchserve.yaml
```

### Testing the Model <!-- omit in toc -->

#### Running a Basic Inference <!-- omit in toc -->

Inside [test](./test/), run the following bash script:

```bash
./inference.sh
```

Expected result:

```bash
{"predictions":["Ljubljanica"]}
```

#### Running a Load Test <!-- omit in toc -->

Run the follwoing command with [Hey](https://github.com/rakyll/hey) (preforms the test in 30s with 5 concurrent users):

```bash
hey -z 30s -c 5 -m POST -host torchserve.default.example.com -H "Content-Type: application/json" -d '{"instances":[{"data":{"text":"Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovina pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah.","question":"Katera reka prečka mesto Ljubljana?"}}]}' http://localhost:80/v1/models/sloberta:predict
```

### Explaining Predictions <!-- omit in toc -->

For a given prediction a model explainers answers the question "Why did my model make this prediction?". KServe integrates [Alibi Explainer](https://github.com/SeldonIO/alibi) implementing a black-box algorithm by generating a bunch of similar instances for a given input and sends them out to model explainer to get the explanation.

TODO: Add implementation.

## With Seldon Core

For Seldon Core installation, read [Install Locally](https://docs.seldon.io/projects/seldon-core/en/latest/install/kind.html).

Port forward using the following command:

```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

Moving inside the [seldon-core](/sloberta/seldon-core/) directory, build a Docker image by running the following command:

```bash
./build.sh
```

Deploy the model by running the following command:

```bash
kubectl apply -f ./k8/setup.yaml
```

### Testing the Model <!-- omit in toc -->

#### Running a Basic Inference <!-- omit in toc -->

Inside the [seldon-core](/sloberta/seldon-core/) folder inference on the model by running the following command:

```bash
./inference.sh
```
