# Requirements:

TODO: add requirements

# How to run the SloBERTa model

## Locally

TODO: add instructions

## With GPU support

TODO: add instructions

## With TorchServe

TODO: add instructions

## With Docker

TODO: add instructions on how to build the image

Running a docker container with the Sloberta model and GPU support

```bash
docker run --gpus all -dp 8080:8080 -p 8081:8081 --name sloberta sloberta
```

## With NVIDIA Triton Inference Server

Move into the [triton-inference-server](./triton-inference-server/) directory and run the following command to export a TorchScript model:

```bash
python export_model.py
```

Or download the model (**TODO**).

Move the model to the [model_repository/sloberta/1](./triton-inference-server/model_repository/sloberta/1/) directory.

Test the inference on the exported TorchScript model by running the following command:

```bash
python test-saved-model.py
```

Run the following command to start the model on Triton Inference Server:

```bash
./setup.sh
```

### Testing the model

#### Running a basic inference

Use Triton Client to call Triton Inference Server and get the inference:

```bash
python test-saved-model.py -s localhost
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

Follow this tutorial to establish PV and PVC: TODO.

Move the model from [pv](/pv/) to the PV:

```bash
kubectl cp ./pv/ model-store-pod:/ -c model-store
```

Apply the [SloBERTa model](/k8s/sloberta/torchserve.yaml) to the cluster:

```bash
kubectl apply -f ./k8s/sloberta/torchserve.yaml
```

### Testing the model

#### Running a basic inference

Inside [test](./test/), run the following bash script:

```bash
./inference.sh
```

Expected result:

```bash
{"predictions":["Ljubljanica"]}
```

#### Running a load test

Run the follwoing command (preforms the test in 30s with 5 concurrent users):

```bash
hey -z 30s -c 5 -m POST -host torchserve.default.example.com -H "Content-Type: application/json" -d '{"instances":[{"data":{"text":"Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovina pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah.","question":"Katera reka prečka mesto Ljubljana?"}}]}' http://localhost:80/v1/models/sloberta:predict
```
