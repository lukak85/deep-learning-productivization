import json

import numpy as np
import torch
from transformers import AutoTokenizer, CamembertForQuestionAnswering

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


model_dir = "./../model-dir"

tokenizer = AutoTokenizer.from_pretrained(model_dir)


def answer_question(triton_client, text, question, timeout=None):
    input_tokens = tokenizer(question, text, return_tensors="pt")

    sequence_length = input_tokens["input_ids"].shape[1]

    # http_tensors = [
    #     httpclient.InferInput("attention_mask", (1, sequence_length), "INT64"),
    #     httpclient.InferInput("input_ids", (1, sequence_length), "INT64"),
    # ]

    inputs = [
        grpcclient.InferInput("input_ids", (1, sequence_length), "INT64"),
        grpcclient.InferInput("attention_mask", (1, sequence_length), "INT64"),
    ]

    outputs = [grpcclient.InferRequestedOutput("logits")]

    # Tokenized input tensors -> triton.
    # http_tensors[0].set_data_from_numpy(inputs["attention_mask"].numpy())
    # http_tensors[1].set_data_from_numpy(inputs["input_ids"].numpy())

    inputs[0].set_data_from_numpy(input_tokens["attention_mask"].numpy())
    inputs[1].set_data_from_numpy(input_tokens["input_ids"].numpy())

    # Get the result from the server.
    result = triton_client.infer(
        model_name="sloberta", inputs=inputs, outputs=outputs, timeout=timeout
    )

    # Reshape back to `sequence_length`
    server_start = result.as_numpy(f"logits")[0][:sequence_length]
    server_end = result.as_numpy(f"logits")[1][:sequence_length]

    # Use numpy to get the predicted start and end position from the
    # output softmax scores.
    answer_start_index = server_start.argmax()
    answer_end_index = server_end.argmax()

    predict_answer_tokens = input_tokens.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]

    # Convert it into human readable string,
    return tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


def main(host, port):
    url = f"{host}:{port}"
    # triton_client = httpclient.InferenceServerClient(url=url)
    triton_client = grpcclient.InferenceServerClient(url=url)

    text = "Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
    question = "Katera reka prečka mesto Ljubljana?"

    output = answer_question(triton_client, text, question)

    print(f"Answer: {output}")

    # Close server connection.
    triton_client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="sloberta-cli")
    parser.add_argument("-s", "--server", required=True, help="Inference server host")
    parser.add_argument(
        "-p",
        "--port",
        required=False,
        default="8001",
        help="Inference server port",
    )
    args = parser.parse_args()
    main(args.server, args.port)
