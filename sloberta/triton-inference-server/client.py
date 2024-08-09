import json

import numpy as np

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient


def answer_question(triton_client, text, question, timeout=None):
    # http_tensors = [
    #     httpclient.InferInput("input_ids", (1, sequence_length), "INT64"),
    #     httpclient.InferInput("attention_mask", (1, sequence_length), "INT64"),
    # ]

    inputs = [
        grpcclient.InferInput("text", [1], "BYTES"),
        grpcclient.InferInput("question", [1], "BYTES"),
    ]

    # Tokenized input tensors -> triton.
    # http_tensors[0].set_data_from_numpy(inputs["attention_mask"].numpy())
    # http_tensors[1].set_data_from_numpy(inputs["input_ids"].numpy())

    inputs[0].set_data_from_numpy(np.array([str(text).encode("utf-8")], dtype=np.object_))
    inputs[1].set_data_from_numpy(np.array([str(question).encode("utf-8")], dtype=np.object_))

    outputs = [grpcclient.InferRequestedOutput("answer")]
    
    # Get the result from the server.
    result = triton_client.infer(
        model_name="sloberta", inputs=inputs, outputs=outputs, timeout=timeout
    )

    # 
    # Extract the answer from the response; also hack to remove b" at the start and " at the end,
    # and convert single quotes to double quotes
    answer = str(result.as_numpy("answer"))[2:-1].replace("'", '"')

    # If we just want the answer
    # answer = json.loads(answer)["answer"]

    # Convert it into human readable string,
    return answer


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
