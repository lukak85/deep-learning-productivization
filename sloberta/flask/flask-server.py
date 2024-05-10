from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, CamembertForQuestionAnswering

app = Flask(__name__)

model_dir = "./../model-dir"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = CamembertForQuestionAnswering.from_pretrained(model_dir)


@app.route("/predict", methods=["POST"])
def predict():
    # Get data from POST request
    text = request.json["text"]
    question = request.json["question"]

    inputs = tokenizer(question, text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]

    return tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
