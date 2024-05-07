from flask import Flask, request, jsonify
from transformers import AutoTokenizer, CamembertForQuestionAnswering
import torch
import json

app = Flask(__name__)

# Load the model and tokenizer
output_dir = "./sloberta-squad2-SLO"
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = CamembertForQuestionAnswering.from_pretrained(output_dir)


@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the POST request.
    data = json.loads(request.get_json())
    print(type(data))
    text = data["text"]
    question = data["question"]

    # Encode the inputs and generate prediction
    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

"""
Examples:
1:
text="Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
question="Katera reka prečka mesto Ljubljana?"
2:
text = "Slovenija je država v srednji Evropi, ki meji na Italijo na zahodu, Avstrijo na severu, Madžarsko na severovzhodu in Hrvaško na jugu. Ima izhod na Jadransko morje. Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko ter kulturno središče. Slovenija je znana po svoji raznoliki pokrajini, ki vključuje alpske gore, goste gozdove, prelepe jezere in kilometre obale. Država je postala članica Evropske unije leta 2004 in je leta 2007 sprejela evro kot uradno valuto. Slovenija je tudi članica Schengenskega območja, NATO in Združenih narodov."
question = "Kdaj je Slovenija postala članica Evropske unije?"
"""
