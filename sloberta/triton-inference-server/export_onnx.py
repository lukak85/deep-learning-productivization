from pathlib import Path

import onnx
import torch

from transformers import AutoTokenizer, CamembertForQuestionAnswering


class TritonModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Setup constants
        MODEL_DIR = "./../model-dir"

        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Setup constants
        self.model = CamembertForQuestionAnswering.from_pretrained(MODEL_DIR)
        self.model.eval()
        self.model.to(self.device)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    def forward(self, question, text):
        # Tokenize the input
        tokenized_data = self.tokenizer(question, text, return_tensors="pt")
        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]

        input_ids = torch.tensor(input_ids).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        answer_start_index = outputs[0].argmax()
        answer_end_index = outputs[1].argmax()

        predict_answer_tokens = tokenized_data.input_ids[
            0, answer_start_index : answer_end_index + 1
        ]

        prediction = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )

        # Return the prediction
        return str(prediction)


# Define the input
text = "Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
question = "Katera reka prečka mesto Ljubljana?"

wrapped_model = TritonModelWrapper()

wrapped_model.eval()

answer = wrapped_model(question, text)

# Export the model to ONNX
torch.onnx.export(
    wrapped_model,
    (question, text),
    Path("model.onnx"),
    verbose=True,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["question", "text"],
    output_names=["prediction"],
)

onnx.checker.check_model("model.onnx")
