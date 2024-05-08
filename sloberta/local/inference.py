from transformers import AutoTokenizer, CamembertForQuestionAnswering
import torch

from utils import get_example


model_dir = "./../model-dir"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = CamembertForQuestionAnswering.from_pretrained(model_dir)

text = None
question = None
predicted_answer = None

# Keep predicting until we get a non-empty answer
while predicted_answer is None or predicted_answer.strip() == "":
    text, question, _ = get_example()

    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]

    predicted_answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

print(text)
print(question)
print(predicted_answer)
