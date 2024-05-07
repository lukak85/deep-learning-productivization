import torch
from transformers import AutoTokenizer

model_repository = "./model_repository/sloberta/1/model.pt"
model_dir = "./../model-dir"

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded = torch.jit.load(model_repository)

# print(loaded)
# print(loaded.code)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

text = "Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
question = "Katera reka prečka mesto Ljubljana?"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = loaded(**inputs.to(device))

print(inputs)
print(outputs)
print(outputs.shape)
print(type(outputs[0][0]))

answer_start_index = outputs[0].argmax()
answer_end_index = outputs[1].argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))
