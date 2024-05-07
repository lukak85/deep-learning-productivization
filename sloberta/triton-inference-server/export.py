import torch

from transformers import AutoTokenizer, CamembertForQuestionAnswering


class TritonModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Make a tensor with shape (2, sequence_length) where the first row is the start logits
        # and the second row is the end logits.
        return torch.cat((outputs["start_logits"], outputs["end_logits"]), dim=0)


model_dir = "./../model-dir"

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CamembertForQuestionAnswering.from_pretrained(model_dir)
model.eval()
model.to(device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Define the input
text = "Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
question = "Katera reka prečka mesto Ljubljana?"

# Tokenize the input
tokenized_data = tokenizer(question, text, return_tensors="pt")
input_ids = tokenized_data["input_ids"]
attention_mask = tokenized_data["attention_mask"]

input_ids = torch.tensor(input_ids).to(device)
attention_mask = torch.tensor(attention_mask).to(device)

wrapped_model = TritonModelWrapper(model)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_model = torch.jit.trace(wrapped_model, [input_ids, attention_mask], strict=False)
torch.jit.save(traced_model, "model.pt")
