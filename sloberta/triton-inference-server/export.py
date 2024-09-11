import torch

from transformers import pipeline, AutoTokenizer


class TritonModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = pipeline("question-answering", model="./../model-dir")

    def forward(self, question, text):
        return self.pipeline(question=question, context=text)


model_dir = "./../model-dir"

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the input
text = "Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
question = "Katera reka prečka mesto Ljubljana?"

wrapped_model = TritonModelWrapper()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_model = torch.jit.trace(wrapped_model, [question, text], strict=False)
torch.jit.save(traced_model, "model.pt")
