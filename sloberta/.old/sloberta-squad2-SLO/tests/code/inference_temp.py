from transformers import CamembertModel, CamembertConfig
import torch

output_dir = './../sloberta-squad2-SLO'
# Load the model configuration
config = CamembertConfig.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CamembertModel.from_pretrained(output_dir, config=config)

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

paragraph="Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."
question="Katera reka prečka mesto Ljubljana?"


# If you have a GPU, put everything on cuda
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()


print(f"Input paragraph: '{paragraph}'")
print(f"Question: '{question}'")
print(f"Predicted sentiment: {classes[predicted_class_id]}")

