from transformers import AutoTokenizer, CamembertForQuestionAnswering

model_dir = "./model-dir"

model = CamembertForQuestionAnswering.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
