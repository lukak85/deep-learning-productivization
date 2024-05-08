import requests

from utils import get_example

text, question, _ = get_example()

example = [{"data": {"text": text, "question": question}}]

print(example)

response = requests.get("http://localhost:8085/predictions/sloberta", json=example)

print(response.text)
