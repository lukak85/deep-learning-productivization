import json
import requests
import sys

# Define the variables
SERVICE_HOSTNAME = "torchserve.default.example.com"
INGRESS_HOST = "localhost"
INGRESS_PORT = "80"
MODEL_NAME = "sloberta"

EXAMPLE = sys.argv[1]

# Load the JSON data from file
with open(f"../test/examples/{EXAMPLE}.json", "r") as f:
    data = json.load(f)

# Define the headers
headers = {
    "Host": SERVICE_HOSTNAME,
    "Content-Type": "application/json",
}

# Send the request
response = requests.post(
    f"http://{INGRESS_HOST}:{INGRESS_PORT}/v1/models/{MODEL_NAME}:predict",
    headers=headers,
    json=data,
)

# Print the response
print(response.text)
