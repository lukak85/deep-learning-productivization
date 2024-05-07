import requests
import json

# The URL where the Flask app is exposed
# url = 'http://localhost:5000/predict'
url = 'http://localhost:8000'


# The text you want to send for prediction
data = {
        'text': 'Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah.',
        'question': 'Katera reka prečka mesto Ljubljana?'
        }

# Convert the dictionary to JSON format
data_json = json.dumps(data)

# Send a POST request to the Flask app
response = requests.post(url, json=data_json)

# Print the response from the server
print(response.json())
