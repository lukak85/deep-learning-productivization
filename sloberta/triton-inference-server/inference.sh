curl --location --request POST 'http://localhost:8000/v2/models/sloberta/infer' \
     --header 'Content-Type: application/json' \
     --data-raw '{
        "inputs":[
        {    
         "name": "text",
         "shape": [1],
         "datatype": "BYTES",
         "data":  ["Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovino pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah."]
        },
        {    
         "name": "question",
         "shape": [1],
         "datatype": "BYTES",
         "data":  ["Katera reka prečka mesto Ljubljana?"]
        }
       ]
     }'
echo