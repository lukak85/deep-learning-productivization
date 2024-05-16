from MyModel import MyModel

model = MyModel()

data = [
    {
        "text": "Ljubljana je glavno mesto Slovenije in njeno politično, gospodarsko, kulturno ter znanstveno središče. Mesto stoji na območju, kjer se alpski svet sreča z dinarskim, kar daje Ljubljani poseben čar. Ljubljanica, reka, ki prečka mesto, je bila skozi zgodovina pomembna za razvoj mesta, od prazgodovinskih naselbin do današnje sodobne prestolnice. Ljubljana je znana po svoji univerzi, ki je bila ustanovljena leta 1919, in po številnih muzejih, gledališčih in knjižnicah.",
        "question": "Katera reka prečka mesto Ljubljana?",
    }
]

print(model.predict(data))
