import json
import re
import torch 
import numpy as np 
from Train import Train
import random
import os 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time
import sys



def check_trained_model(model_name):
    if(model_name=="bert"):
        model_trained = torch.load("model/bert-arb")
        print(sys.getsizeof(model_trained))
    elif(model_name=="roberta"):
        model_trained=torch.load("model/roberta-arb")
        print(sys.getsizeof(model_trained))
    elif(model_name=="distilbert"):
        model_trained=torch.load("model/distilbert")
        print(sys.getsizeof(model_trained))
    return(model_trained)

def get_prediction(str,tokenizer,model_name):
    le_encode = torch.load("encoding/le")
    device = torch.device("cuda")
    max_seq_len=8
    str = re.sub(r"[^a-zA-Z ]+", "", str)
    test_text = [str]
    tokens_test_data = tokenizer(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data["input_ids"])
    test_mask = torch.tensor(tokens_test_data["attention_mask"])
    preds = None
    start = time.time()
    with torch.no_grad():
        train_model = check_trained_model(model_name)
        preds = train_model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    stop = time.time()
    print(f"Prediction time: {stop - start}s")
    print("Intent Identified: ", le_encode.inverse_transform(preds)[0])
    return le_encode.inverse_transform(preds)[0]


def get_response(message,tokenizer,intents,model_name): 
    intent = get_prediction(message,tokenizer,model_name)
    data = json.load(open (intents))
    for i in data['intents']: 
        if i["tag"] == intent:
            result = random.choice(i["responses"])
    print(f"Response : {result}")
    return "Intent: "+ intent + '\n' + "Response: " + result