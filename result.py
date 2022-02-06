from Process import data_process
from Train import Train
from prediction import get_response

'''
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input Dataset")
ap.add_argument("-i", "--intent", required=True, help="path to input intent json file")
ap.add_argument("-m", "--model_name", required=True, help="model type")

args = vars(ap.parse_args())
'''


def result():
    # call dta process with model type ,load some variable for train
    bert_model,train_labels,train_dataloader,tokenizer=data_process("bert", "chitchat.csv")
    # call activate train function and fine_tune the model
    trained_model=Train.train(bert_model,train_labels,train_dataloader,"bert")
    # save the train model
    Train.save_trained("bert",trained_model)
    message = "ما اسمك"
    get_response(message,tokenizer,"data.json","bert")


result()