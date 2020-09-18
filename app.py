# Run with: python -m flask run
from flask import Flask, jsonify, request
import json
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.keras.models import load_model
import setup_model
from sklearn.metrics import classification_report


app = Flask(__name__)



DATA_FILE='../type-data.json'

## LOAD TOKENIZER    
#with open('tokenizers/twin_nc_tokenizer.pickle', 'rb') as handle:
with open('tokenizers/twin_names_tokenizer.pickle', 'rb') as handle:
    lang_tokenizer = pickle.load(handle)
    
## LOAD SAVED MODEL
model = load_model('models/twin__names_TOP__500000_PROG_model.h5')#twin__nc_TOP__PROG_model.h5')


# in1 and in2 are lists of strings to be run through twin model
# i.e., in1[0] compared with in2[0], in1[1] compared with in2[1]...
def run_twin_model(in1, in2):
    in1 = tf.keras.preprocessing.sequence.pad_sequences(lang_tokenizer.texts_to_sequences(in1), maxlen = 90)#.squeeze()
    in2 = tf.keras.preprocessing.sequence.pad_sequences(lang_tokenizer.texts_to_sequences(in2), maxlen = 90)#.squeeze()
    pred = model.predict([in1, in2])
    ## pred is Array<List<List<Float>>>, e.g.:
    #array([[0.94821054],
    #   [0.01682347]], dtype=float32)

    return pred


def get_average(scores):
    sum = 0
    for i in scores:
        sum = sum + i[0]
    return sum / len(scores)


    
## Web API defined below.

@app.route("/")
def receive():
    words = request.args.getlist("words")
    ## First word is the one we want to compare against all remaining words
    words2 = words[1:]
    words1 = [words[0]] * len(words2)
    print("Received: ")
    print(words)
    print(type(words))
    scores = run_twin_model(words1, words2)
    av = np.mean(scores)
    #av = get_average(scores)
    #in1 = request.args.get("in1")
    #in2 = request.args.get("in1")
    return str(av)




if __name__ == "__main__":
    app.run()
