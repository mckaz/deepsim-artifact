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
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import backend as K


app = Flask(__name__)



DATA_FILE='../type-data.json'

## LOAD TOKENIZER    
#with open('tokenizers/twin_nc_tokenizer.pickle', 'rb') as handle:
with open('tokenizers/twin_names_tokenizer.pickle', 'rb') as handle:
    lang_tokenizer = pickle.load(handle)
    
## LOAD SAVED MODEL
model = load_model('models/twin__nc_TOP__PROG_model.h5')#twin__names_TOP__500000_PROG_model.h5')#

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
bert_model = TFRobertaModel.from_pretrained("microsoft/codebert-base")

bert_cache = {}


# in1 and in2 are lists of strings to be run through twin model
# i.e., in1[0] compared with in2[0], in1[1] compared with in2[1]...
def run_twin_model(in1, in2):
    in1 = tf.keras.preprocessing.sequence.pad_sequences(lang_tokenizer.texts_to_sequences(in1), maxlen = 550)#.squeeze()
    in2 = tf.keras.preprocessing.sequence.pad_sequences(lang_tokenizer.texts_to_sequences(in2), maxlen = 550)#.squeeze()
    pred = model.predict([in1, in2])
    ## pred is Array<List<List<Float>>>, e.g.:
    #array([[0.94821054],
    #   [0.01682347]], dtype=float32)

    return pred

def run_bert_model(sources, var1_locs, var2_locs):
    source1 = sources[0]
    source2 = sources[1]
    #begin_var1 = int(var1_locs[0])
    #end_var1 = int(var1_locs[1])
    #begin_loc2 = int(var2_locs[0])
    #end_loc2 = int(var2_locs[1])

    tok_inds1 = get_tok_ind(source1, var1_locs)
    tok_inds2 = get_tok_ind(source2, var2_locs)


    if source1 in bert_cache:
        list_of_vecs = [bert_cache[source1][0][0][index + 1] for index in tok_inds1] ## + 1 because of CLS token
        vec1 = tf.reduce_mean(list_of_vecs, axis=0) ## get average of all relevant vectors        
        #vec1 = bert_cache[source1][0][0][tok_index1 + 1]
    else:
        #tok1 = tokenizer(source1, return_tensors="tf")
        tok1 = tokenizer.tokenize(source1)
        tok1 = tok1[:510] ## 512 is max length accepted by model
        enc1 = tokenizer.encode(tok1, return_tensors="tf")
        res1 = bert_model(enc1)
        list_of_vecs1 = [res1[0][0][index + 1] for index in tok_inds1]
        vec1 = tf.reduce_mean(list_of_vecs1, axis=0) ## get average of all relevant vectors        
        #vec1 = res1[0][0][tok_index1 + 1]
        bert_cache[source1] = res1
        ## Explanation: res1[0] is last hidden state, of shape (batch size, (tokenized) sequence length, 768)
        ## res1[0][0] needed because batch should be of size 1, and we're interested in that 1 input.
        ## res1[0][0][tokindex + 1] gets vector associated with token of interest. +1 needed due to CLS token at start.

    if source2 in bert_cache:
        list_of_vecs2 = [bert_cache[source2][0][0][index + 1] for index in tok_inds2]
        vec2 = tf.reduce_mean(list_of_vecs2, axis=0)
        #vec2 = bert_cache[source2][0][0][tok_index2 + 1]
    else:
        #tok2 = tokenizer(source2, return_tensors="tf")
        tok2 = tokenizer.tokenize(source2)
        tok2 = tok2[:510] ## 512 is max length accepted by model
        enc2 = tokenizer.encode(tok2, return_tensors="tf")
        res2 = bert_model(enc2)
        list_of_vecs2 = [res2[0][0][index + 1] for index in tok_inds2]
        vec2 = tf.reduce_mean(list_of_vecs2, axis=0)
        #vec2 = res2[0][0][tok_index2 + 1]
        bert_cache[source2] = res2

    return cosine_similarity([vec1], [vec2])[0][0]
    #tok_in1 = tokenizer(in1, padding='max_length', max_length=90, return_tensors="tf")
    #tok_in2 = tokenizer(in2, padding='max_length', max_length=90, return_tensors="tf")
    #out_1 = K.mean(bert_model(tok_in1)[0], axis=1)
    #out_2 = K.mean(bert_model(tok_in2)[0], axis=1)
    #return cosine_similarity(out_1, out_2)[0][0]

## Takes source [String] code and location loc [Integer] of relevant var,
## and returns the index [Integer] of the token corresponding to that var when source is tokenized.
def get_tok_ind(source, var_locs):
    tokenized = tokenizer.tokenize(source)
    begin_loc = int(var_locs[0]) ## beginning of var to find
    end_loc = int(var_locs[1]) ## end of var to find
    print("Here with begin_loc {} and end_loc {}".format(begin_loc, end_loc))
    inds = [] ## list of indices to be returned
    next_char = 0 ## next char to look at in the source string

    for i in range(len(tokenized)):
        token_end = next_char + len(tokenized[i]) - 1 ## end of current token
        print("Observing token {} with next_char {} and token_end {}".format(tokenized[i], next_char, token_end))
        if ((token_end >= begin_loc) and (token_end <= end_loc)):
            inds.append(i)
            print("Using vector associated with token {}".format(tokenized[i]))
        elif (token_end > end_loc):
            print("Returning now!")
            return inds
        next_char = token_end + 1
    raise Exception("Given locations {} and {}, which is beyond the range of source code {}".format(begin_loc, end_loc, source))

    #for i in range(len(tokenized)):
    #    if ((seen + len(tokenized[i])) > loc):
            ## if the current token takes us past loc,
            ## we've found the relevant index i.
    #        print("Using vector associated with token {}".format(tokenized[i]))
    #        return i
    #    seen = seen + len(tokenized[i])
    #raise Exception("Given location {}, which is beyond the range of source code {}".format(loc, source))

def get_average(scores):
    sum = 0
    for i in scores:
        sum = sum + i[0]
    return sum / len(scores)


    
## Web API defined below.

@app.route("/")
def receive():
    method = request.args.get("method")
    if (method == "twin"):
        words = request.args.getlist("words")
        ## First word is the one we want to compare against all remaining words
        words2 = words[1:]
        words1 = [words[0]] * len(words2)
        scores = run_twin_model(words1, words2)
        ret = np.mean(scores)
    elif (method == "bert"):
        sources = request.args.getlist("sources")
        var1_locs = request.args.getlist("var1_locs")
        var2_locs = request.args.getlist("var2_locs")
        ret = run_bert_model(sources, var1_locs, var2_locs)
        #if ((len(words1) != 1) or (len(words2) != 1)):
        #    raise Exception("Unexpected number of words in {} or {}".format(words1, words2))
        #ret = run_bert_model(words1[0], words2[0])
    else:
         raise Exception("Unexpected method in request: {}".format(method))   

    #av = get_average(scores)
    #in1 = request.args.get("in1")
    #in2 = request.args.get("in1")
    return str(ret)




if __name__ == "__main__":
    app.run()
