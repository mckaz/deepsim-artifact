# Run with: python -m flask run
from flask import Flask, jsonify, request
import json
import numpy as np
import math
import tensorflow as tf
import tensorflow_text as text
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
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
from sklearn.manifold import TSNE
import time




app = Flask(__name__)



DATA_FILE='../type-data.json'

## LOAD TOKENIZER    
#with open('tokenizers/twin_nc_tokenizer.pickle', 'rb') as handle:
#with open('tokenizers/twin_names_tokenizer.pickle', 'rb') as handle:
#    lang_tokenizer = pickle.load(handle)
    
## LOAD SAVED MODEL
#model = load_model('models/twin__nc_TOP__PROG_model.h5')#twin__names_TOP__500000_PROG_model.h5')#
arg_model = load_model('bert_twin_data/models/twin_bert_arg_200_84349_model.h5')
ret_model = load_model('bert_twin_data/models/twin_bert_ret_200_99167_model.h5')


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
bert_model = TFRobertaModel.from_pretrained("microsoft/codebert-base")

bert_cache = {}
vector_cache = {}
names_cache = {}
types_cache = {}

state = "open"
running_list_of_vecs = []
max_seq_length = 510


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

#def vectorize_arg(source, locs):
#    tok_inds = get_tok_ind(source, locs)
#    if source in bert_cache:
#        list_of_vecs = [bert_cache[source][0][0][index+1] for index in tok_inds if index <= 510]
#    else:
#        tok = tokenizer.tokenize(source)
#        tok = tok[:510]
#        enc = tokenizer.encode(tok, return_tensors="tf")
#        res = bert_model(enc)
#        list_of_vecs = [res[0][0][index+1] for index in tok_inds if index <= 510]
#        bert_cache[source] = res
#    return tf.reduce_mean(list_of_vecs, axis=0)

def split_into_windows(seq, window_size):
    windows = []
    num_windows = math.ceil(2*len(seq) / window_size) - 1
    if num_windows == 0:
        return [seq]
    for i in range(num_windows):
        start = int(i * (window_size / 2))
        window = seq[start:(start + window_size)] ## python automatically stops at end if start+window_size is past len(seq)
        windows.append(window)
    return windows

## Not only calls BERT model, but also splits the input into equal sized,
## max_seq_length token chunks, with a sliding window of max_seq_length/2, averaging overlap.
def run_bert_model(source):
    tok = tokenizer.tokenize(source)
    #print("Running for total of {} tokens.".format(len(tok)))
    if len(tok) <= max_seq_length:
        enc = tokenizer.encode(tok, return_tensors="tf")
        res = bert_model(enc)
        bert_cache[source] = res[0][0]
        return res[0][0]
    windows = split_into_windows(tok, max_seq_length)
    #print("Total number of windows: {}".format(len(windows)))
    results = []
    for w in windows:
        enc = tokenizer.encode(w, return_tensors="tf")
        res = bert_model(enc)
        results.append(res[0][0]) ## has shape (max_seq_length, 768)
    cls_results = []
    code_results = []
    fin_results = []
    ## First, take care of first half of first vector.
    for i in range(int(max_seq_length/2)):
        code_results.append(results[0][i+1]) ## i + because we're shifting past the CLS vector
    for window_num in range(len(windows)-1):
        cls_results.append(results[window_num][0])
        fin_results.append(results[window_num][-1])
        for i in range(int(max_seq_length/2), max_seq_length):
            #print("Averaging vector {} from window {} with vector {} from window {}.".format(i+1, window_num, i-int(max_seq_length/2)+1, window_num+1))
            vecs = [results[window_num][i+1], results[window_num+1][i - int(max_seq_length/2) + 1]]
            av = tf.reduce_mean(vecs, axis=0)
            code_results.append(av)
    ## Finally, add second half of last window.
    #print("Length of final window: {}".format(len(windows[-1])))
    for i in range(int(max_seq_length/2), len(windows[-1])):
        #print("Appending vector {} from window -1.".format(i+1))
        code_results.append(results[-1][i+1])
    ## Add CLS, FIN for last window
    cls_results.append(results[-1][0])
    fin_results.append(results[-1][-1])
    ## Some sanity checks below.
    if len(cls_results) != len(windows):
        raise Exception("Expected {} CLS vectors, got {}.".format(len(windows), len(cls_results)))
    elif len(fin_results) != len(windows):
        raise Exception("Expected {} FIN vectors, got {}.".format(len(windows), len(fin_results)))
    elif len(tok) != len(code_results):
        raise Exception("Expected {} code vectors, got {}.".format(len(tok), len(code_results)))
    cls_av = tf.reduce_mean(cls_results, axis=0)
    fin_av = tf.reduce_mean(fin_results, axis=0)
    final_ret = [cls_av]
    final_ret += code_results
    final_ret.append(fin_av)
    bert_cache[source] = final_ret
    return final_ret 

def vectorize_locs(source, locs):
    tok_inds = get_tok_ind(source, locs)
    if source in bert_cache:
        #list_of_vecs = [bert_cache[source][0][0][index+1] for index in tok_inds if index <= 510]
        list_of_vecs = [bert_cache[source][index+1] for index in tok_inds]
    else:
       # tok = tokenizer.tokenize(source)
       # tok = tok[:510]
       # enc = tokenizer.encode(tok, return_tensors="tf")
       # res = bert_model(enc)
       # list_of_vecs = [res[0][0][index+1] for index in tok_inds if index <= 510]
       # bert_cache[source] = res
        res = run_bert_model(source)
        list_of_vecs = [res[index+1] for index in tok_inds]
    return list_of_vecs

## Takes source [String] code and location loc [Integer] of relevant var,
## and returns the index [Integer] of the token corresponding to that var when source is tokenized.
def get_tok_ind(source, var_locs):
    tokenized = tokenizer.tokenize(source)
    begin_loc = int(var_locs[0]) ## beginning of var to find
    end_loc = int(var_locs[1]) ## end of var to find
    #print("Here with begin_loc {} and end_loc {}".format(begin_loc, end_loc))
    inds = [] ## list of indices to be returned
    next_char = 0 ## next char to look at in the source string
    next_loc = 2 ## next begin_loc in var_locs to look at

    for i in range(len(tokenized)):
        token_end = next_char + len(tokenized[i]) - 1 ## end of current token
        #print("Observing token {} with next_char {} and token_end {}".format(tokenized[i], next_char, token_end))
        if ((token_end >= begin_loc) and (token_end < end_loc)):
            inds.append(i)
            #print("Using vector associated with token {}".format(tokenized[i]))
        elif (token_end >= end_loc):
            inds.append(i)
            #print("Using vector associated with token {}".format(tokenized[i]))
            if next_loc < len(var_locs):
                ## if there are still begin/end location pairs, move to next pair
                begin_loc = int(var_locs[next_loc])
                end_loc = int(var_locs[next_loc + 1])
                next_loc = next_loc + 2
            else:
                ## otherwise, we've collected tokens associated with all locations. Return.
                return inds
            #elif (token_end > end_loc):
            #print("Returning now!")
            #return inds
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

def tsne_plot(obj_ids):
    vecs = [] ## list of BERT embeddings
    labels = [] ## list of names of vars/methods
    ts = [] ## list of types
    for obj_id in obj_ids:
        obj_id = int(obj_id)
        if (obj_id not in vector_cache): #or (obj_id not in names_cache) or (obj_id not in types_cache):
            #print("Could not find object id {}".format(obj_id))
            continue
        vecs.append(vector_cache[obj_id])
        labels.append(names_cache[obj_id]) ## TODO: Implement names cache
        ts.append(types_cache[obj_id]) ## TODO: Implement types cache
    uniq_ts = list(set(ts))

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23) ## TODO: figure out good parameters

    new_values = tsne_model.fit_transform(vecs)
    cmap = plt.get_cmap("tab20")

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    x = np.array(x)
    y = np.array(y)
    labels = np.array(labels)
    ts = np.array(ts)
        
    plt.figure(figsize=(16,16))

    for t in np.unique(ts):
        ix = np.where(ts == t)
        plt.scatter(x[ix], y[ix], c = cmap(uniq_ts.index(t)), label = t, s=100)
    #for i in range(len(x)):
    #    plt.annotate(labels[i],
    #                 xy = (x[i], y[i]),
    #                 xytext=(5,2),
    #                 textcoords='offset points',
    #                 ha='right',
    #                 va='bottom')

    #for i in range(len(x)):
    #    plt.scatter(x[i],y[i],c=cmap(uniq_ts.index(ts[i])))
    #    plt.annotate(labels[i],
    #                 xy=(x[i], y[i]),
    #                 xytext=(5, 2),
    #                 textcoords='offset points',
    #                 ha='right',
    #                 va='bottom')

    plt.legend()
    plt.savefig('figures/type_plot.png')
    #plt.show()

def get_similarity(id1, id2, kind1, kind2):
    in1 = vector_cache[id1]
    in2 = vector_cache[id2]
    if (kind1 != kind2) or (kind1 == "var"):
        return cosine_similarity([in1], [in2])[0][0]
    elif (kind1 == "arg"):
        ret = arg_model.predict([np.reshape(in1, [1, 768]), np.reshape(in2, [1,768])])
        return ret[0][0]
    elif (kind1 == "ret"):
        ret = ret_model.predict([np.reshape(in1, [1, 768]), np.reshape(in2, [1,768])])
        return ret[0][0]
    else:
        raise Exception("Unexpected kind {}".format(kind1))

## Web API defined below.

@app.route("/")
def receive():
    global state
    global running_list_of_vecs
    action = request.args.get("action")
    if (state != "open"):
        if (action != "bert_vectorize") or (request.args.get("category") != "var"):
            raise Exception("State is not open, but received action {} and category {}".format(action, request.args.get("category")))
    if (action == "twin"):
        words = request.args.getlist("words")
        ## First word is the one we want to compare against all remaining words
        words2 = words[1:]
        words1 = [words[0]] * len(words2)
        scores = run_twin_model(words1, words2)
        ret = np.mean(scores)
    elif (action == "bert_vectorize"):
        category = request.args.get("category")
        if (category == "arg") or (category == "ret"):
            object_id = int(request.args.get("object_id"))
            if object_id not in vector_cache: ## if it is in cache, we're already done, nothing to do
                source = request.args.get("source")
                locs = request.args.getlist("locs")
                list_of_vecs = vectorize_locs(source, locs)
                #print("Caching object_id {}".format(object_id))
                #print("ABOUT TO ADD {} TO VECTOR_CACHE".format(object_id))
                vector_cache[object_id] = tf.reduce_mean(list_of_vecs, axis=0)
                ret = True
        elif (category == "var"):
            object_id = int(request.args.get("object_id"))
            average = request.args.get("average")
            if (state != "open") and (object_id != state):
                raise Exception("Conflicting states.")
            if (average == "true"):
                vector_cache[object_id] = tf.reduce_mean(running_list_of_vecs, axis=0)
                state = "open"
                running_list_of_vecs = []
                ret = True
            elif object_id not in vector_cache:
                state = object_id
                source = request.args.get("source")
                locs = request.args.getlist("locs")
                #print("Given locs {}".format(locs))
                list_of_vecs = vectorize_locs(source, locs)
                running_list_of_vecs += list_of_vecs
                ret = True
        else:
            raise Exception("Unexpected category: {}".format(category))
    elif (action == "get_similarity"):
        start = time.time()
        id1 = int(request.args.get("id1"))
        id2 = int(request.args.get("id2"))
        kind1 = request.args.get("kind1")
        kind2 = request.args.get("kind2")
        #print("Looking up ids {} and {}".format(id1, id2))
        if (id1 not in vector_cache) or (id2 not in vector_cache):
            raise Exception("Could not find given object_id in vector_cache.")
        ret = get_similarity(id1, id2, kind1, kind2)
        end = time.time()
        print("Total time taken for {} and {}: {}".format(id1, id2, end-start))
        #ret = cosine_similarity([vector_cache[id1]], [vector_cache[id2]])[0][0]
    elif (action == "add_info"):
        object_id = int(request.args.get("object_id"))
        typ = request.args.get("type")
        name = request.args.get("name")
        names_cache[object_id] = name
        types_cache[object_id] = typ
        ret = True
    elif (action == "visualize"):
        id_list = request.args.getlist("id_list")
        tsne_plot(id_list)
        ret = True
    else:
         raise Exception("Unexpected action in request: {}".format(action))   

    return str(ret)




if __name__ == "__main__":
    app.run()
