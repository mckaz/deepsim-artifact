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


MODEL_TYPE = "comments"
LABEL_CHOICE = "TOP"
USE_OTHER_TYPE = False

DATA_FILE='./type-data.json'

## LOAD TOKENIZER    
with open('tokenizers/{}_tokenizer.pickle'.format(MODEL_TYPE), 'rb') as handle:
    lang_tokenizer = pickle.load(handle)

## LOAD LABELS
with open('labels/{}_{}_idx_to_label.pkl'.format(MODEL_TYPE, LABEL_CHOICE), 'rb') as f:    
    idx_to_label = pickle.load(f)

## LOAD LABELS
with open('labels/{}_{}_label_to_idx.pkl'.format(MODEL_TYPE, LABEL_CHOICE), 'rb') as f:    
    label_to_idx = pickle.load(f)
    
## LOAD SAVED MODEL
model = load_model('models/{}_{}_model.h5'.format(MODEL_TYPE, LABEL_CHOICE))

def evaluate_test_set():
    if MODEL_TYPE == "names":
        test_dataset, _ = setup_model.create_names_dataset(DATA_FILE, True)
    elif MODEL_TYPE == "comments":
        test_dataset, _ = setup_model.create_comments_dataset(DATA_FILE, True)
    elif MODEL_TYPE == "nc":
        test_dataset, _ = setup_model.create_nc_dataset(DATA_FILE, True)
        ### TODO: twin model tests

#<<<<<<< HEAD

#    test_ds = setup_model.prepare_data(test_dataset, lang_tokenizer, label_to_idx, USE_OTHER_TYPE, True)
#    model.evaluate(test_ds)#, batch_size=128)
#=======
    test_xs, test_ys = setup_model.prepare_data(test_dataset, lang_tokenizer, label_to_idx, USE_OTHER_TYPE, True)
    y_preds = model.predict(test_xs)
    y_preds_bool = np.argmax(y_preds, axis=1)
    print(classification_report(test_ys, y_preds_bool))
#>>>>>>> 3fdefbd443722b975021e5ba98d162a65c918134



### Methods for running on a single input
def run_model(inp):
    x = model.predict(lang_tokenizer.texts_to_sequences([inp]))
    print(idx_to_label[np.argmax(x)])


def run_twin_model(in1, in2):
    emb1, emb2 = tf.keras.preprocessing.sequence.pad_sequences(lang_tokenizer.texts_to_sequences([in1, in2]), maxlen = 550).squeeze()
    print(model.predict([[emb1], [emb2]]))
