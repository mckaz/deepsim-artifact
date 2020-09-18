import setup_model
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.keras import backend as K
from transformers import RobertaTokenizer, TFRobertaModel

bert_model_path = './pretrained_codebert'

DATA_FILE='./type-data.json'

## LABEL_CHOICE:
##   -"TOP" if picking the top most occuring labels in the dataset
##   -"PROG" if picking the labels occuring in at least MIN_PROGNUM_LABELS programs
LABEL_CHOICE = "TOP"

## Number of labels to pick from.
LABEL_NUM = 1000

## When LABEL_CHOICE is "PROG", this is the minimum number of programs a type should occur
## in for it to be used as a label.
MIN_PROGNUM_LABELS = 5

## When true, special type "#other#" will be used for all types out side of core labels.
USE_OTHER_TYPE = False

## Delimiter to separate names from comments
DELIMITER = "^"

## Should be "names", "comments" or "nc" (which is combination of both).
## Gives type of input to train network on.
input_type = "names"

DATA_SIZE = 100000

if USE_OTHER_TYPE:
    other_tag = "_OTHER_"
else:
    other_tag = ""

if input_type == "names":
    dataset, prog_type_dict = setup_model.create_names_dataset(DATA_FILE)
elif input_type == "comments":
    dataset, prog_type_dict = setup_model.create_comments_dataset(DATA_FILE)
elif input_type == "nc":
    dataset, prog_type_dict = setup_model.create_nc_dataset(DATA_FILE, DELIMITER)
else:
    raise Exception("Got unexpected input_type of {}".format(input_type))

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
label_to_idx, idx_to_label = setup_model.create_labels(dataset, prog_type_dict, LABEL_CHOICE, USE_OTHER_TYPE, LABEL_NUM, MIN_PROGNUM_LABELS)
setup_model.save_labels(label_to_idx, 'twinbert_names_{}_{}label_to_idx'.format(LABEL_CHOICE, other_tag))
setup_model.save_labels(idx_to_label, 'twinbert_names_{}_{}idx_to_label'.format(LABEL_CHOICE, other_tag))
num_labels = len(label_to_idx)




    

def get_twin_net(input_dim):
    bert_model = TFRobertaModel.from_pretrained(bert_model_path, 'roberta-base')

    left_input = tf.keras.Input(input_dim, dtype='int64')
    right_input = tf.keras.Input(input_dim, dtype='int64')

    encoded_l = bert_model(left_input)[0]
    encoded_r = bert_model(right_input)[0]

    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = tf.keras.layers.Dense(1,activation='sigmoid')(L1_distance)

    twin_net = tf.keras.models.Model(inputs=[left_input,right_input],outputs=prediction)
    return twin_net



model = get_twin_net(input_dim)

optimizer = tf.keras.optimizers.Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer, metrics=['accuracy'])


