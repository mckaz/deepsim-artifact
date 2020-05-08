## TODO
# 1. Do a train/dev/test split?
# 2. Prevent overfitting by adding a dropout layer

import setup_model
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
import sys
import os

## parameters
tf.compat.v1.flags.DEFINE_integer("emb_dim", 128, "Number of embedding dims for character)")
tf.compat.v1.flags.DEFINE_integer("batch_size", 128, "Batch size")
tf.compat.v1.flags.DEFINE_integer("epochs", 10, "Number of epochs")
tf.compat.v1.flags.DEFINE_float("lr", 1e-4, "Which learning rate to start with")
tf.compat.v1.flags.DEFINE_integer("hidden_units_lstm", 64, "Number of LSTM hidden units")
tf.compat.v1.flags.DEFINE_integer("hidden_units_dense", 64, "Number of relu units")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
print("emb_dim={}".format(FLAGS.emb_dim))
print("batch_size={}".format(FLAGS.batch_size))
print("epochs={}".format(FLAGS.epochs))
print("lr={}".format(FLAGS.lr))
print("hidden_units_lstm={}".format(FLAGS.hidden_units_lstm))
print("hidden_units_dense={}".format(FLAGS.hidden_units_dense))
print("")
#E

DATA_FILE='./type-data.json'


## LABEL_CHOICE:
##   -"TOP" if picking the top most occuring labels in the dataset
##   -"PROG" if picking the labels occuring in at least MIN_PROGNUM_LABELS programs
LABEL_CHOICE = "PROG"

## Number of labels to pick from.
LABEL_NUM = 100

## When LABEL_CHOICE is "PROG", this is the minimum number of programs a type should occur
## in for it to be used as a label.
MIN_PROGNUM_LABELS = 5

## When true, special type "#other#" will be used for all types out side of core labels.
USE_OTHER_TYPE = False

if USE_OTHER_TYPE:
    other_tag = "_OTHER_"
else:
    other_tag = ""

    
dataset, prog_type_dict = setup_model.create_names_dataset(DATA_FILE)
lang_tokenizer = setup_model.create_tokenizer(dataset)
vocab_size = max(lang_tokenizer.index_word.keys())
# SAVE TOKENIZER
with open('tokenizers/names_tokenizer.pickle', 'wb') as handle:
    pickle.dump(lang_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# LOAD TOKENIZER    
#with open('tokenizers/names_tokenizer.pickle', 'rb') as handle:
#    lang_tokenizer = pickle.load(handle)
label_to_idx, idx_to_label = setup_model.create_labels(dataset, prog_type_dict, LABEL_CHOICE, USE_OTHER_TYPE, LABEL_NUM, MIN_PROGNUM_LABELS)
setup_model.save_labels(label_to_idx, 'names_{}_{}label_to_idx'.format(LABEL_CHOICE, other_tag))
setup_model.save_labels(idx_to_label, 'names_{}_{}idx_to_label'.format(LABEL_CHOICE, other_tag))
num_labels = len(label_to_idx)
train_dataset, dev_dataset = setup_model.split_train_dev(dataset)
train_ds = setup_model.prepare_data(train_dataset, lang_tokenizer, label_to_idx, USE_OTHER_TYPE)
dev_ds = setup_model.prepare_data(dev_dataset, lang_tokenizer, label_to_idx, USE_OTHER_TYPE)
# shuffle and create batches
batch_size = FLAGS.batch_size
train_ds = train_ds.shuffle(20000).batch(batch_size)
dev_ds = dev_ds.shuffle(20000).batch(batch_size)



## initialize/compile model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, FLAGS.emb_dim, mask_zero=True),
    #tf.keras.layers.Masking(mask_value=0.,input_shape=(66, vocab_size)),
    tf.keras.layers.LSTM(FLAGS.hidden_units_lstm),
    tf.keras.layers.Dense(FLAGS.hidden_units_dense, activation='relu'),
    tf.keras.layers.Dense(num_labels + 1)
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(FLAGS.lr),
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=FLAGS.epochs, validation_data=dev_ds)


model.save('models/names_{}_{}_{}_{}_{}_{}_{}_{}_model.h5'.format(LABEL_CHOICE, other_tag, FLAGS.emb_dim, FLAGS.batch_size, FLAGS.epochs, FLAGS.lr, FLAGS.hidden_units_lstm, FLAGS.hidden_units_dense))

## LOAD SAVED MODEL
#model = load_model('models/names_{}_model.h5'.format(LABEL_CHOICE))

## test out the model on ID name "str"
#x = model.predict(lang_tokenizer.texts_to_sequences(["str"]))

## convert max output back into label, print
#print(idx_to_label[np.argmax(x)])

