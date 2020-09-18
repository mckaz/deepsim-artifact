import setup_model
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping


DATA_FILE='./type-data.json'

## LABEL_CHOICE:
##   -"TOP" if picking the top most occuring labels in the dataset
##   -"PROG" if picking the labels occuring in at least MIN_PROGNUM_LABELS programs
LABEL_CHOICE = "TOP"

## Number of labels to pick from.
LABEL_NUM = 10000

EPOCHS=30

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

DATA_SIZE = 500000

if USE_OTHER_TYPE:
    other_tag = "_OTHER_"
else:
    other_tag = ""

#dataset, prog_type_dict = setup_model.create_nc_dataset(DATA_FILE)
if input_type == "names":
    dataset, prog_type_dict = setup_model.create_names_dataset(DATA_FILE)
elif input_type == "comments":
    dataset, prog_type_dict = setup_model.create_comments_dataset(DATA_FILE)
elif input_type == "nc":
    dataset, prog_type_dict = setup_model.create_nc_dataset(DATA_FILE, DELIMITER)
else:
    raise Exception("Got unexpected input_type of {}".format(input_type))

lang_tokenizer = setup_model.create_tokenizer(dataset)
vocab_size = max(lang_tokenizer.index_word.keys())
## SAVE TOKENIZER
with open('tokenizers/twin_{}_tokenizer.pickle'.format(input_type), 'wb') as handle:
    pickle.dump(lang_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
## LOAD TOKENIZER    
#with open('tokenizers/twin_{}_tokenizer.pickle'.format(input_type), 'rb') as handle:
#    lang_tokenizer = pickle.load(handle)
label_to_idx, idx_to_label = setup_model.create_labels(dataset, prog_type_dict, LABEL_CHOICE, USE_OTHER_TYPE, LABEL_NUM, MIN_PROGNUM_LABELS)
setup_model.save_labels(label_to_idx, 'twin_{}_{}_{}label_to_idx'.format(input_type, LABEL_CHOICE, other_tag))
setup_model.save_labels(idx_to_label, 'twin_{}_{}_{}idx_to_label'.format(input_type, LABEL_CHOICE, other_tag))
num_labels = len(label_to_idx)
train_dataset, dev_dataset = setup_model.split_train_dev(dataset)
#train_ds = setup_model.prepare_data(train_dataset, lang_tokenizer, label_to_idx, USE_OTHER_TYPE)
#dev_ds = setup_model.prepare_data(dev_dataset, lang_tokenizer, label_to_idx, USE_OTHER_TYPE)
#train_ds = setup_model.get_twin_data(train_dataset, DATA_SIZE, lang_tokenizer, label_to_idx, USE_OTHER_TYPE)
train_ds = setup_model.get_prog_twin_data(train_dataset, DATA_SIZE, lang_tokenizer, label_to_idx, USE_OTHER_TYPE, 90)
dev_ds = setup_model.get_prog_twin_data(dev_dataset, int(DATA_SIZE * 0.1), lang_tokenizer, label_to_idx, USE_OTHER_TYPE, 90)
input_dim = len(train_ds[0][0][0])


def get_twin_net(input_dim):
    left_input = tf.keras.Input(input_dim)
    right_input = tf.keras.Input(input_dim)

    encoder_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, 128, mask_zero=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_labels + 1, activation='sigmoid')
    ])

    # Generate the encodings (feature vectors) for the two images
    encoded_l = encoder_model(left_input)
    encoded_r = encoder_model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = tf.keras.layers.Dense(1,activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    twin_net = tf.keras.models.Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return twin_net



model = get_twin_net(input_dim)

optimizer = tf.keras.optimizers.Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer, metrics=['accuracy'])
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
model.fit(x=train_ds[0], y=train_ds[1], epochs=EPOCHS, validation_data=dev_ds)#,callbacks=[es])

model.save('models/twin__{}_{}_{}_{}_{}_PROG_model.h5'.format(input_type, LABEL_CHOICE, EPOCHS, other_tag, DATA_SIZE))

    



