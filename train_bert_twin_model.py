import setup_model
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import numpy as np
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.keras import backend as K
from transformers import RobertaTokenizer, TFRobertaModel, RobertaConfig

DATA_TYPE = "arg" # "arg" or "ret"
DATA_SIZE = 84349
EPOCHS = 10

data = np.load('bert_twin_data/data_{}_{}.npz'.format(DATA_TYPE, DATA_SIZE))

def get_twin_net():
    left_input = tf.keras.Input(768, dtype='float32') 
    right_input = tf.keras.Input(768, dtype='float32')

    L1_layer = tf.keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([left_input, right_input])

    concat = tf.keras.layers.Concatenate([left_input, right_input, L1_distance])

    prediction = tf.keras.layers.Dense(1,activation='sigmoid')(concat)

    twin_net = tf.keras.models.Model(inputs=[left_input,right_input],outputs=prediction)

    return twin_net


model = get_twin_net()

optimizer = tf.keras.optimizers.Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer, metrics=['accuracy'])
model.fit(x=[data['input1'], data['input2']], y=data['output'], epochs=EPOCHS)#, validation_data=dev_ds)

model.save('models/twin_bert_{}_{}_{}_model.h5'.format(input_type, LABEL_CHOICE, EPOCHS, other_tag, DATA_SIZE))
