import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import random
from tensorflow.keras import models
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.keras import activations


# model
class LSTM_rnn(keras.Model):

    def __init__(self, vocabulary_size, embedding_dimension, rnn_units):
        super().__init__(self)
        self.embedding = Embedding(vocabulary_size, embedding_dimension)
        self.lstm = LSTM(rnn_units, activation="tanh", return_sequences=True, return_state=True, unit_forget_bias=True)
        self.dense = Dense(vocabulary_size)
        self.dropout = Dropout(0.2)
        #self.softmax = Activation(activation="softmax")

    def call(self, inputs, memory_states=None, carry_states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
    
        if memory_states is None and carry_states is None:
            memory_states, carry_states = self.lstm.get_initial_state(x)
            
        x, memory_states, carry_states = self.lstm(x, initial_state=[memory_states, carry_states], training=training)
        x = self.dense(x, training=training)
        #x = self.softmax(x, training=training)
        #x = self.dropout(x, training=training)

        if return_state==True:
            return x, memory_states, carry_states
        else:
            return x

# predictor
class OneStepForecast(keras.Model):

    def __init__(self, model, int2char, char2int, temperature=1.0):
        super().__init__()
        self.model = model
        self.int2char = int2char
        self.char2int = char2int
        self.temperature = temperature

        # Handly generated vocabulary had one character less than the one generated using StringLookup
        # The reason for that is that StringLookup generates also an ["UNK"] char - we don't want to
        # generate it anyhow so we need to mask it

        skip = tf.reshape(self.char2int(["[UNK]"]), shape=(1,-1))
        mask = tf.SparseTensor(indices=skip, values=[-float("inf")], dense_shape=[len(char2int.get_vocabulary())])
        self.mask = tf.sparse.to_dense(mask)

    @tf.function
    def one_step_forecasting(self, inputs, memory_states=None, carry_states=None):
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ints = self.char2int(input_chars).to_tensor()
        prediction, memory_states, carry_states = self.model(inputs=input_ints, memory_states=memory_states, carry_states=carry_states, return_state=True)

        prediction = prediction[:, -1, :]
        prediction = prediction/self.temperature
        prediction = prediction + self.mask

        predicted_ints = tf.random.categorical(prediction, num_samples=1)
        predicted_ints = tf.squeeze(predicted_ints, axis=-1)

        predicted_chars = self.int2char(predicted_ints)

        return predicted_chars, memory_states, carry_states


        