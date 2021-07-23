import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
from tensorflow.compat.v1 import enable_eager_execution



def create_samples(text_sequence):
    # Creating data samples with corresponding targets

    X = text_sequence[:-1]
    y = text_sequence[1:]
    # X, y should have the same shape

    return X, y




class LSTM_RNN(keras.Model):

    """
    Parameters:
    -----------
    vocabulary_size: int
        length of the chars' vocabulary
    embedding_dimension: int
        embedding dimension
    rnn_units: int
        number of RNN units
    """

    def __init__(self, vocabulary_size, embedding_dimension, rnn_units):
        super().__init__(self)
        self.embedding = Embedding(vocabulary_size, embedding_dimension)
        self.lstm = LSTM(rnn_units, activation="tanh", return_sequences=True, return_state=True, unit_forget_bias=True)
        self.dense = Dense(vocabulary_size)

    def call(self, inputs, memory_states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
    
        if memory_states is None:
            memory_states = self.lstm.get_initial_state(x)
    
        x, memory_states, carry_states = self.lstm(x, initial_state=memory_states, training=training)
        x = self.dense(x, training=training)

        if return_state==True:
            return x, memory_states
        else:
            return x



def run():

    ###--- PREPROCESSING ---###

    lyrics_text = open("BROCKHAMPTON.txt", "r", encoding="utf-8").read()
    #print(len(lyrics_text))

    # Unique chars in lyrics file
    vocabulary = list(sorted(set(lyrics_text))) 

    # Mappings
    char2int = keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocabulary, mask_token=None)
    int2char = keras.layers.experimental.preprocessing.StringLookup(vocabulary=char2int.get_vocabulary(), invert=True, mask_token=None)

    lyrics_as_ints = char2int(tf.strings.unicode_split(input=lyrics_text, input_encoding="UTF-8"))
    
    # Creating tf Dataset
    dataset = tf.data.Dataset.from_tensor_slices(lyrics_as_ints)
    
    # Creating bathces of data
    seq_len = 100
    text_sequences = dataset.batch(batch_size=seq_len+1, drop_remainder=True)

    # Creating data sort of dataframe (X,y) - paired
    batch_size=64
    buffer_size=1000

    lyrics_data = text_sequences.map(create_samples)
    lyrics_data = (lyrics_data.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    #print(f"Lyrics data defined as: \n{lyrics_data}")

    model = LSTM_RNN(vocabulary_size=len(char2int.get_vocabulary()), embedding_dimension=256, rnn_units=1024)
    model.build(input_shape=(batch_size, seq_len))

    print(model.summary())


if __name__ == "__main__":
    run()