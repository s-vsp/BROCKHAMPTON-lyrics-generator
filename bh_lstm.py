import numpy as np
import pandas as pd
import os
import h5py
import sys
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD



def create_samples(text_sequence):
    # Creating data samples with corresponding targets

    X = text_sequence[:-1]
    y = text_sequence[1:]
    # X, y should have the same shape

    return X, y



def get_title():

    lyrics_text = open("BROCKHAMPTON.txt", "r", encoding="utf-8").read()

    t = np.random.randint(1,len(lyrics_text),size=1)

    ignorable = [" ", "\n", ".", ",", "[", "]", "(", ")"]

    chars = []

    for i, c in enumerate(lyrics_text[t.item():t.item()+20]):
        chars.append(c)
        if c in ignorable and i == 0:
            continue
        elif c in ignorable and i != 0:
            chars.remove(c)
            break
    
    title = ""

    return print(title.join((chars)).upper())



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
    SEQUENCE_LENGTH = 100
    text_sequences = dataset.batch(batch_size=SEQUENCE_LENGTH+1, drop_remainder=True)

    # Creating data sort of dataframe (X,y) - paired
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000

    lyrics_data = text_sequences.map(create_samples)
    lyrics_data = (lyrics_data.shuffle(buffer_size=BUFFER_SIZE).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    #print(f"Lyrics data defined as: \n{lyrics_data}")



    ###--- MODELING ---###

    EMBEDDING_DIMENSION = 64
    RNN_UNITS = 256

    model = LSTM_RNN(vocabulary_size=len(char2int.get_vocabulary()), embedding_dimension=EMBEDDING_DIMENSION, rnn_units=RNN_UNITS)
    model.build(input_shape=(BATCH_SIZE, SEQUENCE_LENGTH))

    print(model.summary())

    loss_fc = SparseCategoricalCrossentropy(from_logits=True, name="sparse_categorical_crossentropy")
    #sgd_opt = SGD(momentum=0.9, learning_rate=0.005, nesterov=True)

    model.compile(optimizer="adam", loss=loss_fc)

    # TensorBoard callback configurations
    log_dir = r"c:\Users\Kamil\My_repo\BROCKHAMPTON-lyrics-generator\BROCKHAMPTON-lyrics-generator" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callbacks = TensorBoard(log_dir=log_dir, histogram_freq=1)

    EPOCHS = 120

    model.fit(lyrics_data, epochs=EPOCHS, callbacks=[tensorboard_callbacks])




if __name__ == "__main__":
    run()