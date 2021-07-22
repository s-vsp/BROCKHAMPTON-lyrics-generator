import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras



def create_samples(text_sequence):
    # Creating data samples with corresponding targets

    X = text_sequence[:-1]
    y = text_sequence[1:]
    # X, y should have the same shape

    return X, y


def run():

    ###--- PREPROCESSING ---###

    lyrics_text = open("BROCKHAMPTON.txt", "r", encoding="utf-8").read()

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
    lyrics_data = text_sequences.map(create_samples)
    lyrics_data = (lyrics_data.shuffle(buffer_size=1000).batch(batch_size=64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    print(f"Lyrics data defined as: \n{lyrics_data}")
    print(len(vocabulary))





if __name__ == "__main__":
    run()