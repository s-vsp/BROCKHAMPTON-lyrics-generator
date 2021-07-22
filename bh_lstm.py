import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras




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
    lyrics_data = tf.data.Dataset.from_tensor_slices(lyrics_as_ints)
    
    # Creating bathces of data
    seq_len = 100
    text_sequence = lyrics_data.batch(batch_size=seq_len+1, drop_remainder=True)

    for sequence in text_sequence.take(1):
        # taking one batch, should return a tensor of shape (seq_len+1,)
        print(tf.strings.reduce_join(int2char(sequence)).numpy())




if __name__ == "__main__":
    run()