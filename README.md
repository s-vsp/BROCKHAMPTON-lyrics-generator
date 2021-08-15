# Brockhampton lyrics generator
LSTM RNN approach to generate new Brockhampton songs.

*Keywords: Deep Learning, RNN, LSTM, One-Step Forecasting, TensorFlow, Keras, Text Generation, Music Generation*

# Table of contents:
- [Idea](#Idea)
- [Project](#Project)
  - [Preprocessing](#Preprocessing)
- [References](#References)


## Idea <a name="Idea"></a>

The main idea of the project was to generate lyrics using Long short-term memory recurrent neural network. Followed the TensorFlow tutorial [[1]](https://www.tensorflow.org/text/tutorials/text_generation) where Shakespeare's poems where generated based on GRU architecture I decided to perform a similar approach, but based on LSTM. LSTMs are genuinely described in Colah's blog [[2]](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
The dataset includes lyrics from all the Brockhampton songs gathered from [genius.com](https://genius.com/) and having 207 651 chars.

## Project <a name="Project"></a>

The whole project can be divided into 3 parts: preprocessing, training and evaluating.

### Preprocessing <a name="Preprocessing"></a>

During preprocessing the vocabulary from the loaded Brockhampton lyrics was created and included exactly 100 unique chars. Following that, 2 StringLookup mappings were generated - first one mapping chars into integers (char2int) and the second one mapping integers into chars (int2char). Lastly the data was transformed into TensorFlow dataset (from tensor slices) and proceeded as sequences made of 100 chars. Finally the data used for training the model was inputed in form of pairs -> (input [X], target [y]), both having the same shape, but formed the way that the input is the current char and target is the next char, e.g. X = [Hell], y = [ello]. Data was shuffled, controlled by the BUFFER_SIZE parameter and inputed to the net as batches of size 64.

### Training

Before training the model, the obvious thing to do was to build the whole network. It contains 4 layers: embedding layer, lstm layer, dense layer and dropout layer. During the first phases of this project, the dropout layer wasn't used and it yield to overfitting the net, that's why it is important to add it with a dropout value equal to 0.1 (or any other value lesser than 0.3 -> larger ones yield to poor loss/accuracy performance and weak predictions). Here we can see it's summary:
![Model's summary](https://imgur.com/a/3UhiNJJ)


## References <a name="References"></a>
- [1]. [Generate Text with RNNs](https://www.tensorflow.org/text/tutorials/text_generation)
- [2]. [LSTMs by Colah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
