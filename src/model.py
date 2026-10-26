"""RNN classifier: Embedding -> recurrent layer -> Dense -> sigmoid."""
import tensorflow as tf
from tensorflow.keras import layers

from config import MAX_LEN, VOCAB_SIZE

RNN_LAYERS = {"simplernn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}


def build_model(rnn_type="simplernn", embed_dim=64, rnn_units=64, dense_units=32,
                dropout=0.0, recurrent_dropout=0.0, learning_rate=1e-3, clipnorm=None):
    rnn = RNN_LAYERS[rnn_type]
    model = tf.keras.Sequential([
        layers.Input(shape=(MAX_LEN,), dtype="int32"),
        layers.Embedding(VOCAB_SIZE, embed_dim, mask_zero=True, name="embedding"),
        rnn(rnn_units, dropout=dropout, recurrent_dropout=recurrent_dropout, name=rnn_type),
        layers.Dense(dense_units, activation="relu", name="dense"),
        layers.Dropout(dropout, name="dropout"),
        layers.Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=clipnorm),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model
