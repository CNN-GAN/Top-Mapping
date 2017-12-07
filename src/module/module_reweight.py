import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


flags = tf.app.flags
args = flags.FLAGS

def LSTM_Encoder(inputs, embedding_dim=2, text_dim=2, vocab_size=1000, \
                 is_train=True, reuse=False, name="LSTM_Encoder", return_last=False):

    weight = tf.random_normal_initializer(stddev=0.02)
    LSTMCells = tf.contrib.rnn.LSTMCell
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net = tl.layers.EmbeddingInputlayer(
            inputs=inputs, vocabulary_size=vocab_size,
            embedding_size=embedding_dim, E_init=weight,
            name="EMB")

        net = tl.layers.DynamicRNNLayer(
            net, cell_fn=LSTMCells,
            cell_init_args={"state_is_tuple": True},
            n_hidden=text_dim, dropout=(0.95 if is_train else None),
            initializer=weight,
            sequence_length=tl.layers.retrieve_seq_length_op2(inputs),
            return_last=True,
            name="cells")

        net = tl.layers.DenseLayer(net, n_units=1, act=tf.identity, name="output")
        return net
