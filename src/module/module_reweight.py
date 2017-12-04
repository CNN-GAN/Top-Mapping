import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


flags = tf.app.flags
args = flags.FLAGS

def RNN_encoder(captions, is_train=True, reuse=False, name="RNN_encoder", return_last=False):
    ''' Input captions -> embedding -> text_dim (encoder dim) '''
    weight = tf.random_normal_initializer(stddev=0.02)
    LSTMCells = tf.contrib.rnn.LSTMCell
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.DynamicRNNLayer(
            rnn_encoder, cell_fn=LSTMCells,
            cell_init_args={"state_is_tuple": True},
            n_hidden=text_dim, 
            dropout=(0.95 if is_train else None),
            initializer=weight,
            return_last=return_last,
            name="codes")
        net = tl.layers.DenseLayer(net, n_units=1, act=tf.identity, name="output")
        return net
