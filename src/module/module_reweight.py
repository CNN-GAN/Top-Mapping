import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


flags = tf.app.flags
args = flags.FLAGS

def LSTM_encoder(net, is_train=True, reuse=False, name="LSTM_encoder", return_last=False):
    ''' Input captions -> embedding -> text_dim (encoder dim) '''
    weight = tf.random_normal_initializer(stddev=0.02)
    LSTMCells = tf.contrib.rnn.LSTMCell
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net = tl.layers.DynamicRNNLayer(
            net, cell_fn=LSTMCells,
            cell_init_args={"state_is_tuple": True},
            dropout=(0.95 if is_train else None),
            initializer=weight,
            return_last=return_last,
            name="codes")

        net = tl.layers.DenseLayer(net, n_units=1, act=tf.identity, name="output")
        return net

def Mahalanobis_Loss(cc, cn, cf, is_train=True, reuse=False, name="Mahalanobis"):

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_cc = InputLayer(cc, name='cc')
        net_cn = InputLayer(cn, name='cn')
        net_cf = InputLayer(cf, name='cf')

        
        return net
