import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

flags = tf.app.flags
args = flags.FLAGS

def LSTM_Encoder(inputs, is_train=True, reuse=False, name="LSTM_Encoder", return_last=False):

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net = InputLayer(input_X, name='In')
        net = ReshapeLayer(net, shape=[1, -1, args.code_dim])
        net = DynamicRNNLayer(net, 
                              cell_fn = tf.contrib.rnn.BasicLSTMCell,
                              n_hidden = code_dim,
                              dropout = 0.7,
                              sequence_length = 1,
                              return_seq_2d = true,
                              name = 'LSTM')
        
        net = DenseLayer(net, n_units=1, act=tf.identity, name="Weighting")
        return net
