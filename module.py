import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


flags = tf.app.flags
args = flags.FLAGS

def discriminator_condition(input_X, input_Z, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)    

    with tf.variable_scope("DISC_CONDITION", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        ## For Image
        netX_in = InputLayer(input_X, name='DX/in')
        netX_h0 = Conv2d(netX_in, args.img_filter, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h0/conv2d')
        netX_h0 = BatchNormLayer(netX_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h0/batch_norm')

        netX_h1 = Conv2d(netX_h0, args.img_filter*2, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h1/conv2d')
        netX_h1 = BatchNormLayer(netX_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h1/batch_norm')

        netX_h2 = Conv2d(netX_h1, args.img_filter*4, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h2/conv2d')
        netX_h2 = BatchNormLayer(netX_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h2/batch_norm')

        netX_h3 = Conv2d(netX_h2, args.img_filter*8, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h3/conv2d')
        netX_h3 = BatchNormLayer(netX_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h3/batch_norm')

        netX_h4 = FlattenLayer(netX_h3, name='DX/h4/flatten')
        netX_h4 = DenseLayer(netX_h4, n_units=args.dX_dim, act=tf.identity,
                             W_init = w_init, name='DX/h4/lin_sigmoid')
        netX_h4 = BatchNormLayer(netX_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h4/batch_norm')

        ## For Code
        netZ_in = InputLayer(input_Z, name='DZ/in')
        netZ_h0 = DenseLayer(netZ_in, n_units=args.dZ_dim, act=tf.identity,
                             W_init = w_init, name='DZ/h0/fcn')
        netZ_h1 = DenseLayer(netZ_h0, n_units=args.dZ_dim, act=tf.identity, 
                             W_init = w_init, name='DZ/h1/fcn')
        
        ## For Joint (Image, Code)
        net_in = ConcatLayer(layer=[netX_h4, netZ_h1], name='DIS/in')
        net_h0 = DenseLayer(net_in, n_units=args.dJ_dim, act=lambda x: tl.act.lrelu(x, 0.2),
                            W_init = w_init, name='DIS/h0/fcn')
        net_h1 = DenseLayer(net_h0, n_units=args.dJ_dim, act=lambda x: tl.act.lrelu(x, 0.2),
                            W_init = w_init, name='DIS/h1/fcn')
        net_h2 = DenseLayer(net_h1, n_units=1,  act=lambda x: tl.act.lrelu(x, 0.2),
                            W_init = w_init, name='DIS/h2/fcn')
        logits = net_h2.outputs

    return net_h2, logits

def decoder_condition(condition, inputs, is_train=True, reuse=False):

    s0, s2, s4, s8, s16 = int(args.output_size), int(args.output_size/2), \
                          int(args.output_size/4), int(args.output_size/8), int(args.output_size/16)
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("DECODER", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # latent coder from conditions
        net_c_in = InputLayer(condition, name='Condition/in')
        net_c_h0 = DenseLayer(net_c_in, n_units=512, act=tf.identity, W_init = w_init, name='Condition/fc1')
        net_c_h1 = DenseLayer(net_c_h0, n_units=args.condition_dim, act=tf.identity, W_init = w_init, \
                              name='Condition/fc2')

        # latent coder from Encoder
        net_e_in = InputLayer(inputs, name='Encoder/in')
        net_e_h0 = DenseLayer(net_e_in, n_units=1024, act=tf.identity, W_init = w_init, name='Encoder/fc1')
        net_e_h1 = DenseLayer(net_e_h0, n_units=args.code_dim, act=tf.identity, W_init = w_init, \
                              name='Encoder/fc2')

        # Joint coder for decoding
        net_h0 = ConcatLayer(layer=[net_c_h1, net_e_h1], name='De/concat')
        net_h0 = DenseLayer(net_h0, n_units=args.img_filter*8*s16*s16, W_init=w_init,
                            act = tf.identity, name='De/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, args.img_filter*8], name='De/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, args.img_filter*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, args.img_filter*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, args.img_filter, (5, 5), out_size=(s2, s2), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, args.img_dim, (5, 5), out_size=(s0, s0), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h4/decon2d')
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
        logits = net_h4.outputs

    return net_h4, logits

def decoder(inputs, is_train=True, reuse=False):

    s0, s2, s4, s8, s16 = int(args.output_size), int(args.output_size/2), \
                          int(args.output_size/4), int(args.output_size/8), int(args.output_size/16)
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("DECODER", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='De/in')
        net_h0 = DenseLayer(net_in, n_units=args.img_filter*8*s16*s16, W_init=w_init,
                            act = tf.identity, name='De/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, args.img_filter*8], name='De/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, args.img_filter*4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, args.img_filter*2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, args.img_filter, (5, 5), out_size=(s2, s2), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=gamma_init, name='De/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, args.img_dim, (5, 5), out_size=(s0, s0), strides=(2, 2),
                          padding='SAME', batch_size=args.batch_size, act=None, W_init=w_init, name='De/h4/decon2d')
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
        logits = net_h4.outputs

    return net_h4, logits


def encoder(inputs, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("ENCODER", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='En/in')
        net_h0 = Conv2d(net_in, args.img_filter, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='En/h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h0/batch_norm')

        net_h1 = Conv2d(net_h0, args.img_filter*2, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='En/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h1/batch_norm')

        net_h2 = Conv2d(net_h1, args.img_filter*4, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='En/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h2/batch_norm')

        net_h3 = Conv2d(net_h2, args.img_filter*8, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='En/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='En/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=args.code_dim, act=tf.identity,
                            W_init = w_init, name='En/h4/lin_sigmoid')
        logits = net_h4.outputs

    return net_h4, logits

def classify(inputs, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)    

    with tf.variable_scope("CLASSIFY", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        net_in = InputLayer(inputs, name='in')
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - (mean/127.5 - 1.0)
        network = Conv2d(net_in, 64, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='h0/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='h0/batch_norm')

        network = Conv2d(network, 128, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='h1/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='h1/batch_norm')

        network = Conv2d(network, 256, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='h2/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='h2/batch_norm')

        network = Conv2d(network, 512, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='h3/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='h3/batch_norm')

        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=3, act=tf.identity, name='fc')
        
        logits = network.outputs

    return network, logits


def discriminator_X(input_X, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)    

    with tf.variable_scope("DISC_X", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        ## For Image
        netX_in = InputLayer(input_X, name='DX/in')
        netX_h0 = Conv2d(netX_in, args.img_filter, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h0/conv2d')
        netX_h0 = BatchNormLayer(netX_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h0/batch_norm')

        netX_h1 = Conv2d(netX_h0, args.img_filter*2, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h1/conv2d')
        netX_h1 = BatchNormLayer(netX_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h1/batch_norm')

        netX_h2 = Conv2d(netX_h1, args.img_filter*4, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h2/conv2d')
        netX_h2 = BatchNormLayer(netX_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h2/batch_norm')

        netX_h3 = Conv2d(netX_h2, args.img_filter*8, (5, 5), (2, 2), act=None,
                         padding='SAME', W_init=w_init, name='DX/h3/conv2d')
        netX_h3 = BatchNormLayer(netX_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h3/batch_norm')

        netX_h4 = FlattenLayer(netX_h3, name='DX/h4/flatten')
        netX_h4 = DenseLayer(netX_h4, n_units=args.dX_dim, act=tf.identity,
                             W_init = w_init, name='DX/h4/lin_sigmoid')
        netX_h4 = BatchNormLayer(netX_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h4/batch_norm')

        logits = netX_h4.outputs

    return netX_h4, logits


def discriminator_Z(input_Z, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)    

    with tf.variable_scope("DISC_Z", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        ## For Code
        netZ_in = InputLayer(input_Z, name='DZ/in')
        netZ_h0 = DropoutLayer(netZ_in, keep=0.8, name='DZ/h0/drop')
        netZ_h0 = DenseLayer(netZ_h0, n_units=args.dZ_dim, act=tf.identity,
                             W_init = w_init, name='DZ/h0/fcn')
        netZ_h1 = DropoutLayer(netZ_h0, keep=0.8, name='DZ/h1/drop')
        netZ_h1 = DenseLayer(netZ_h1, n_units=args.dZ_dim, act=tf.identity, 
                             W_init = w_init, name='DZ/h1/fcn')
        
        logits = netZ_h1.outputs

    return netZ_h1, logits

def discriminator_J(input_X, input_Z, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)    

    with tf.variable_scope("DISC_J", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        ## For Image
        netX_in = InputLayer(input_X, name='DX/in')
        netX_h0 = Conv2d(netX_in, args.img_filter, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='DX/h0/conv2d')
        netX_h0 = BatchNormLayer(netX_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='DX/h0/batch_norm')

        netX_h1 = Conv2d(netX_h0, args.img_filter*2, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='DX/h1/conv2d')
        netX_h1 = BatchNormLayer(netX_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='DX/h1/batch_norm')

        netX_h2 = Conv2d(netX_h1, args.img_filter*4, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='DX/h2/conv2d')
        netX_h2 = BatchNormLayer(netX_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='DX/h2/batch_norm')

        netX_h3 = Conv2d(netX_h2, args.img_filter*8, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='DX/h3/conv2d')
        netX_h3 = BatchNormLayer(netX_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='DX/h3/batch_norm')

        netX_h4 = FlattenLayer(netX_h3, name='DX/h4/flatten')
        netX_h4 = DenseLayer(netX_h4, n_units=args.dX_dim, act=tf.identity,
                            W_init = w_init, name='DX/h4/lin_sigmoid')
        netX_h4 = BatchNormLayer(netX_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                                 is_train=is_train, gamma_init=gamma_init, name='DX/h4/batch_norm')

        ## For Code
        netZ_in = InputLayer(input_Z, name='DZ/in')
        netZ_h0 = DropoutLayer(netZ_in, keep=0.8, name='DZ/h0/drop')
        netZ_h0 = DenseLayer(netZ_h0, n_units=args.dZ_dim, act=tf.identity,
                             W_init = w_init, name='DZ/h0/fcn')
        netZ_h1 = DropoutLayer(netZ_h0, keep=0.8, name='DZ/h1/drop')
        netZ_h1 = DenseLayer(netZ_h1, n_units=args.dZ_dim, act=tf.identity, 
                             W_init = w_init, name='DZ/h1/fcn')
        
        ## For Joint (Image, Code)
        net_in = ConcatLayer(layer=[netX_h4, netZ_h1], name='DIS/in')
        net_h0 = DropoutLayer(net_in, keep=0.8, name='DIS/h0/drop')
        net_h0 = DenseLayer(net_h0, n_units=args.dJ_dim, act=lambda x: tl.act.lrelu(x, 0.2),
                            W_init = w_init, name='DIS/h0/fcn')
        net_h1 = DropoutLayer(net_h0, keep=0.8, name='DIS/h1/drop')
        net_h1 = DenseLayer(net_h1, n_units=args.dJ_dim, act=lambda x: tl.act.lrelu(x, 0.2),
                            W_init = w_init, name='DIS/h1/fcn')
        net_h2 = DropoutLayer(net_h1, keep=0.8, name='DIS/h2/drop')
        net_h2 = DenseLayer(net_h2, n_units=1,  act=lambda x: tl.act.lrelu(x, 0.2),
                            W_init = w_init, name='DIS/h2/fcn')
        logits = net_h2.outputs

    return net_h2, logits

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def wgan_criterion(in_, target):
    return tf.reduce_mean(in_ - target)

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def cross_loss(logits, labels):
    return tl.cost.cross_entropy(logits, labels, name='cross_entropy')
