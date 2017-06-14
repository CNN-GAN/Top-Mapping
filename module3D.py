import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


flags = tf.app.flags
args = flags.FLAGS

## In this module, the inputs is the 3D point_cloud generated from the dynamic mapping module.
## The dynamic map is cylinder with radius at 30m, and height at 5m, the resolution level is set to 
## 0.8m, then the map is extracted as (40/0.8, 40/0.8, 10/0.8)->(50, 50, 12)

def encoder3D(inputs, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("ENCODER", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='En/in')
        net_h0 = Conv3dLayer(net_in, act=None, shape=[3,3,3,1,args.voxel_filter], stride=[1,2,2,2,1],\
                             padding='SAME', W_init=w_init, b_init=b_init, name='En/h0/conv2d')
        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h0/batch_norm')

        net_h1 = Conv3dLayer(net_h0, act=None, shape=[3,3,3,1,args.voxel_filter*2], stride=[1,2,2,1,1],\
                             padding='SAME', W_init=w_init, b_init=b_init, name='En/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h1/batch_norm')

        net_h2 = Conv3dLayer(net_h1, act=None, shape=[3,3,3,1,args.voxel_filter*4], stride=[1,2,2,1,1],\
                             padding='SAME', W_init=w_init, b_init=b_init, name='En/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h2/batch_norm')

        net_h3 = Conv3dLayer(net_h2, act=None, shape=[3,3,3,1,args.voxel_filter*8], stride=[1,2,2,2,1],\
                             padding='SAME', W_init=w_init, b_init=b_init, name='En/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='En/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='En/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=args.code_dim, act=tf.identity,
                            W_init = w_init, name='En/h4/lin_sigmoid')
        logits = net_h4.outputs

    return net_h4, logits


def decoder3D(inputs, is_train=True, reuse=False):
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
