from __future__ import division

import os
import sys
import scipy.misc
import pprint
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from random import shuffle
from six.moves import xrange
from collections import namedtuple
from glob import glob
from tensorlayer.layers import *
from module import *
from utils import *

class Net(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        self.method = args.method


        self._build_model()

    def _build_model(self):
        self.d.real.x = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, \
                                                      self.img_dim], name='real_image')
        self.d.real.z  = tf.placeholder(tf.float32, [self.batch_size, self.code_dim], name="real_code")

        self.n.fake.x, self.d.fake.x = decoder(self.d.real.z, is_train=True, reuse=False)
        self.n.cycl.z, self.d.cycl.z = encoder(self.d.fake.x, is_train=True, reuse=True )
        self.n.fake.z, self.d.fake.z = encoder(self.d.real.x, is_train=True, reuse=False)
        self.n.cycl.x, self.d.cycl.x = decoder(self.d.fake.z, is_train=True, reuse=True )

        with tf.name_scope('real'):
            true_image = tf.reshape(real_images, [-1, 64, 64, 3])
            tf.summary.image('real', true_image[0:4], 4)

        with tf.name_scope('fake'):
            fake_image = tf.reshape(cyc_X, [-1, 64, 64, 3])
            tf.summary.image('fake', fake_image[0:4], 4)

        self.n.dic.x,  self.d.dic.x  = discriminator_X(self.d.real.x, is_train=True, reuse=False)
        self.n.dic.z,  self.d.dic.z  = discriminator_Z(self.d.real.z, is_train=True, reuse=False)
        self.n.dic.J,  self.d.dic.J  = discriminator_J(self.d.real.J, is_train=True, reuse=False)

        self.n.dic.fx, self.d.dic.fx = discriminator_X(self.d.fake.x, is_train=True, reuse=True)
        self.n.dic.fz, self.d.dic.fz = discriminator_Y(self.d.fake.y, is_train=True, reuse=True)
        self.n.dic.fJ, self.d.dic.fJ = discriminator_J(self.d.fake.J, is_train=True, reuse=True)

        # Apply Loss
        with tf.name_scope('generator'):
            self.loss.encoder = self.param.side_D * tf.reduce_mean((dic_fZ - 1)**2)
            tf.summary.scalar('encoder_loss', self.loss.encoder)

            self.loss.decoder = self.param.side_D * tf.reduce_mean((dic_fX - 1)**2)
            tf.summary.scalar('decoder_loss', self.loss.decoder)

            self.loss.cycle   = self.param.cycle  * tf.reduce_mean(tf.abs(real_images - cyc_X)) \
                   + FLAGS.lamda * tf.reduce_mean(tf.abs(z - cyc_Z))
            tf.summary.scalar('clc_loss', clc_loss)

        """ Least Square Loss """
        # discriminator for Joint
        dic_J_loss = 0.5 * (tf.reduce_mean((dic_J - 1)**2) + tf.reduce_mean((dic_fJ)**2))
        tf.summary.scalar('d_J_loss', dic_J_loss)

        # discriminator for Joint Fake
        dic_fJ_loss = 0.5 * (tf.reduce_mean((dic_J)**2) + tf.reduce_mean((dic_fJ - 1)**2))
        tf.summary.scalar('d_fJ_loss', dic_fJ_loss)

        # discriminator for X
        dic_X_loss = FLAGS.side_dic * 0.5 * (tf.reduce_mean((dic_X - 1)**2) + tf.reduce_mean((dic_fX)**2))
        tf.summary.scalar('d_X_loss', dic_X_loss)

        # discriminator for Z
        dic_Z_loss = FLAGS.side_dic * 0.5 * (tf.reduce_mean((dic_Z - 1)**2) + tf.reduce_mean((dic_fZ)**2))
        tf.summary.scalar('d_Z_loss', dic_Z_loss)
        
    with tf.name_scope('generator'):
        en_vars = tl.layers.get_variables_with_name('ENCODER', True, True)
        de_vars = tl.layers.get_variables_with_name('DECODER', True, True)
        gen_vars = en_vars
        gen_vars.extend(de_vars)

    with tf.name_scope('discriminator'):
        Z_vars = tl.layers.get_variables_with_name('DISC_Z', True, True)
        X_vars = tl.layers.get_variables_with_name('DISC_X', True, True)
        J_vars = tl.layers.get_variables_with_name('DISC_J', True, True)
        #variable_summaries(d_vars)

    n_fake_X.print_params(False)
    print("---------------")
    n_fake_Z.print_params(False)
    
