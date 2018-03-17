from __future__ import division

import os
import sys
import scipy.misc
import pprint
import time
import json

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from random import shuffle
from six.moves import xrange
from collections import namedtuple
from glob import glob
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from tensorlayer.layers import *
from src.module.module_simpleCYC import *
from src.util.utils import *
from src.data import *

class Net_HEADINGINV(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        # ALI approach
        self.model    = args.method
        self.is_train = args.is_train 
        
        # Network module
        self.encoder  = encoder
        self.decoder  = decoder_condition
        self.discriminator = discriminator
        
        # Loss function
        self.lossGAN = abs_criterion
        self.lossCYC = abs_criterion
        self.lossCross = cross_loss
        self.lossCode  = abs_criterion

        # SeqSLAM
        self.vec_D    = Euclidean

        self.data_iter = HeadINVdataSampler

        # Test
        if args.is_train == False:
            self.test_epoch = 0

        self._build_model(args)

    def _build_model(self, args):

        # placeholder
        self.d_R0  = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_r0')
        self.d_RP1 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_rp1')
        self.d_RP2 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_rp2')
        self.d_RN1 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_rn1')
        self.d_RN2 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_rn2')

        self.d_OR0  = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                 args.img_dim], name='real_or0')
        self.d_ORP1 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                 args.img_dim], name='real_orp1')
        self.d_ORP2 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                 args.img_dim], name='real_orp2')
        self.d_ORN1 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                 args.img_dim], name='real_orn1')
        self.d_ORN2 = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                 args.img_dim], name='real_orn2')

        self.d_fake_c = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='rand_code')
        
        ## Encoder
        self.n_real_c, self.d_real_c = self.encoder(self.d_R0,  is_train=True, reuse=False)
        self.n_rp1_c, self.d_rp1_c   = self.encoder(self.d_RP1, is_train=True, reuse=True)
        self.n_rp2_c, self.d_rp2_c   = self.encoder(self.d_RP2, is_train=True, reuse=True)
        self.n_rn1_c, self.d_rn1_c   = self.encoder(self.d_RN1, is_train=True, reuse=True)
        self.n_rn2_c, self.d_rn2_c   = self.encoder(self.d_RN2, is_train=True, reuse=True)

        self.n_other_c, self.d_other_c = self.encoder(self.d_OR0,  is_train=True, reuse=True)
        self.n_orp1_c, self.d_orp1_c   = self.encoder(self.d_ORP1, is_train=True, reuse=True)
        self.n_orp2_c, self.d_orp2_c   = self.encoder(self.d_ORP2, is_train=True, reuse=True)
        self.n_orn1_c, self.d_orn1_c   = self.encoder(self.d_ORN1, is_train=True, reuse=True)
        self.n_orn2_c, self.d_orn2_c   = self.encoder(self.d_ORN2, is_train=True, reuse=True)

        ## Decoder
        self.n_reco_r0,  self.d_reco_r0, self.n_code_r0,  self.d_code_r0 = self.decoder(self.d_real_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=False)
        self.n_reco_rn2, self.d_reco_rn2, self.n_code_rn2, self.d_code_rn2 = self.decoder(self.d_real_c, tf.tile(tf.reshape(tf.one_hot(0, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_reco_rn1, self.d_reco_rn1, self.n_code_rn1, self.d_code_rn1 = self.decoder(self.d_real_c, tf.tile(tf.reshape(tf.one_hot(1, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_reco_rd,  self.d_reco_rd, self.n_code_rd, self.d_code_rd  = self.decoder(self.d_fake_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_reco_rp1, self.d_reco_rp1, self.n_code_rp1, self.d_code_rp1 = self.decoder(self.d_real_c, tf.tile(tf.reshape(tf.one_hot(3, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_reco_rp2, self.d_reco_rp2, self.n_code_rp2, self.d_code_rp2 = self.decoder(self.d_real_c, tf.tile(tf.reshape(tf.one_hot(4, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )

        self.n_cyc_rn2, self.d_cyc_rn2, self.n_code_crn2, self.d_code_crn2 = self.decoder(self.d_rn2_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_cyc_rn1, self.d_cyc_rn1, self.n_code_crn1, self.d_code_crn1 = self.decoder(self.d_rn1_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_cyc_rp1, self.d_cyc_rp1, self.n_code_crp1, self.d_code_crp1 = self.decoder(self.d_rp1_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )
        self.n_cyc_rp2, self.d_cyc_rp2, self.n_code_crp2, self.d_code_crp2 = self.decoder(self.d_rp2_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=True, reuse=True )

        self.n_reco_or0,  self.d_reco_or0, self.n_code_or0,  self.d_code_or0 = self.decoder(self.d_other_c, tf.tile(tf.reshape(tf.one_hot(2, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True)
        self.n_reco_orn2, self.d_reco_orn2, self.n_code_orn2, self.d_code_orn2 = self.decoder(self.d_orn2_c, tf.tile(tf.reshape(tf.one_hot(0, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )
        self.n_reco_orn1, self.d_reco_orn1, self.n_code_orn1, self.d_code_orn1 = self.decoder(self.d_orn1_c, tf.tile(tf.reshape(tf.one_hot(1, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )
        self.n_reco_orp1, self.d_reco_orp1, self.n_code_orp1, self.d_code_orp1 = self.decoder(self.d_orp1_c, tf.tile(tf.reshape(tf.one_hot(3, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )
        self.n_reco_orp2, self.d_reco_orp2, self.n_code_orp2, self.d_code_orp2 = self.decoder(self.d_orp2_c, tf.tile(tf.reshape(tf.one_hot(4, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )

        #self.n_back_rn2, self.d_back_rn2 = self.decoder(self.d_rn2_c, tf.tile(tf.reshape(tf.one_hot(4, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )
        #self.n_back_rn1, self.d_back_rn1 = self.decoder(self.d_rn1_c, tf.tile(tf.reshape(tf.one_hot(3, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )
        #self.n_back_rp1, self.d_back_rp1 = self.decoder(self.d_rp1_c, tf.tile(tf.reshape(tf.one_hot(1, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )
        #self.n_back_rp2, self.d_back_rp2 = self.decoder(self.d_rp2_c, tf.tile(tf.reshape(tf.one_hot(0, 5), [-1, 5]), [args.batch_size, 5]), is_train=False, reuse=True )

        ## Discriminator
        self.n_dis_orig, self.d_dis_orig = self.discriminator(self.d_R0,      is_train=True, reuse=False)
        self.n_dis_reco, self.d_dis_reco = self.discriminator(self.d_reco_r0, is_train=True, reuse=True )
        self.n_dis_rand, self.d_dis_rand = self.discriminator(self.d_reco_rd, is_train=True, reuse=True )
        
        ## loss for cycle updating
        self.loss_cyc_r0   = self.lossCYC(self.d_R0,  self.d_reco_r0)
        self.loss_cyc_rp1  = self.lossCYC(self.d_RP1, self.d_cyc_rp1)
        self.loss_cyc_rp2  = self.lossCYC(self.d_RP2, self.d_cyc_rp2)
        self.loss_cyc_rn1  = self.lossCYC(self.d_RN1, self.d_cyc_rn1)
        self.loss_cyc_rn2  = self.lossCYC(self.d_RN2, self.d_cyc_rn2)

        self.loss_rp1  = self.lossCYC(self.d_RP1, self.d_reco_rp1)
        self.loss_rp2  = self.lossCYC(self.d_RP2, self.d_reco_rp2)
        self.loss_rn1  = self.lossCYC(self.d_RN1, self.d_reco_rn1)
        self.loss_rn2  = self.lossCYC(self.d_RN2, self.d_reco_rn2)

        self.loss_cycle = self.loss_cyc_r0 + self.loss_cyc_rp1 + self.loss_cyc_rp2 + self.loss_cyc_rn1 + self.loss_cyc_rn2
        self.loss_head_diff = self.loss_rp1 + self.loss_rp2 + self.loss_rn1 + self.loss_rn2

        ## loss for transformation
        self.loss_trans_rp1  = self.lossCYC(self.d_code_r0, self.d_code_rp1)
        self.loss_trans_rp2  = self.lossCYC(self.d_code_r0, self.d_code_rp2)
        self.loss_trans_rn1  = self.lossCYC(self.d_code_r0, self.d_code_rn1)
        self.loss_trans_rn2  = self.lossCYC(self.d_code_r0, self.d_code_rn2)
        
        self.loss_trans = (self.loss_trans_rn2 + self.loss_trans_rn1 + self.loss_trans_rp1 + self.loss_trans_rp2)/4.0

        ## loss for encoder
        '''
        self.loss_crp1  = self.lossCYC(self.d_rp1_c, self.d_real_c)
        self.loss_crp2  = self.lossCYC(self.d_rp2_c, self.d_real_c)
        self.loss_crn1  = self.lossCYC(self.d_rn1_c, self.d_real_c)
        self.loss_crn2  = self.lossCYC(self.d_rn2_c, self.d_real_c)
        

        self.loss_same_similar = self.lossCYC(self.d_rp1_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rp2_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rn1_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rn2_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rp2_c, self.d_rp1_c)  + \
                                 self.lossCYC(self.d_rn1_c, self.d_rp1_c)  + \
                                 self.lossCYC(self.d_rn2_c, self.d_rp1_c)  + \
                                 self.lossCYC(self.d_rn1_c, self.d_rp2_c)  + \
                                 self.lossCYC(self.d_rn2_c, self.d_rp2_c)  + \
                                 self.lossCYC(self.d_rn2_c, self.d_rn1_c)


        self.loss_other_similar = self.lossCYC(self.d_orp1_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orp2_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orn1_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orn2_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orp2_c, self.d_orp1_c) + \
                                  self.lossCYC(self.d_orn1_c, self.d_orp1_c) + \
                                  self.lossCYC(self.d_orn2_c, self.d_orp1_c) + \
                                  self.lossCYC(self.d_orn1_c, self.d_orp2_c) + \
                                  self.lossCYC(self.d_orn2_c, self.d_orp2_c) + \
                                  self.lossCYC(self.d_orn2_c, self.d_orn1_c)
        '''
        '''
        self.loss_same_center  = self.lossCYC(self.d_real_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rp1_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rp2_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rn1_c, self.d_real_c) + \
                                 self.lossCYC(self.d_rn2_c, self.d_real_c)

        self.loss_other_center  = self.lossCYC(self.d_other_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orp1_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orp2_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orn1_c, self.d_real_c) + \
                                  self.lossCYC(self.d_orn2_c, self.d_real_c)
        '''

        self.loss_same_center  = (self.lossCYC(self.d_code_r0, self.d_code_r0) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rp1) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rp2) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rn1) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rn2))/5.0

        self.loss_other_center = (self.lossCYC(self.d_code_r0, self.d_code_or0) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rn2) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rn1) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rp1) + \
                                  self.lossCYC(self.d_code_r0, self.d_code_rp2)) / 5.0

        # max(0, 1-||ht-ht_n||/(||ht-ht_1||+beta))
        self.loss_maha = tf.nn.relu(tf.subtract(1.0, tf.div(self.loss_other_center, tf.add(self.loss_same_center, self.loss_same_center * args.distance_weighting))))

        ## loss for generator          
        self.loss_gen = self.lossGAN(self.d_dis_reco, 1) + self.lossGAN(self.d_dis_rand, 1)

        ## loss for discriminator
        self.loss_dis = (self.lossGAN(self.d_dis_orig, 1) + \
                         self.lossGAN(self.d_dis_reco, 0) + \
                         self.lossGAN(self.d_dis_rand, 0)) / 3.0

        # Make summary
        with tf.name_scope('cycle'):        
            self.summ_cyc = tf.summary.scalar('cyc_loss', self.loss_cycle)
            self.summ_head_diff = tf.summary.scalar('head_loss', self.loss_head_diff)
            #self.summ_same_similar = tf.summary.scalar('similar_same_loss', self.loss_same_similar)
            #self.summ_other_similar = tf.summary.scalar('similar_other_loss', self.loss_other_similar)
            self.summ_same_center  = tf.summary.scalar('similar_same',  self.loss_same_center)
            self.summ_other_center = tf.summary.scalar('similar_other', self.loss_other_center)
            self.summ_maha         = tf.summary.scalar('maha_distance', self.loss_maha)
            self.summ_trans        = tf.summary.scalar('trans_distance', self.loss_trans)

        with tf.name_scope('gen-dis-loss'):
            self.summ_gen = tf.summary.scalar('gen_loss', self.loss_gen)
            self.summ_dis = tf.summary.scalar('dis_loss', self.loss_dis)

        with tf.name_scope('R0'):
            true_image = tf.reshape(self.d_R0,      [-1, args.output_size, args.output_size, 3])
            self.summ_r0_real = tf.summary.image('r0_orig', true_image[0:4], 4)
            true_image = tf.reshape(self.d_reco_r0, [-1, args.output_size, args.output_size, 3])
            self.summ_r0_reco = tf.summary.image('r0_reco', true_image[0:4], 4)
            true_image = tf.reshape(self.d_OR0, [-1, args.output_size, args.output_size, 3])
            self.summ_or0_real = tf.summary.image('or0_real', true_image[0:4], 4)

        with tf.name_scope('RP1'):
            true_image = tf.reshape(self.d_RP1,      [-1, args.output_size, args.output_size, 3])
            self.summ_rp1_real = tf.summary.image('rp1_orig', true_image[0:4], 4)
            true_image = tf.reshape(self.d_reco_rp1, [-1, args.output_size, args.output_size, 3])
            self.summ_rp1_reco = tf.summary.image('rp1_reco', true_image[0:4], 4)
            true_image = tf.reshape(self.d_cyc_rp1, [-1, args.output_size, args.output_size, 3])
            self.summ_rp1_cyc = tf.summary.image('rp1_cyc', true_image[0:4], 4)
            #true_image = tf.reshape(self.d_ORP1, [-1, args.output_size, args.output_size, 3])
            #self.summ_orp1_real = tf.summary.image('orp1_real', true_image[0:4], 4)

        with tf.name_scope('RP2'):
            true_image = tf.reshape(self.d_RP2,      [-1, args.output_size, args.output_size, 3])
            self.summ_rp2_real = tf.summary.image('rp2_orig', true_image[0:4], 4)
            true_image = tf.reshape(self.d_reco_rp2, [-1, args.output_size, args.output_size, 3])
            self.summ_rp2_reco = tf.summary.image('rp2_reco', true_image[0:4], 4)
            true_image = tf.reshape(self.d_cyc_rp2, [-1, args.output_size, args.output_size, 3])
            self.summ_rp2_cyc = tf.summary.image('rp2_cyc', true_image[0:4], 4)
            #true_image = tf.reshape(self.d_ORP2, [-1, args.output_size, args.output_size, 3])
            #self.summ_orp2_real = tf.summary.image('orp2_real', true_image[0:4], 4)

        with tf.name_scope('RN1'):
            true_image = tf.reshape(self.d_RN1,      [-1, args.output_size, args.output_size, 3])
            self.summ_rn1_real = tf.summary.image('rn1_orig', true_image[0:4], 4)
            true_image = tf.reshape(self.d_reco_rn1, [-1, args.output_size, args.output_size, 3])
            self.summ_rn1_reco = tf.summary.image('rn1_reco', true_image[0:4], 4)
            true_image = tf.reshape(self.d_cyc_rn1, [-1, args.output_size, args.output_size, 3])
            self.summ_rn1_cyc = tf.summary.image('rn1_cyc', true_image[0:4], 4)
            #true_image = tf.reshape(self.d_ORN1, [-1, args.output_size, args.output_size, 3])
            #self.summ_orn1_real = tf.summary.image('orn1_real', true_image[0:4], 4)

        with tf.name_scope('RN2'):
            true_image = tf.reshape(self.d_RN2,      [-1, args.output_size, args.output_size, 3])
            self.summ_rn2_real = tf.summary.image('rn2_orig', true_image[0:4], 4)
            true_image = tf.reshape(self.d_reco_rn2, [-1, args.output_size, args.output_size, 3])
            self.summ_rn2_reco = tf.summary.image('rn2_reco', true_image[0:4], 4)
            true_image = tf.reshape(self.d_cyc_rn2, [-1, args.output_size, args.output_size, 3])
            self.summ_rn2_cyc = tf.summary.image('rn2_cyc', true_image[0:4], 4)
            #true_image = tf.reshape(self.d_ORN2, [-1, args.output_size, args.output_size, 3])
            #self.summ_orn2_real = tf.summary.image('orn2_real', true_image[0:4], 4)


        self.summ_merge = tf.summary.merge_all()

        # Extract variables
        self.var_encoder = tl.layers.get_variables_with_name('ENCODER', True, True)
        self.var_decoder = tl.layers.get_variables_with_name('DECODER', True, True)
        self.var_dis     = tl.layers.get_variables_with_name('DISC',    True, True)

        self.var_cycle   = self.var_encoder
        self.var_cycle.extend(self.var_decoder)

    def feed_datas(self, data_iter):
        batch_r0_images, batch_rp1_images, batch_rp2_images, batch_rn1_images, batch_rn2_images, batch_or0_images, batch_orp1_images, batch_orp2_images, batch_orn1_images, batch_orn2_images = data_iter()
        batch_codes = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.code_dim)).astype(np.float32)

        feed_dict={self.d_R0: batch_r0_images, self.d_OR0: batch_or0_images, \
                   self.d_RP1: batch_rp1_images, self.d_RP2: batch_rp2_images, self.d_RN1: batch_rn1_images, self.d_RN2: batch_rn2_images, \
                   self.d_ORP1: batch_orp1_images, self.d_ORP2: batch_orp2_images, self.d_ORN1: batch_orn1_images, self.d_ORN2: batch_orn2_images, \
                   self.d_fake_c: batch_codes}

        return feed_dict

    def train(self, args):
        
        # Set optimal for cycle updating
        self.cyc_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_cycle * args.cycle, var_list=self.var_cycle)
        self.head_diff_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                       .minimize(self.loss_head_diff *  args.head_diff, var_list=self.var_cycle)
        self.maha_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                             .minimize(self.loss_maha * args.maha, var_list=self.var_encoder)

        # Set optimal for discriminator
        self.gen_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_gen, var_list=self.var_decoder)
        self.dis_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_dis, var_list=self.var_dis)

        # Initial layer's variables
        tl.layers.initialize_global_variables(self.sess)
        if args.restore == True:
            self.loadParam(args)
            print("[*] Load network done")
        else:
            print("[!] Initial network done")

        # Initial global variables
        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # For new_loam dataset
        if args.dataset == 'new_loam':
            data_dir = ['01', '02', '03', '04','05', '06','07', '08', '09']

        # For NCTL dataset            
        if args.dataset == 'NCTL':
            data_dir = ['2012-01-08', '2012-01-15', '2012-01-22']

        rn2_files = []
        rn1_files = []
        r0_files  = []
        rp1_files = []
        rp2_files = []

        for data_id, data_name in enumerate(data_dir):
            
            print (data_name)
            files = glob(os.path.join(args.data_dir, args.dataset, data_name, 'R0.5/img/*.jpg'))
            files.sort()
            rn2_files += files[200:]

            files = glob(os.path.join(args.data_dir, args.dataset, data_name, 'R1.0/img/*.jpg'))
            files.sort()
            rn1_files += files[200:]

            files  = glob(os.path.join(args.data_dir, args.dataset, data_name, 'R1.5/img/*.jpg'))
            files.sort()
            r0_files += files[200:]

            files = glob(os.path.join(args.data_dir, args.dataset, data_name, 'R2.0/img/*.jpg'))
            files.sort()
            rp1_files += files[200:]

            files = glob(os.path.join(args.data_dir, args.dataset, data_name, 'R2.5/img/*.jpg'))
            files.sort()
            rp2_files += files[200:]

        print ("Bag length is {}".format(len(r0_files)))

        self.iter_counter = 0
        data_iter = self.data_iter(args, rn2_files, rn1_files, r0_files, rp1_files, rp2_files)
        # Main loop for Training
        for it_loop in range(0, args.iter_num):

            ### Update Nets ###
            start_time = time.time()
            feed_dict = self.feed_datas(data_iter)
            ## Discriminator
            #err_dis,  _ = self.sess.run([self.loss_dis,    self.dis_optim],  feed_dict=feed_dict)
            
            ## Generator
            #for gen_loop in range(args.g_iter):
            #    err_gen, _ = self.sess.run([self.loss_gen, self.gen_optim], feed_dict=feed_dict)
            err_dis = 0
            err_gen = 0

            ## Cycle updating    
            err_cyc, err_head, err_trans, err_same, err_other, err_maha, _, _, _ = \
                                    self.sess.run([self.loss_cycle, self.loss_head_diff, self.loss_trans, self.loss_same_center, \
                                                   self.loss_other_center, self.loss_maha, self.cyc_optim, self.maha_optim, self.head_diff_optim], \
                                                  feed_dict=feed_dict)

            print("Loop: [%2d/%2d] time: %4.4f, sim_same: %4.4f, sim_other: %4.4f, cyc: %4.4f, head: %4.4f, trans: %4.4f, maha: %4.4f"  % \
                  (it_loop, args.iter_num, time.time() - start_time, \
                   err_same, err_other, err_cyc, err_head, err_trans, err_maha))
            sys.stdout.flush()
            self.iter_counter += 1
            
            if np.mod(self.iter_counter, args.sample_step) == 0:
                summary = self.sess.run([self.summ_merge], feed_dict=self.feed_datas(data_iter))
                self.writer.add_summary(summary[0], self.iter_counter)
                
            if np.mod(self.iter_counter, args.save_step) == 0:
                self.saveParam(args)
                print("[*] Saving checkpoints SUCCESS!")

        # Shutdown writer
        self.writer.close()

    def test(self, args):

        test_dir = ["R0.50", "R0.75", "R1.00", "R1.25", "R1.50", "R1.75", "R2.00", "R2.25", "R2.50", "R3.00"]

        # For new_loam dataset
        if args.dataset == 'new_loam':
            sequence_name = '00'

        # For NCTL dataset            
        if args.dataset == 'NCTL':
            sequence_name = '2012-02-02'

        for test_epoch in range(16, 22):

            # Initial layer's variables
            self.test_epoch = test_epoch
            self.loadParam(args)
            print("[*] Load network done")

            for dir_id, dir_name in enumerate(test_dir):

                ## Evaulate train data
                test_path = os.path.join(args.data_dir, args.dataset, sequence_name, dir_name, "img/*.jpg")
                test_files = glob(test_path)
                test_files.sort()
                
                print (test_path)
                print (len(test_files))
                ## Extract Train data code
                test_code  = np.zeros([args.test_len, 5, args.code_dim]).astype(np.float32)
                count = 0
                time_sum = 0
                time_min = 10000
                time_max = -1.0
                for id in range(2000, len(test_files)):

                    start_time = time.time()
                    if id%args.frame_skip != 0:
                        continue

                    sample_file = test_files[id]
                    sample = get_image(sample_file, args.image_size, is_crop=args.is_crop, \
                                       resize_w=args.output_size, is_grayscale=0)
                    sample_image = np.array(sample).astype(np.float32)
                    sample_image = sample_image.reshape([1,64,64,3])
                    #print ("Load data {}".format(sample_file))
                    feed_dict={self.d_R0: sample_image}
                    if count >= args.test_len:
                        break

                    code_rn2, code_rn1, code_r0, code_rp1, code_rp2 = self.sess.run([self.d_code_rn2, self.d_code_rn1, self.d_code_r0, \
                                                                                     self.d_code_rp1, self.d_code_rp2], feed_dict=feed_dict)
                    test_code[count, 0] = code_rn2
                    test_code[count, 1] = code_rn1
                    test_code[count, 2] = code_r0
                    test_code[count, 3] = code_rp1
                    test_code[count, 4] = code_rp2

                    count = count+1
                    time_len = time.time() - start_time
                    time_sum += time_len

                    if time_max < time_len:
                        time_max = time_len
                    if time_min > time_len:
                        time_min = time_len
                            
                print("For {}".format(dir_name))
                print("Average time: %4.4f"  % (time_sum/args.test_len))
                print("Min time: %4.4f"  % time_min)
                print("Max time: %4.4f"  % time_max)
                print("save file {}".format(str(test_epoch)+'_'+dir_name+'_vt.npy'))
                GTvector_path = os.path.join(args.result_dir, str(test_epoch)+'_'+dir_name+'_vt.npy')
                np.save(GTvector_path, test_code)


    def loadParam(self, args):
        # load the latest checkpoints
        if args.is_train == True:
            load_de  = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                         name='/net_de_%d.npz' % args.c_epoch)
            load_en  = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                         name='/net_en_%d.npz' % args.c_epoch)
            load_cls = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                         name='/net_cls_%d.npz' % args.c_epoch)
            load_dis = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                         name='/net_dis_%d.npz' % args.c_epoch)
            tl.files.assign_params(self.sess, load_en, self.n_c_A)
            tl.files.assign_params(self.sess, load_de, self.n_h_B)
            tl.files.assign_params(self.sess, load_cls, self.n_f_A)
            tl.files.assign_params(self.sess, load_dis, self.n_dis_h_A)
        else:
            load_en = tl.files.load_npz(path=args.checkpoint_dir, name='/net_en_%d00.npz' % self.test_epoch)
            load_de = tl.files.load_npz(path=args.checkpoint_dir, name='/net_de_%d00.npz' % self.test_epoch)
            tl.files.assign_params(self.sess, load_en, self.n_real_c)
            tl.files.assign_params(self.sess, load_de, self.n_reco_r0)


    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        save_dir = args.checkpoint_dir
        print (args.checkpoint_dir)

        # this version is for future re-check and visualization analysis
        net_en_iter_name  = os.path.join(save_dir, 'net_en_%d.npz' % self.iter_counter)
        net_de_iter_name  = os.path.join(save_dir, 'net_de_%d.npz' % self.iter_counter)
        net_dis_iter_name = os.path.join(save_dir, 'net_dis_%d.npz' % self.iter_counter)

        tl.files.save_npz(self.n_real_c.all_params,    name=net_en_iter_name,  sess=self.sess)
        tl.files.save_npz(self.n_reco_r0.all_params,   name=net_de_iter_name,  sess=self.sess)
        tl.files.save_npz(self.n_dis_orig.all_params,  name=net_dis_iter_name, sess=self.sess)
