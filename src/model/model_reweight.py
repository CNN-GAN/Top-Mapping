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
import src.module.module   as module2d
import src.module.module3d as module3d
from src.util.utils import *
from src.data import *

class Net_REWEIGHT(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        # ALI approach
        self.model    = args.method
        self.is_train = args.is_train 
        
        # Network module
        self.encoder2D = module2d.encoder
        self.encoder3D = module3d.encoder

        # Data iterator
        self.data_iter = DataSampler
        self.noise_iter = NoiseSampler

        # SeqSLAM
        self.vec_D    = Euclidean

        # Test
        if args.is_train == False:
            self.test_epoch = 0
            self._build_model(args)

    def _build_model(self, args):
        self.d_img = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                 args.img_dim], name='real_img')
        self.d_pcd = tf.placeholder(tf.float32, [args.batch_size, args.voxel_size, args.voxel_size, \
                                                 args.voxel_size/8, args.voxel_dim], name='real_pcd')

        self.n_img, self.d_img_c = self.encoder(self.d_img, is_train=False, reuse=False)
        self.n_pcd, self.d_pcd_c = self.encoder(self.d_pcd, is_train=False, reuse=False)

        # Add LSTMs to estimate the weighting of each kind of feature
        
        # Apply loss
        self.loss_encoder = args.side_D * self.lossGAN(self.d_dic_fz, 1)
        self.loss_decoder = args.side_D * self.lossGAN(self.d_dic_fx, 1)
        
        # Make summary
        with tf.name_scope('X_space'):
            self.summ_decoder = tf.summary.scalar('decoder_loss', self.loss_decoder/args.side_D)
            self.summ_dicX    = tf.summary.scalar('d_X_loss',     self.loss_dicX/args.side_D)            

        if self.model == 'ALI_CLC':
            self.summ_merge = tf.summary.merge_all()

        # Extract variables
        self.var_dicJ     = tl.layers.get_variables_with_name('DISC_J',  True, True)
        self.var_gen    = self.var_encoder
        self.var_gen.extend(self.var_decoder)

    def feed_datas(self, data_iter, noise_iter):
        batch_images = data_iter()
        batch_codes = noise_iter()
        feed_dict={self.d_real_x: batch_images, self.d_real_z: batch_codes }
        if self.model == 'ALI_CLC':
            feed_dict.update(self.n_dic_J.all_drop)
            feed_dict.update(self.n_dic_fJ.all_drop)
            feed_dict.update(self.n_dic_z.all_drop)
            feed_dict.update(self.n_dic_fz.all_drop)
        else:
            feed_dict.update(self.n_dic_J.all_drop)
            feed_dict.update(self.n_dic_fJ.all_drop)
            
        return feed_dict
        

    def train(self, args):
        
        # Set optimal for nets
        self.optim_dicX    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                     .minimize(self.loss_dicX,    var_list=self.var_dicX)
        self.optim_dicZ    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                     .minimize(self.loss_dicZ,    var_list=self.var_dicZ)
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

        # Load Data files
        data_dir = ['01', '02', '03', '04','05', '06','07', '08']
        data_files = []
        for data_name in data_dir:
            read_path = os.path.join("./data", args.dataset, data_name, "img/*.jpg")
            print (read_path)
            data_file = glob(read_path)
            data_files = data_files + data_file
            
        print (len(data_files))
        data_iter = self.data_iter(args, data_files)
        noise_iter = self.noise_iter(args)
        
        # Main loop for Training
        self.iter_counter = 0
        begin_epoch = 0
        if args.restore == True:
            begin_epoch = args.c_epoch+1

        for idx in xrange(0, args.iteration):
            
            ### Update Nets ###
            start_time = time.time()
            if self.model == 'ALI_CLC':
                # Update Joint
                feed_dict = self.feed_datas(data_iter, noise_iter)
                #self.sess.run(self.clip_J)
                errJ, _ = self.sess.run([self.loss_dicJ,   self.optim_dicJ],    feed_dict=feed_dict)
                for g_id in range(args.g_iter):
                    errfJ, _  = self.sess.run([self.loss_dicfJ, self.optim_dicfJ], feed_dict=feed_dict)
                    errClc, _ = self.sess.run([self.loss_cycle, self.optim_cycle], feed_dict=feed_dict)


                errX, _ = self.sess.run([self.loss_dicX,   self.optim_dicX],    feed_dict=feed_dict)
                errD, _   = self.sess.run([self.loss_decoder, self.optim_decoder], feed_dict=feed_dict)

            elif self.model == 'ALI':
                feed_dict = self.feed_datas(data_iter, noise_iter)
                errJ, _ = self.sess.run([self.loss_dicJ,   self.optim_dicJ],    feed_dict=feed_dict)
                for _ in range(args.g_iter):
                    errfJ, _  = self.sess.run([self.loss_dicfJ, self.optim_dicfJ], feed_dict=feed_dict)

            print("Iteration [%4d] time: %4.4f"  % \
                  (self.iter_counter, time.time() - start_time))
            sys.stdout.flush()
            self.iter_counter += 1
            
            if np.mod(self.iter_counter, args.sample_step) == 0:
                self.makeSample(feed_dict, args.sample_dir, self.iter_counter)
                
            if np.mod(self.iter_counter, args.save_step) == 0:
                self.saveParam(args)
                print("[*] Saving checkpoints SUCCESS!")

        # Shutdown writer
        self.writer.close()
