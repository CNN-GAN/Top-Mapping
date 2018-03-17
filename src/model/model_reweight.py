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
import src.module.module   as module2D
import src.module.module3D as module3D
from src.module.module_reweight import *
from src.util.utils import *
from src.data import *

class Net_REWEIGHT(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        self.LSTM_Encoder = LSTM_Encoder

        # ALI approach
        self.model    = args.method
        self.is_train = args.is_train 

        # Data iterator
        self.data_iter = TriDataSampler

        # SeqSLAM
        self.vec_D    = Euclidean

        # Test
        if args.is_train == False:
            self.test_epoch = 0

        self._build_model(args)

    def _build_model(self, args):
        self.cc_img = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='cc_img')
        self.cn_img = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='cn_img')
        self.cf_img = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='cf_img')
        self.cc_pcd = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='cc_pcd')
        self.cn_pcd = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='cn_pcd')
        self.cf_pcd = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='cf_pcd')

        # Add LSTMs to estimate the weighting of each kind of feature
        self.n_img_encoder, self.w_img = self.LSTM_Encoder(self.cc_img, is_train=True, reuse=False, name="IMG_W")
        self.n_pcd_encoder, self.w_pcd = self.LSTM_Encoder(self.cc_pcd, is_train=True, reuse=False, name="PCD_W")

        self.weights = tf.concat([self.w_img, self.w_pcd])
        self.weights = tf.nn.softmax(self.weights)

        self.cc_code = tf.concat([tf.scalar_mul(self.weights[0], self.cc_img), tf.scalar_mul(self.weights[1], self.cc_pcd)], 0)
        self.cn_code = tf.concat([tf.scalar_mul(self.weights[0], self.cn_img), tf.scalar_mul(self.weights[1], self.cn_pcd)], 0)
        self.cf_code = tf.concat([tf.scalar_mul(self.weights[0], self.cf_img), tf.scalar_mul(self.weights[1], self.cf_pcd)], 0)

        # Apply Mahalanobis loss
        self.loss_near = tl.cost.mean_squared_error(self.cc_code, self.cn_code)
        self.loss_far  = tl.cost.mean_squared_error(self.cc_code, self.cf_code)

        # max(0, 1-||ht-ht_n||/(||ht-ht_1||+beta))
        self.loss_maha = tf.nn.relu(tf.sub(1, tf.div(self.loss_far, tf.add(self.loss_near, args.distance_threshold))))

        # Make summary
        with tf.name_scope('LSTM_Encoder'):
            self.summ_maha  = tf.summary.scalar('loss_maha',  self.loss_maha)
            self.summ_w_img = tf.summary.scalar('weight_img', self.weights[0])

        self.summ_merge = tf.summary.merge_all()

        # Extract variables
        self.var_LSTM_IMG = tl.layers.get_variables_with_name('IMG_W',  True, True)
        self.var_LSTM_PCD = tl.layers.get_variables_with_name('PCD_W',  True, True)
        self.var_LSTM     = self.var_LSTM_IMG
        self.var_LSTM.extend(self.var_LSTM_PCD)

    def feed_datas(self, data_iter, noise_iter):
        cc_img, cc_pcd, cn_img, cn_pcd, cf_img, cf_pcd = data_iter()
        feed_dict={self.cc_img: cc_img, self.cc_pcd: cc_pcd, \
                   self.cn_img: cc_pcd, self.cn_pcd: cc_pcd, \
                   self.cf_img: cc_pcd, self.cf_pcd: cc_pcd}

        feed_dict.update(self.n_img_encoder.all_drop)
        feed_dict.update(self.n_pcd_encoder.all_drop)
            
        return feed_dict

    def train(self, args):
        
        # Set optimal for nets
        self.optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                          .minimize(self.loss_maha, var_list=self.var_LSTM)
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

        # Load codes
        v_img_path = os.path.join(args.v_img_path, str(args.img_epoch)+'_code_vt.npy')
        v_img = np.load(v_img_path)
        v_pcd_path = os.path.join(args.v_pcd_path, str(args.pcd_epoch)+'_code_vt.npy')
        v_pcd = np.load(v_pcd_path)

        # For new_loam dataset
        if args.dataset == 'new_loam':
            data_dir = ['01', '02', '03', '04','05', '06','07', '08']

        # For NCTL dataset            
        if args.dataset == 'NCTL':
            data_dir = ['2012-01-08', '2012-01-15', '2012-01-22']

        pose_path = os.path.join(args.data_dir, args.dataset, data_dir[1], "pose.txt")
        v_pose = np.loadtxt(v_pose_path)
        for data_id in range(1, len(data_dir)):
            pose_path = os.path.join(args.data_dir, args.dataset, data_dir[data_id], "pose.txt")
            pose = np.loadtxt(v_pose_path)
            v_pose = np.append(v_pose, pose, axis=0)
    
        data_iter = self.data_iter(args, v_img, v_pcd, v_pose)
        
        # Main loop for Training
        self.iter_counter = 0

        for idx in xrange(0, args.iteration):
            self.iter_counter += 1
            
            ### Update Nets ###
            start_time = time.time()
            feed_dict = self.feed_datas(data_iter, noise_iter)
            err, summ, _ = self.sess.run([self.loss_dicJ, self.summ_merge, self.optim], feed_dict=feed_dict)
            self.writer.add_summary(summ[0], self.iter_counter)
            print("Iteration [%4d] time: %4.4f"  % (self.iter_counter, time.time() - start_time))
            sys.stdout.flush()

            if np.mod(self.iter_counter, args.save_step) == 0:
                self.saveParam(args)
                print("[*] Saving checkpoints SUCCESS!")

        # Shutdown writer
        self.writer.close()


    def loadParam(self, args):
        # load the latest checkpoints    
        load_lstm_en = tl.files.load_npz(path=args.checkpoint_dir, \
                                    name='/net_lstm_en_%d00.npz' % args.c_epoch)
        tl.files.assign_params(self.sess, load_lstm_en, self.n_fake_z)

    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        # the latest version location
        net_en_iter_name = os.path.join(args.checkpoint_dir, 'net_lstm_en_%d.npz' % self.iter_counter)
        tl.files.save_npz(self.n_fake_z.all_params, name=net_en_iter_name, sess=self.sess)
