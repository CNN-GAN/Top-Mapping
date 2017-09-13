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
from src.module.module import *
from src.util.utils import *
from src.data import *

class Net_BIGAN_GTAV(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        # ALI approach
        self.model    = args.method
        self.is_train = args.is_train 
        
        # Network module
        self.encoder = encoder
        self.decoder = decoder
        self.discX   = discriminator_X
        self.discZ   = discriminator_Z
        self.discJ   = discriminator_J
        
        # Loss function
        self.lossGAN = mae_criterion
        self.lossCYC = mae_criterion

        # Data iterator
        self.data_iter = DataSampler
        self.noise_iter = NoiseSampler

        # SeqSLAM
        self.vec_D    = Euclidean
        if args.Search == 'N':
            self.getMatch = getMatches
        else:
            self.getMatch = getAnnMatches

        # Test
        if args.is_train == False:
            self.test_epoch = 0
        self._build_model(args)

    def _build_model(self, args):
        self.d_real_x = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                      args.img_dim], name='real_image')
        self.d_real_z  = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name="real_code")

        self.n_fake_x, self.d_fake_x = self.decoder(self.d_real_z, is_train=True, reuse=False)
        self.n_fake_z, self.d_fake_z = self.encoder(self.d_real_x, is_train=True, reuse=False)
        self.n_cycl_z, self.d_cycl_z = self.encoder(self.d_fake_x, is_train=True, reuse=True)
        self.n_cycl_x, self.d_cycl_x = self.decoder(self.d_fake_z, is_train=True, reuse=True)

        with tf.name_scope('real'):
            true_image = tf.reshape(self.d_real_x, [-1, 64, 64, 3])
            self.summ_image_real = tf.summary.image('real', true_image[0:4], 4)

        with tf.name_scope('fake'):
            fake_image = tf.reshape(self.d_cycl_x, [-1, 64, 64, 3])
            self.summ_image_fake = tf.summary.image('fake', fake_image[0:4], 4)

        self.n_dic_x,  self.d_dic_x  = self.discX(self.d_real_x, is_train=True, reuse=False)
        self.n_dic_fx, self.d_dic_fx = self.discX(self.d_fake_x, is_train=True, reuse=True)
        self.n_dic_z,  self.d_dic_z  = self.discZ(self.d_real_z, is_train=True, reuse=False)
        self.n_dic_fz, self.d_dic_fz = self.discZ(self.d_fake_z, is_train=True, reuse=True)
        self.n_dic_J,  self.d_dic_J  = self.discJ(self.d_real_x, self.d_fake_z, is_train=True, reuse=False)
        self.n_dic_fJ, self.d_dic_fJ = self.discJ(self.d_fake_x, self.d_real_z, is_train=True, reuse=True)

        self.loss_dicJ    = 0.5 * (self.lossGAN(self.d_dic_J, 1) + self.lossGAN(self.d_dic_fJ, 0))
        self.loss_dicfJ   = 0.5 * (self.lossGAN(self.d_dic_J, 0) + self.lossGAN(self.d_dic_fJ, 1))
        
        # Make summary
        with tf.name_scope('J_space'):
            self.summ_dicJ    = tf.summary.scalar('d_J_loss',     self.loss_dicJ)
            self.summ_dicfJ   = tf.summary.scalar('d_fJ_loss',    self.loss_dicfJ)
            
        self.summ_merge = tf.summary.merge([self.summ_image_real, self.summ_image_fake, \
                                            self.summ_dicJ, self.summ_dicfJ])  
        # Extract variables
        self.var_encoder  = tl.layers.get_variables_with_name('ENCODER', True, True)
        self.var_decoder  = tl.layers.get_variables_with_name('DECODER', True, True)
        self.var_dicX     = tl.layers.get_variables_with_name('DISC_X',  True, True)
        self.var_dicZ     = tl.layers.get_variables_with_name('DISC_Z',  True, True)
        self.var_dicJ     = tl.layers.get_variables_with_name('DISC_J',  True, True)
        self.var_gen    = self.var_encoder
        self.var_gen.extend(self.var_decoder)


    def feed_datas(self, data_iter, noise_iter):
        batch_images = data_iter()
        batch_codes = noise_iter()
        feed_dict={self.d_real_x: batch_images, self.d_real_z: batch_codes }
        
        feed_dict.update(self.n_dic_J.all_drop)
        feed_dict.update(self.n_dic_fJ.all_drop)
            
        return feed_dict
                                

    def train(self, args):
        
        # Set optimal for nets
        self.optim_dicJ    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                     .minimize(self.loss_dicJ,    var_list=self.var_dicJ)
        self.optim_dicfJ   = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                     .minimize(self.loss_dicfJ,   var_list=self.var_gen)
        # Initial layer's variables
        tl.layers.initialize_global_variables(self.sess)
        if args.restore == True:
            self.loadParam(args)
            print("[*] Load network done")
        else:
            print("[!] Initial network done")

        # Initial global variables
        log_dir = os.path.join(args.log_dir, args.method, args.log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)     
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        # Load Data files
        route_names = ['Route1', 'Route2']
        data_names = ['FOGGY1', 'FOGGY2', 'SUNNY1', 'SUNNY2', 'RAIN1', 'RAIN2']
        data_files = []

        for data_id, data_name in enumerate(data_names):
            for route_id, route_name in enumerate(route_names):
                read_path = os.path.join("./data/GTAV", route_name, data_name, "*.jpg")
                tmp_file = glob(read_path)
                data_files += tmp_file

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

    def test(self, args):

        route_dir = ["Route1", "Route2"]
        test_dir = ["FOGGY", "RAIN", "SUNNY"]
        result_dir = os.path.join(args.result_dir, args.method, args.log_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for test_epoch in range(1, 18):

            # Initial layer's variables
            self.test_epoch = test_epoch
            self.loadParam(args)
            print("[*] Load network done")

            for route_index, route_name in enumerate(route_dir):

                for dir_index, file_name in enumerate(test_dir):

                    ## Evaulate test data
                    test_files  = glob(os.path.join(args.data_dir, args.dataset, route_name, file_name, "*.jpg"))
                    test_files.sort()
                    test_files = test_files[0:args.test_len]
                    
                    ## Extract Test data code
                    test_code = np.zeros([len(test_files), 512]).astype(np.float32)
                    
                    count = 0
                    time_sum = 0
                    time_min = 10000
                    time_max = -1.0
                    
                    for id in range(len(test_files)):
                        
                        start_time = time.time()
                        if id%args.frame_skip != 0:
                            continue
                            
                        sample_file = test_files[id]
                        sample = get_image(sample_file, args.image_size, is_crop=args.is_crop, \
                                    resize_w=args.output_size, is_grayscale=0)
                        sample_image = np.array(sample).astype(np.float32)
                        sample_image = sample_image.reshape([1,64,64,3])
                        feed_dict={self.d_real_x: sample_image}
                        if count >= args.test_len:
                            break

                        test_code[count]  = self.sess.run(self.d_fake_z, feed_dict=feed_dict)
                        count = count+1
                        time_len = time.time() - start_time
                        time_sum += time_len
                        
                        if time_max < time_len:
                            time_max = time_len
                        if time_min > time_len:
                            time_min = time_len

                    print("save file {}".format(str(test_epoch)+'_'+route_name+'_'+file_name+'_vt.npy'))                
                    print("Average time: %4.4f"  % (time_sum/args.test_len))
                    print("Min time: %4.4f"  % time_min)
                    print("Max time: %4.4f"  % time_max)
                    GTvector_path = os.path.join(result_dir, str(test_epoch)+'_'+route_name+'_'+file_name+'_vt.npy')
                    np.save(GTvector_path, test_code)

    def reconstruct(self, args):

        result_dir = os.path.join(args.result_dir, args.method, args.log_name, 'reconstruct')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for test_epoch in range(22, 23):

            # Initial layer's variables
            self.test_epoch = test_epoch
            self.loadParam(args)
            print("[*] Load network done")

            ## Evaulate train data
            train_files = glob(os.path.join(args.data_dir, args.dataset, "00", "gt/img/*.jpg"))
            train_files.sort()
            
            ## Extract Train data code
            for id in range(1, 400):
                if id%5 != 0:
                    continue

                sample_file = train_files[id]
                sample = get_image(sample_file, args.image_size, is_crop=args.is_crop, \
                               resize_w=args.output_size, is_grayscale=0)
                sample_image = np.array(sample).astype(np.float32)
                sample_image = sample_image.reshape([1,64,64,3])
                print ("Load data {}".format(sample_file))
                feed_dict={self.d_real_x: sample_image}

                cycl_img = self.sess.run([self.n_cycl_x.outputs], feed_dict=feed_dict)

                # save image
                cycl_img = (np.array(cycl_img) + 1) / 2 * 255
                cycl_img = cycl_img.reshape((64, 64, 3))
                sample_image = sample_image.reshape((64, 64, 3))
                print (cycl_img.shape)
                print (sample_image.shape)
                scipy.misc.imsave('{}/{}_origin_{:04d}.png'.format(result_dir, str(test_epoch), id), sample_image)
                scipy.misc.imsave('{}/{}_cycle_{:04d}.png'.format(result_dir, str(test_epoch), id), cycl_img)

        
    def makeSample(self, feed_dict, sample_dir, idx):
        summary, img = self.sess.run([self.summ_merge, self.n_fake_x.outputs], feed_dict=feed_dict)

        # update summary
        self.writer.add_summary(summary, self.iter_counter)
        # save image
        img = (np.array(img) + 1) / 2 * 255
        save_images(img, [8, 8],'./{}/train_{:04d}.png'.format(sample_dir, idx))

    def loadParam(self, args):
        # load the latest checkpoints
        if args.is_train == True:
            load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_de_%d00.npz' % args.c_epoch)
            load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_en_%d00.npz' % args.c_epoch)
            load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_dJ_%d00.npz' % args.c_epoch)
            tl.files.assign_params(self.sess, load_en, self.n_fake_z)
            tl.files.assign_params(self.sess, load_de, self.n_fake_x)
            tl.files.assign_params(self.sess, load_dJ, self.n_dic_J)
        else:
            load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_de_%d00.npz' % self.test_epoch)
            load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_en_%d00.npz' % self.test_epoch)
            load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_dJ_%d00.npz' % self.test_epoch)
            tl.files.assign_params(self.sess, load_en, self.n_fake_z)
            tl.files.assign_params(self.sess, load_de, self.n_fake_x)
            tl.files.assign_params(self.sess, load_dJ, self.n_dic_J)

    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        save_dir = os.path.join(args.checkpoint_dir, args.method, args.log_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
        print (save_dir)

        # the latest version location
        net_de_name = os.path.join(save_dir, 'net_de.npz')
        net_en_name = os.path.join(save_dir, 'net_en.npz')
        net_dJ_name = os.path.join(save_dir, 'net_dJ.npz')
        # this version is for future re-check and visualization analysis
        net_de_iter_name = os.path.join(save_dir, 'net_de_%d.npz' % self.iter_counter)
        net_en_iter_name = os.path.join(save_dir, 'net_en_%d.npz' % self.iter_counter)
        net_dJ_iter_name = os.path.join(save_dir, 'net_dJ_%d.npz' % self.iter_counter)
        
        tl.files.save_npz(self.n_fake_x.all_params, name=net_de_name, sess=self.sess)
        tl.files.save_npz(self.n_fake_z.all_params, name=net_en_name, sess=self.sess)
        tl.files.save_npz(self.n_dic_J.all_params,  name=net_dJ_name, sess=self.sess)
        
        tl.files.save_npz(self.n_fake_x.all_params, name=net_de_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_fake_z.all_params, name=net_en_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_dic_J.all_params,  name=net_dJ_iter_name, sess=self.sess)
