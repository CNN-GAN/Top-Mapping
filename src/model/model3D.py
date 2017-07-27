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
from src.module.module3D import *
from src.util.utils import *

class Net3D(object):
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
        if args.Loss == 'WGAN':
            self.lossGAN = wasserstein_criterion
        elif args.Loss == 'LSGAN':
            self.lossGAN = mae_criterion

        self.lossCYC = abs_criterion

        # SeqSLAM
        self.vec_D    = Euclidean
        if args.match_method == 'ANN':
            self.getMatch = getAnnMatches
        else:
            self.getMatch = getMatches

        # Test
        if args.is_train == False:
            self.test_epoch = 0

        self._build_model(args)

    def _build_model(self, args):
        self.d_real_x = tf.placeholder(tf.float32, [args.batch_size, args.voxel_size, args.voxel_size, \
                                                    args.voxel_size/8, args.voxel_dim], name='real_pcd')
        self.d_real_z  = tf.placeholder(tf.float32, [args.batch_size, args.voxel_code], name="real_code")

        self.n_fake_x, self.d_fake_x = self.decoder(self.d_real_z, is_train=True, reuse=False)
        self.n_fake_z, self.d_fake_z = self.encoder(self.d_real_x, is_train=True, reuse=False)
        self.n_cycl_x, self.d_cycl_x = self.decoder(self.d_fake_z, is_train=True, reuse=True)
        self.n_cycl_z, self.d_cycl_z = self.encoder(self.d_fake_x, is_train=True, reuse=True)


        self.n_dic_x,  self.d_dic_x  = self.discX(self.d_real_x, is_train=True, reuse=False)
        self.n_dic_fx, self.d_dic_fx = self.discX(self.d_fake_x, is_train=True, reuse=True)
        self.n_dic_z,  self.d_dic_z  = self.discZ(self.d_real_z, is_train=True, reuse=False)
        self.n_dic_fz, self.d_dic_fz = self.discZ(self.d_fake_z, is_train=True, reuse=True)
        self.n_dic_J,  self.d_dic_J  = self.discJ(self.d_real_x, self.d_fake_z, is_train=True, reuse=False)
        self.n_dic_fJ, self.d_dic_fJ = self.discJ(self.d_fake_x, self.d_real_z, is_train=True, reuse=True)

        # Apply Loss
        self.loss_encoder = args.side_D * self.lossGAN(self.d_dic_fz, 1)
        self.loss_decoder = args.side_D * self.lossGAN(self.d_dic_fx, 1)

        self.loss_cycle   = args.cycle * (self.lossCYC(self.d_real_x, self.d_cycl_x) + \
                                          self.lossCYC(self.d_real_z, self.d_cycl_z))

        self.loss_dicJ    = tf.reduce_mean(self.d_dic_J - self.d_dic_fJ)
        self.loss_dicfJ   = tf.reduce_mean(self.d_dic_fJ - self.d_dic_J)

        self.loss_dicX    = args.side_D*0.5*(self.lossGAN(self.d_dic_x, 1) + \
                                             self.lossGAN(self.d_dic_fx,0))
        self.loss_dicZ    = args.side_D*0.5*(self.lossGAN(self.d_dic_z, 1) + \
                                             self.lossGAN(self.d_dic_fz,0))

        # Make summary
        with tf.name_scope('Joint'):
            self.summ_dicJ    = tf.summary.scalar('d_J_loss',     self.loss_dicJ)
            self.summ_dicfJ   = tf.summary.scalar('d_fJ_loss',    self.loss_dicfJ)

        with tf.name_scope('DicX'):
            self.summ_dicX    = tf.summary.scalar('d_X_loss',     self.loss_dicX)
            self.summ_decoder = tf.summary.scalar('decoder_loss', self.loss_decoder)

        with tf.name_scope('DicZ'):        
            self.summ_dicZ    = tf.summary.scalar('d_Z_loss',     self.loss_dicZ)
            self.summ_encoder = tf.summary.scalar('encoder_loss', self.loss_encoder)


        self.summ_cycle   = tf.summary.scalar('clc_loss',     self.loss_cycle)

        if self.model == 'ALI_CLC':
            self.summ_merge = tf.summary.merge_all()
        elif self.model == 'ALI':
            self.summ_merge = tf.summary.merge([self.summ_dicJ, self.summ_dicfJ])

        # Extract variables
        self.var_encoder  = tl.layers.get_variables_with_name('ENCODER', True, True)
        self.var_decoder  = tl.layers.get_variables_with_name('DECODER', True, True)
        self.var_dicX     = tl.layers.get_variables_with_name('DISC_X',  True, True)
        self.var_dicZ     = tl.layers.get_variables_with_name('DISC_Z',  True, True)
        self.var_dicJ     = tl.layers.get_variables_with_name('DISC_J',  True, True)
        self.var_gen      = self.var_encoder
        self.var_gen.extend(self.var_decoder)

    def train(self, args):
        
        # Set optimal for nets
        if self.model == 'ALI_CLC':
            self.optim_encoder = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                         .minimize(self.loss_encoder, var_list=self.var_encoder)
            self.optim_decoder = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                         .minimize(self.loss_decoder, var_list=self.var_decoder)
            self.optim_cycle   = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                         .minimize(self.loss_cycle,   var_list=self.var_gen)
            self.optim_dicX    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                         .minimize(self.loss_dicX,    var_list=self.var_dicX)
            self.optim_dicZ    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                         .minimize(self.loss_dicZ,    var_list=self.var_dicZ)

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
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)        

        # Load Data files
        data_dir = ['01', '02', '03', '04','05', '06','07', '08']
        data_files = []
        for data_name in data_dir:
            read_path = os.path.join("./data", args.dataset, data_name, "pcd/*.pcd")
            print (read_path)
            data_file = glob(read_path)
            data_files = data_files + data_file
            
        print (len(data_files))
        # Main loop for Training
        self.iter_counter = 0
        begin_epoch = 0
        if args.restore == True:
            begin_epoch = args.c_epoch+1
            self.iter_counter = 250

        for epoch in range(begin_epoch, args.epoch):
            ## shuffle data
            shuffle(data_files)
            print("[*] Dataset shuffled!")
            
            ## load image data
            batch_idxs = min(len(data_files), args.train_size) // args.batch_size
            
            for idx in xrange(0, batch_idxs):
                ### Get datas ###
                batch_files  = data_files[idx*args.batch_size:(idx+1)*args.batch_size]
                ## get real pcds
                batch        = [get_pcd(batch_file, args) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                ## get real code
                batch_codes  = np.random.normal(loc=0.0, scale=1.0, \
                                                size=(args.batch_size, args.code_dim)).astype(np.float32)
                
                ### Update Nets ###
                start_time = time.time()
                feed_dict={self.d_real_x: batch_images, self.d_real_z: batch_codes }
                feed_dict.update(self.n_dic_J.all_drop)
                feed_dict.update(self.n_dic_fJ.all_drop)

                if self.model == 'ALI_CLC':
                    feed_dict.update(self.n_dic_z.all_drop)
                    feed_dict.update(self.n_dic_fz.all_drop)
                    errX, _ = self.sess.run([self.loss_dicX,   self.optim_dicX],    feed_dict=feed_dict)
                    errZ, _ = self.sess.run([self.loss_dicZ,   self.optim_dicZ],    feed_dict=feed_dict)
                    errJ, _ = self.sess.run([self.loss_dicJ,   self.optim_dicJ],    feed_dict=feed_dict)

                    errE, _   = self.sess.run([self.loss_encoder, self.optim_encoder], feed_dict=feed_dict)
                    errD, _   = self.sess.run([self.loss_decoder, self.optim_decoder], feed_dict=feed_dict)

                    ## updates the Joint Generator multi times to avoid Discriminator converge early
                    for e_id in range(8):
                        errfJ, _  = self.sess.run([self.loss_dicfJ, self.optim_dicfJ], feed_dict=feed_dict)

                    ## update inverse mapping
                    errClc, _ = self.sess.run([self.loss_cycle, self.optim_cycle], feed_dict=feed_dict)
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, J_loss: %.8f, fJ_loss: %.8f,  clc_loss: %.8f"  % \
                              (epoch, args.epoch, idx, batch_idxs, time.time() - start_time, errJ, errfJ, errClc))

                    sys.stdout.flush()

                elif self.model == 'ALI':
                    errJ, _ = self.sess.run([self.loss_dicJ,   self.optim_dicJ],    feed_dict=feed_dict)

                    ## updates the Joint Generator multi times to avoid Discriminator converge early
                    for _ in range(8):
                        errfJ, _  = self.sess.run([self.loss_dicfJ, self.optim_dicfJ], feed_dict=feed_dict)

                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, J_loss: %.8f, fJ_loss: %.8f"  % \
                          (epoch, args.epoch, idx, batch_idxs, time.time() - start_time,  errJ, errfJ))
                    sys.stdout.flush()


                self.iter_counter += 1

                if np.mod(self.iter_counter, args.sample_step) == 0:
                    self.makeSample(feed_dict, args.sample_dir, epoch, idx)
                    
                if np.mod(self.iter_counter, args.save_step) == 0:
                    self.saveParam(args)
                    print("[*] Saving checkpoints SUCCESS!")

        # Shutdown writer
        self.writer.close()

    def test(self, args):

        result_model = self.model + '_3D'
        result_dir = os.path.join(args.result_dir, result_model)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        test_dir = ["T1_R0.1", "T1_R0.5", "T1_R1", "T1_R1.5", "T1_R2", "T5_R1", "T10_R1"] 
        for test_id in range(1, 7):

            # Initial layer's variables
            test_epoch = test_id * 50
            self.test_epoch = test_epoch
            self.loadParam(args)
            print("[*] Load network done")

            ## Evaulate train data
            train_files = glob(os.path.join(args.data_dir, args.dataset, '00', "pcd/*.pcd"))
            train_files.sort()

            ## Extract Train data code
            start_time = time.time()
            train_code  = np.zeros([args.test_len, 512]).astype(np.float32)
            count = 0
            for id in range(len(train_files)):
                if id%args.frame_skip != 0:
                    continue

                sample_file = train_files[id]
                sample = get_pcd(sample_file, args)
                sample_image = np.array(sample).astype(np.float32)
                sample_image = sample_image.reshape([1, args.voxel_size, args.voxel_size, \
                                                     int(args.voxel_size/8), 1])
                print ("Load data {}".format(sample_file))
                feed_dict={self.d_real_x: sample_image}
                if count >= args.test_len:
                    break
                train_code[count]  = self.sess.run(self.d_fake_z, feed_dict=feed_dict)
                count = count+1
                
            print("Train code extraction time: %4.4f"  % (time.time() - start_time))


            GTvector_path = os.path.join(result_dir, str(test_epoch)+'_gt_vt.npy')
            np.save(GTvector_path, train_code)

            for dir_id, dir_name in enumerate(test_dir):
                
                ## Evaulate test data
                test_files  = glob(os.path.join(args.data_dir, args.dataset,'00', dir_name, "pcd/*.pcd"))
                test_files.sort()
                
                ## Extract Test data code
                start_time = time.time()
                test_code = np.zeros([args.test_len, 512]).astype(np.float32)
                count = 0
                for id in range(test_code.shape[0]):
                    if id%args.frame_skip != 0:
                        continue

                    sample_file = test_files[id]
                    sample = get_pcd(sample_file, args)
                    sample_image = np.array(sample).astype(np.float32)
                    sample_image = sample_image.reshape([1, args.voxel_size, args.voxel_size, \
                                                         int(args.voxel_size/8), 1])
                    print ("Load data {}".format(sample_file))
                    feed_dict={self.d_real_x: sample_image}
                    if count >= args.test_len:
                        break
                    test_code[count]  = self.sess.run(self.d_fake_z, feed_dict=feed_dict)
                    count = count+1
                    
                print("test code extraction time: %4.4f"  % (time.time() - start_time))
                Testvector_path = os.path.join(result_dir, str(test_epoch)+'_'+dir_name+'_vt.npy')
                np.save(Testvector_path, test_code)


    def makeSample(self, feed_dict, sample_dir, epoch, idx):
        summary = self.sess.run([self.summ_merge], feed_dict=feed_dict)
        self.writer.add_summary(summary, self.iter_counter)

    def loadParam(self, args):
        # load the latest checkpoints
        if self.model == 'ALI_CLC':
            if args.is_3D == True:
                check_path = self.model + '_3D'
            else:
                check_path = self.model

            if args.is_train == True:
                load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_de.npz')
                load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_en.npz')
                load_dX = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dX.npz')
                load_dZ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dZ.npz')
                load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dJ.npz')
                tl.files.assign_params(self.sess, load_en, self.n_fake_z)
                tl.files.assign_params(self.sess, load_de, self.n_fake_x)
                tl.files.assign_params(self.sess, load_dX, self.n_dic_x)
                tl.files.assign_params(self.sess, load_dZ, self.n_dic_z)
                tl.files.assign_params(self.sess, load_dJ, self.n_dic_J)
            else:
                load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_de_%d.npz' % self.test_epoch)
                load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_en_%d.npz' % self.test_epoch)
                load_dX = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dX_%d.npz' % self.test_epoch)
                load_dZ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dZ_%d.npz' % self.test_epoch)
                load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dJ_%d.npz' % self.test_epoch)
                tl.files.assign_params(self.sess, load_en, self.n_fake_z)
                tl.files.assign_params(self.sess, load_de, self.n_fake_x)
                tl.files.assign_params(self.sess, load_dX, self.n_dic_x)
                tl.files.assign_params(self.sess, load_dZ, self.n_dic_z)
                tl.files.assign_params(self.sess, load_dJ, self.n_dic_J)
        elif self.model == 'ALI':
            if args.is_3D == True:
                check_path = self.model + '_3D'
            else:
                check_path = self.model

            if args.is_train == True:
                load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_de_%d00.npz' % args.c_epoch)
                load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_en_%d00.npz' % args.c_epoch)
                load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dJ_%d00.npz' % args.c_epoch)
                tl.files.assign_params(self.sess, load_en, self.n_fake_z)
                tl.files.assign_params(self.sess, load_de, self.n_fake_x)
                tl.files.assign_params(self.sess, load_dJ, self.n_dic_J)
            else:
                load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_de_%d.npz' % self.test_epoch)
                load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_en_%d.npz' % self.test_epoch)
                load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, check_path), \
                                            name='/net_dJ_%d.npz' % self.test_epoch)
                tl.files.assign_params(self.sess, load_en, self.n_fake_z)
                tl.files.assign_params(self.sess, load_de, self.n_fake_x)
                tl.files.assign_params(self.sess, load_dJ, self.n_dic_J)

    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        if args.is_3D == True:
            save_dir = os.path.join(args.checkpoint_dir, args.method+"_3D")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print (save_dir)

        if self.model == 'ALI_CLC':
            # the latest version location
            net_de_name = os.path.join(save_dir, 'net_de.npz')
            net_en_name = os.path.join(save_dir, 'net_en.npz')
            net_dX_name = os.path.join(save_dir, 'net_dX.npz')
            net_dZ_name = os.path.join(save_dir, 'net_dZ.npz')
            net_dJ_name = os.path.join(save_dir, 'net_dJ.npz')
            # this version is for future re-check and visualization analysis
            net_de_iter_name = os.path.join(save_dir, 'net_de_%d.npz' % self.iter_counter)
            net_en_iter_name = os.path.join(save_dir, 'net_en_%d.npz' % self.iter_counter)
            net_dX_iter_name = os.path.join(save_dir, 'net_dX_%d.npz' % self.iter_counter)
            net_dZ_iter_name = os.path.join(save_dir, 'net_dZ_%d.npz' % self.iter_counter)
            net_dJ_iter_name = os.path.join(save_dir, 'net_dJ_%d.npz' % self.iter_counter)
            
            tl.files.save_npz(self.n_fake_x.all_params, name=net_de_name, sess=self.sess)
            tl.files.save_npz(self.n_fake_z.all_params, name=net_en_name, sess=self.sess)
            tl.files.save_npz(self.n_dic_x.all_params,  name=net_dX_name, sess=self.sess)
            tl.files.save_npz(self.n_dic_z.all_params,  name=net_dZ_name, sess=self.sess)
            tl.files.save_npz(self.n_dic_J.all_params,  name=net_dJ_name, sess=self.sess)

            tl.files.save_npz(self.n_fake_x.all_params, name=net_de_iter_name, sess=self.sess)
            tl.files.save_npz(self.n_fake_z.all_params, name=net_en_iter_name, sess=self.sess)
            tl.files.save_npz(self.n_dic_x.all_params,  name=net_dX_iter_name, sess=self.sess)
            tl.files.save_npz(self.n_dic_z.all_params,  name=net_dZ_iter_name, sess=self.sess)
            tl.files.save_npz(self.n_dic_J.all_params,  name=net_dJ_iter_name, sess=self.sess)
        elif self.model == 'ALI':
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
