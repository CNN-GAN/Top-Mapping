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

class Net_simpleCYC(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        # ALI approach
        self.model    = args.method
        self.is_train = args.is_train 
        
        # Network module
        if args.log_name == 'r1_C_0.1':
            self.encoder  = encoder
        else:
            self.encoder  = encoder_condition

        self.decoder  = decoder_condition
        self.classify = classify
        self.discriminatorX = discriminator_condition
        self.discriminatorZ = discriminator_Z
        
        # Loss function
        self.lossGAN = abs_criterion
        self.lossCYC = abs_criterion
        self.lossCross = cross_loss
        self.lossCode  = abs_criterion

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

        # placeholder
        self.d_real_A = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_A')
        self.d_real_B = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='real_B')
        self.d_id_A = tf.placeholder(tf.int32, shape=[args.batch_size, ], name='id_A')
        self.d_id_B = tf.placeholder(tf.int32, shape=[args.batch_size, ], name='id_B')

        self.rand_code = tf.placeholder(tf.float32, [args.batch_size, args.code_dim], name='rand_code')
        
        ## Classifier
        self.n_f_A,   self.d_f_A     = self.classify(self.d_real_A, is_train=True, reuse=False)
        self.n_f_B,   self.d_f_B     = self.classify(self.d_real_B, is_train=True, reuse=True)

        ## Encoder
        if args.log_name == 'r1_C_0.1':
            self.n_c_A,   self.d_c_A     = self.encoder(self.d_real_A, is_train=True, reuse=False)
        else:
            self.n_c_A,   self.d_c_A     = self.encoder(self.d_f_A, self.d_real_A, is_train=True, reuse=False)

        ## Decoder
        self.n_fake_A, self.d_fake_A = self.decoder(self.d_f_A, self.d_c_A, is_train=True, reuse=False)
        self.n_fake_B, self.d_fake_B = self.decoder(self.d_f_B, self.d_c_A, is_train=True, reuse=True)
        self.n_rand_A, self.d_rand_A = self.decoder(self.d_f_A, self.rand_code, is_train=True, reuse=True)
        self.n_rand_B, self.d_rand_B = self.decoder(self.d_f_B, self.rand_code, is_train=True, reuse=True)

        ## Discriminator
        self.n_dis_real_A, self.d_dis_real_A = self.discriminatorX(self.d_f_A, self.d_real_A, \
                                                                  is_train=True, reuse=False) 
        self.n_dis_real_B, self.d_dis_real_B = self.discriminatorX(self.d_f_B, self.d_real_B, \
                                                                  is_train=True, reuse=True)
        self.n_dis_fake_A, self.d_dis_fake_A = self.discriminatorX(self.d_f_A, self.d_fake_A, \
                                                                  is_train=True, reuse=True)
        self.n_dis_fake_B, self.d_dis_fake_B = self.discriminatorX(self.d_f_B, self.d_fake_B, \
                                                                  is_train=True, reuse=True)
        self.n_dis_rand_A, self.d_dis_rand_A = self.discriminatorX(self.d_f_A, self.d_rand_A, \
                                                                  is_train=True, reuse=True)        
        self.n_dis_rand_B, self.d_dis_rand_B = self.discriminatorX(self.d_f_B, self.d_rand_B, \
                                                                  is_train=True, reuse=True)

        self.n_dis_real_code, self.d_dis_real_code = self.discriminatorZ(self.d_c_A, \
                                                                          is_train=True, reuse=False)
        self.n_dis_fake_code, self.d_dis_fake_code = self.discriminatorZ(self.rand_code, \
                                                                          is_train=True, reuse=True)
        
        # Loss funciton

        ## loss for classification
        self.loss_classA = self.lossCross(self.d_f_A, self.d_id_A)
        self.loss_classB = self.lossCross(self.d_f_B, self.d_id_B)
        self.loss_class = self.loss_classA + self.loss_classB
        
        ## loss for cycle updating
        self.loss_cyc   = args.cycle * self.lossCYC(self.d_real_A, self.d_fake_A)
        
        ## loss for generator          
        self.loss_gen_A = self.lossGAN(self.d_dis_fake_A, 1) + self.lossGAN(self.d_dis_rand_A, 1)
        self.loss_gen_B = self.lossGAN(self.d_dis_fake_B, 1) + self.lossGAN(self.d_dis_rand_B, 1)
        self.loss_gen_code = self.lossGAN(self.d_dis_fake_code, 1)
        self.loss_gen = self.loss_gen_A + self.loss_gen_B + self.loss_gen_code

        ## loss for discriminator
        self.loss_dis_A = (self.lossGAN(self.d_dis_real_A, 1) + \
                           self.lossGAN(self.d_dis_fake_A, 0) + \
                           self.lossGAN(self.d_dis_rand_A, 0)) / 3.0
        self.loss_dis_B = (self.lossGAN(self.d_dis_real_B, 1) + \
                           self.lossGAN(self.d_dis_fake_B, 0) + \
                           self.lossGAN(self.d_dis_rand_B, 0)) / 3.0
        self.loss_dis_X = self.loss_dis_A + self.loss_dis_B

        self.loss_dis_code = self.lossGAN(self.d_dis_real_code, 1) + \
                             self.lossGAN(self.d_dis_fake_code, 0)

        # Make summary
        with tf.name_scope('class'):
            self.summ_loss_realA  = tf.summary.scalar('class_A', self.loss_classA)
            self.summ_loss_realB  = tf.summary.scalar('class_B', self.loss_classB)

        with tf.name_scope('cycle'):        
            self.summ_cyc = tf.summary.scalar('cyc_loss', self.loss_cyc)

        with tf.name_scope('generator-discriminator'):
            self.summ_da     = tf.summary.scalar('dA_loss', self.loss_dis_A)
            self.summ_db     = tf.summary.scalar('dB_loss', self.loss_dis_B)
            self.summ_dcode  = tf.summary.scalar('dcode_loss', self.loss_dis_code)
            self.summ_ga     = tf.summary.scalar('genA_loss', self.loss_gen_A)
            self.summ_gb     = tf.summary.scalar('genB_loss', self.loss_gen_B)
            self.summ_gcode  = tf.summary.scalar('gen_code_loss', self.loss_gen_code)

        with tf.name_scope('realA'):
            true_image = tf.reshape(self.d_real_A, [-1, args.output_size, args.output_size, 3])
            self.summ_image_real = tf.summary.image('realA', true_image[0:4], 4)

        with tf.name_scope('realB'):
            true_image = tf.reshape(self.d_real_B, [-1, args.output_size, args.output_size, 3])
            self.summ_image_real = tf.summary.image('realB', true_image[0:4], 4)

        with tf.name_scope('fakeA'):
            fake_image = tf.reshape(self.d_fake_A, [-1, args.output_size, args.output_size, 3])
            self.summ_image_fake = tf.summary.image('fakeA', fake_image[0:4], 4)

        with tf.name_scope('fakeB'):
            fake_image = tf.reshape(self.d_fake_B, [-1, args.output_size, args.output_size, 3])
            self.summ_image_fake = tf.summary.image('fakeB', fake_image[0:4], 4)

        self.summ_merge = tf.summary.merge_all()

        # Extract variables
        self.var_encoder  = tl.layers.get_variables_with_name('ENCODER', True, True)
        self.var_decoder  = tl.layers.get_variables_with_name('DECODER', True, True)
        self.var_classify = tl.layers.get_variables_with_name('CLASSIFY',True, True)
        self.var_disX     = tl.layers.get_variables_with_name('DISC_X',True, True)
        self.var_disZ     = tl.layers.get_variables_with_name('DISC_Z',True, True)

        self.var_en_de = self.var_encoder
        self.var_en_de.extend(self.var_decoder)

    def train(self, args):
        
        # Set optimal for classification
        self.class_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                    .minimize(self.loss_class, var_list=self.var_classify)

        # Set optimal for cycle updating
        self.cyc_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_cyc, var_list=self.var_en_de)

        # Set optimal for discriminator
        self.gen_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_gen, var_list=self.var_en_de)
        self.disX_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_dis_X, var_list=self.var_disX)
        self.disZ_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_dis_code, var_list=self.var_disZ)


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
        data_bags = [[],[],[]]
        min_len = 20000

        route_names = ['Route1', 'Route2']
        data_names = ['FOGGY1', 'FOGGY2', 'SUNNY1', 'SUNNY2', 'RAIN1', 'RAIN2']

        for data_id, data_name in enumerate(data_names):
            data_file = []
            for route_id, route_name in enumerate(route_names):
                read_path = os.path.join("./data/GTAV", route_name, data_name, "*.jpg")
                tmp_file = glob(read_path)
                data_file += tmp_file

            data_code = np.ones(len(data_file)) * np.int(data_id/2)
            datas = zip(data_file, data_code)
            data_bags[np.int(data_id/2)] += datas

            #print ('shape is {}'.format(np.array(data_bags[np.int(data_id/2)])))
            if data_id%2==1:
                if len(data_bags[np.int(data_id/2)]) < min_len:
                    min_len = len(data_bags[np.int(data_id/2)])

        '''
        for data_id in range(0, 3):
            read_path = os.path.join("./data/GTAV", 'data'+str(data_id), "*.jpg")
            data_file = glob(read_path)
            #data_file = data_file[10:2010]
            data_code = np.ones(len(data_file)) * data_id
            if len(data_file) < min_len:
                min_len = len(data_file)

            datas = zip(data_file, data_code)
            data_bags[data_id] = datas
        '''

        print ("Minimum length is {}".format(min_len))

        # Main loop for Training
        self.iter_counter = 0
        begin_epoch = 0
        if args.restore == True:
            begin_epoch = args.c_epoch+1

        for epoch in range(begin_epoch, args.epoch):
            ## shuffle data
            for data_id in range(0, 3):
                shuffle(data_bags[data_id])

            print("[*] Dataset shuffled!")
            
            ## load image data
            batch_idxs = min(min_len, args.train_size) // args.batch_size
            
            for idx in xrange(0, batch_idxs):
                ### Get datas ###
                data_ids     = np.random.randint(3, size=args.batch_size)
                data_add_ids = np.random.randint(2, size=args.batch_size)
                data_A = []
                data_B = []
                id_A = []
                id_B = []
                start_id = idx*args.batch_size
                for index, valA in enumerate(data_ids):
                    
                    data_A += [data_bags[valA][index+start_id][0]]
                    id_A += [data_bags[valA][index+start_id][1]]

                    valB = (valA+data_add_ids[index]+1)%3            
                    data_B += [data_bags[valB][index+start_id][0]]
                    id_B += [data_bags[valB][index+start_id][1]]                  


                id_A = np.array(id_A).astype(int32)
                id_B = np.array(id_B).astype(int32)


                ## get real images
                batch_A       = [get_image(batch_file, args.image_size, is_crop=args.is_crop, \
                                           resize_w=args.output_size, is_grayscale = 0) \
                                 for batch_file in data_A]
                batch_A_images = np.array(batch_A).astype(np.float32)
                batch_B       = [get_image(batch_file, args.image_size, is_crop=args.is_crop, \
                                           resize_w=args.output_size, is_grayscale = 0) \
                                 for batch_file in data_B]
                batch_B_images = np.array(batch_B).astype(np.float32)
                batch_codes  = np.random.normal(loc=0.0, scale=1.0, \
                                                size=(args.sample_size, args.code_dim)).astype(np.float32)

                ### Update Nets ###
                start_time = time.time()
                feed_dict={self.d_real_A: batch_A_images, self.d_real_B: batch_B_images,\
                           self.d_id_A: id_A, self.d_id_B: id_B, self.rand_code: batch_codes}

                # Update G and D in A2B
                ## classification
                err_class, _ = self.sess.run([self.loss_class, self.class_optim], feed_dict=feed_dict)

                ## Discriminator
                err_disX,  _ = self.sess.run([self.loss_dis_X,    self.disX_optim],  feed_dict=feed_dict)
                err_disZ,  _ = self.sess.run([self.loss_dis_code, self.disZ_optim],  feed_dict=feed_dict)
                
                ## Generator
                for gen_loop in range(8):
                    err_gen, _ = self.sess.run([self.loss_gen, self.gen_optim], feed_dict=feed_dict)

                ## Cycle updating    
                err_cyc, _ = self.sess.run([self.loss_cyc, self.cyc_optim], feed_dict=feed_dict)

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, class: %4.4f, gen: %4.4f, cyc: %4.4f, disX: %4.4f, disZ: %4.4f"  % \
                      (epoch, args.epoch, idx, batch_idxs, time.time() - start_time, \
                       err_class, err_gen, err_cyc, err_disX, err_disZ))
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

        route_dir = ["Route1", "Route2", "Route3"]
        test_dir = ["FOGGY", "RAIN", "SUNNY"]
        result_dir = os.path.join(args.result_dir, args.method, args.log_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for test_epoch in range(1, 27):

            # Initial layer's variables
            self.test_epoch = test_epoch * 50
            self.loadParam(args)
            print("[*] Load network done")

            for route_index, route_name in enumerate(route_dir):

                for dir_index, file_name in enumerate(test_dir):
                
                    ## Evaulate test data
                    test_files  = glob(os.path.join(args.data_dir, args.dataset, route_name, file_name, "*.jpg"))
                    test_files.sort()
                    if route_index != 2:
                        test_files = test_files[0:args.test_len]
                    else:
                        test_files = test_files[400:args.test_len+400]
                    
                    ## Extract Test data code
                    start_time = time.time()
                    test_code = np.zeros([len(test_files), 512]).astype(np.float32)

                    for img_index, file_img in enumerate(test_files):

                        sample = get_image(file_img, args.image_size, is_crop=args.is_crop, \
                                           resize_w=args.output_size, is_grayscale=0)
                        sample_image = np.array(sample).astype(np.float32)
                        sample_image = sample_image.reshape([1,args.output_size,args.output_size,3])
                        feed_dict={self.d_real_A: sample_image}
                        test_code[img_index]  = self.sess.run(self.d_c_A, feed_dict=feed_dict)

    
                    print("Test code extraction time: %4.4f"  % (time.time() - start_time))
                    if route_index==2:
                        route_name = "Route3"

                    Testvector_path = os.path.join(result_dir, str(test_epoch)+'_'+route_name+'_'+file_name+'_vt.npy')
                    print ("save path {}".format(Testvector_path))
                    np.save(Testvector_path, test_code)



    def makeSample(self, feed_dict, sample_dir, epoch, idx):
        summary, img = self.sess.run([self.summ_merge, self.n_fake_A.outputs], feed_dict=feed_dict)

        # update summary
        self.writer.add_summary(summary, self.iter_counter)
        # save image
        img = (np.array(img) + 1) / 2 * 255
        save_images(img, [8, 8],'./{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))


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
            load_cls = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                         name='/net_cls_%d.npz' % self.test_epoch)
            load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method, args.log_name), \
                                        name='/net_en_%d.npz' % self.test_epoch)
            tl.files.assign_params(self.sess, load_cls, self.n_f_A)
            tl.files.assign_params(self.sess, load_en, self.n_c_A)


    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        save_dir = os.path.join(args.checkpoint_dir, args.method, args.log_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print (save_dir)

        # the latest version location
        net_en_name  = os.path.join(save_dir, 'net_en.npz')
        net_de_name  = os.path.join(save_dir, 'net_de.npz')
        net_cls_name = os.path.join(save_dir, 'net_cls.npz')
        net_disX_name = os.path.join(save_dir, 'net_disX.npz')
        net_disZ_name = os.path.join(save_dir, 'net_disZ.npz')

        # this version is for future re-check and visualization analysis
        net_en_iter_name  = os.path.join(save_dir, 'net_en_%d.npz' % self.iter_counter)
        net_de_iter_name  = os.path.join(save_dir, 'net_de_%d.npz' % self.iter_counter)
        net_cls_iter_name = os.path.join(save_dir, 'net_cls_%d.npz' % self.iter_counter)
        net_disX_iter_name = os.path.join(save_dir, 'net_disX_%d.npz' % self.iter_counter)
        net_disZ_iter_name = os.path.join(save_dir, 'net_disZ_%d.npz' % self.iter_counter)
        
        tl.files.save_npz(self.n_c_A.all_params,      name=net_en_name, sess=self.sess)
        tl.files.save_npz(self.n_fake_A.all_params,      name=net_de_name, sess=self.sess)
        tl.files.save_npz(self.n_f_A.all_params,      name=net_cls_name, sess=self.sess)
        tl.files.save_npz(self.n_dis_real_A.all_params,  name=net_disX_name, sess=self.sess)
        tl.files.save_npz(self.n_dis_real_code.all_params,  name=net_disZ_name, sess=self.sess)
        
        tl.files.save_npz(self.n_c_A.all_params,      name=net_en_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_fake_A.all_params,      name=net_de_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_f_A.all_params,      name=net_cls_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_dis_real_A.all_params,  name=net_disX_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_dis_real_code.all_params,  name=net_disZ_iter_name, sess=self.sess)
