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

class Net_Feature(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.summary = tf.summary

        # ALI approach
        self.model    = args.method
        self.is_train = args.is_train 
        
        # Network module
        self.encoder  = encoder
        self.decoder  = decoder_condition
        self.classify = classify
        self.discriminator = discriminator_condition
        
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
        self.d_fake_A = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='fake_A')
        self.d_fake_B = tf.placeholder(tf.float32, [args.batch_size, args.output_size, args.output_size, \
                                                    args.img_dim], name='fake_B')
        self.d_id_A = tf.placeholder(tf.int32, shape=[args.batch_size, ], name='id_A')
        self.d_id_B = tf.placeholder(tf.int32, shape=[args.batch_size, ], name='id_B')

        # construct the net module
        # A->(Classify)->f_A
        # B->(Classify)->f_B    
        # A->(Encoder)->(c_A+f_B)->(Decoder)->h_B->(Encoder)->(c_h_B+f_A)->(Decoder)->r_A
        #               (c_A+f_A)->(Decoder)->cyc_A
        # B->(Encoder)->(c_B+f_A)->(Decoder)->h_A->(Encoder)->(c_h_A+f_B)->(Decoder)->r_B
        #               (c_B+f_B)->(Decoder)->cyc_B

        ## Encoder1 and Classifer
        self.n_c_A,   self.d_c_A     = self.encoder(self.d_real_A, is_train=True, reuse=False)
        self.n_c_B,   self.d_c_B     = self.encoder(self.d_real_B, is_train=True, reuse=True)

        self.n_f_A,   self.d_f_A     = self.classify(self.d_real_A, is_train=True, reuse=False)
        self.n_f_B,   self.d_f_B     = self.classify(self.d_real_B, is_train=True, reuse=True)

        ## (Decoder1)
        self.n_h_B,   self.d_h_B     = self.decoder(self.d_f_B, self.d_c_A, is_train=True, reuse=False)
        self.n_h_A,   self.d_h_A     = self.decoder(self.d_f_A, self.d_c_B, is_train=True, reuse=True)

        ## (Decoder1_1)
        self.n_cyc_A, self.d_cyc_A   = self.decoder(self.d_f_A, self.d_c_A, is_train=True, reuse=True)
        self.n_cyc_B, self.d_cyc_B   = self.decoder(self.d_f_B, self.d_c_B, is_train=True, reuse=True)

        ## (Encoder2)
        self.n_c_h_B, self.d_c_h_B   = self.encoder(self.d_h_B, is_train=True, reuse=True)
        self.n_c_h_A, self.d_c_h_A   = self.encoder(self.d_h_A, is_train=True, reuse=True)

        ## (Decoder2)
        self.n_r_B,   self.d_r_B     = self.decoder(self.d_f_B, self.d_c_h_A, is_train=True, reuse=True)
        self.n_r_A,   self.d_r_A     = self.decoder(self.d_f_A, self.d_c_h_B, is_train=True, reuse=True)

        ## Discriminator
        self.n_dis_h_A, self.d_dis_h_A = self.discriminator(self.d_h_A, self.d_f_A, \
                                                            is_train=True, reuse=False)
        self.n_dis_h_B, self.d_dis_h_B = self.discriminator(self.d_h_B, self.d_f_B, \
                                                            is_train=True, reuse=True)
        self.n_dis_real_A, self.d_dis_real_A = self.discriminator(self.d_real_A, self.d_f_A, \
                                                                  is_train=True, reuse=True) 
        self.n_dis_real_B, self.d_dis_real_B = self.discriminator(self.d_real_B, self.d_f_B, \
                                                                  is_train=True, reuse=True)
        self.n_dis_fake_A, self.d_dis_fake_A = self.discriminator(self.d_fake_A, self.d_f_A, \
                                                                  is_train=True, reuse=True)        
        self.n_dis_fake_B, self.d_dis_fake_B = self.discriminator(self.d_fake_B, self.d_f_B, \
                                                                  is_train=True, reuse=True)
       
        # Loss funciton

        ## loss for classification
        self.loss_classA = self.lossCross(self.d_f_A, self.d_id_A)
        self.loss_classB = self.lossCross(self.d_f_B, self.d_id_B)
        
        ## loss for cycle generator updating
        self.loss_codeA2B = self.lossCode(self.d_c_A, self.d_c_h_B)
        self.loss_codeB2A = self.lossCode(self.d_c_B, self.d_c_h_A)
        self.loss_cycle   = args.cycle * (self.lossCYC(self.d_real_A, self.d_r_A) + \
                                          self.lossCYC(self.d_real_B, self.d_r_B)) + \
                            args.in_cycle * (self.lossCYC(self.d_real_A, self.d_cyc_A) + \
                                             self.lossCYC(self.d_real_B, self.d_cyc_B))
            

        self.loss_gen_b = self.lossGAN(self.d_dis_h_B, 1)
        self.loss_gen_a = self.lossGAN(self.d_dis_h_A, 1)
        self.loss_a2b = self.loss_gen_b + self.loss_cycle + self.loss_codeA2B
        self.loss_b2a = self.loss_gen_a + self.loss_cycle + self.loss_codeB2A

        ## loss for discriminator
        #self.loss_da     = (self.lossGAN(self.d_dis_real_A, 1) + self.lossGAN(self.d_dis_fake_A, 0)) / 2.0
        #self.loss_db     = (self.lossGAN(self.d_dis_real_B, 1) + self.lossGAN(self.d_dis_fake_B, 0)) / 2.0
        self.loss_da     = (self.lossGAN(self.d_dis_real_A, 1) + self.lossGAN(self.d_dis_h_A, 0)) / 2.0
        self.loss_db     = (self.lossGAN(self.d_dis_real_B, 1) + self.lossGAN(self.d_dis_h_B, 0)) / 2.0

        # Make summary
        with tf.name_scope('generator-discriminator'):
            self.summ_da  = tf.summary.scalar('dA_loss', self.loss_da)
            self.summ_db  = tf.summary.scalar('dB_loss', self.loss_db)
            self.summ_ga  = tf.summary.scalar('gA_loss', self.loss_gen_a)
            self.summ_gb  = tf.summary.scalar('gB_loss', self.loss_gen_b)

        with tf.name_scope('class'):
            self.summ_loss_realA  = tf.summary.scalar('class_A', self.loss_classA)
            self.summ_loss_realB  = tf.summary.scalar('class_B', self.loss_classB)

        with tf.name_scope('cycle'):        
            self.summ_loss_codeA  = tf.summary.scalar('code_A',  self.loss_codeA2B)
            self.summ_loss_codeB  = tf.summary.scalar('code_B',  self.loss_codeB2A)
            self.summ_a2b = tf.summary.scalar('a2b_loss', self.loss_a2b)
            self.summ_b2a = tf.summary.scalar('b2a_loss', self.loss_b2a)
            self.summ_cyc = tf.summary.scalar('cyc_loss', self.loss_cycle)

        with tf.name_scope('realA'):
            true_image = tf.reshape(self.d_real_A, [-1, args.output_size, args.output_size, 3])
            self.summ_image_real = tf.summary.image('realA', true_image[0:4], 4)

        with tf.name_scope('realB'):
            true_image = tf.reshape(self.d_real_B, [-1, args.output_size, args.output_size, 3])
            self.summ_image_real = tf.summary.image('realB', true_image[0:4], 4)

        with tf.name_scope('cyclA'):
            fake_image = tf.reshape(self.d_cyc_A, [-1, args.output_size, args.output_size, 3])
            self.summ_image_fake = tf.summary.image('cyclA', fake_image[0:4], 4)

        with tf.name_scope('fakeB'):
            fake_image = tf.reshape(self.d_h_B, [-1, args.output_size, args.output_size, 3])
            self.summ_image_fake = tf.summary.image('fakeB', fake_image[0:4], 4)

        self.summ_merge = tf.summary.merge_all()

        # Extract variables
        self.var_encoder  = tl.layers.get_variables_with_name('ENCODER', True, True)
        self.var_decoder  = tl.layers.get_variables_with_name('DECODER', True, True)
        self.var_classify = tl.layers.get_variables_with_name('CLASSIFY',True, True)
        self.var_disc     = tl.layers.get_variables_with_name('DISC_CONDITION',True, True)

        self.var_en_de = self.var_encoder
        self.var_en_de.extend(self.var_decoder)

    def train(self, args):
        
        # Set optimal for classification
        self.classA_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                    .minimize(self.loss_classA, var_list=self.var_classify)
        self.classB_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                    .minimize(self.loss_classB, var_list=self.var_classify)

        # Set optimal for cycle updating
        self.a2b_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_a2b, var_list=self.var_en_de)
        self.b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                 .minimize(self.loss_b2a, var_list=self.var_en_de)

        # Set optimal for discriminator
        self.da_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss_da, var_list=self.var_disc)
        self.db_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss_db, var_list=self.var_disc)

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
        data_bags = [[],[],[]]
        min_len = 20000
        for data_id in range(0, 3):
            read_path = os.path.join("./data/GTAV", 'data'+str(data_id), "*.jpg")
            data_file = glob(read_path)
            #data_file = data_file[10:2010]
            data_code = np.ones(len(data_file)) * data_id
            if len(data_file) < min_len:
                min_len = len(data_file)

            datas = zip(data_file, data_code)
            data_bags[data_id] = datas

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


                ### Update Nets ###
                start_time = time.time()
                
                # Forward G network
                feed_dict={self.d_real_A: batch_A_images, self.d_real_B: batch_A_images}
                fake_A, fake_B = self.sess.run([self.d_h_A, self.d_h_B], feed_dict=feed_dict)
                
                feed_dict={self.d_real_A: batch_A_images, self.d_real_B: batch_B_images,\
                           self.d_id_A: id_A, self.d_id_B: id_B, \
                           self.d_fake_A: fake_A, self.d_fake_B: fake_B}

                # Update G and D in A2B
                err_classA, _ = self.sess.run([self.loss_classA, self.classA_optim], feed_dict=feed_dict)
                #err_codeA, _ = self.sess.run([self.loss_codeA, self.codeA_optim], feed_dict=feed_dict)
                err_db,  _ = self.sess.run([self.loss_db,  self.db_optim],  feed_dict=feed_dict)
                for gen_loop in range(3):
                    err_a2b, err_gen_b, _ = self.sess.run([self.loss_a2b, self.loss_gen_b, self.a2b_optim], \
                                                          feed_dict=feed_dict)

                # Update G and D in B2A
                err_classB, _ = self.sess.run([self.loss_classB, self.classB_optim], feed_dict=feed_dict)
                #err_codeB, _ = self.sess.run([self.loss_codeB, self.codeB_optim], feed_dict=feed_dict)
                err_da,  _ = self.sess.run([self.loss_da,  self.da_optim],  feed_dict=feed_dict)
                for gen_loop in range(3):
                    err_b2a, err_gen_a, _ = self.sess.run([self.loss_b2a, self.loss_gen_b, self.b2a_optim], \
                                                          feed_dict=feed_dict)       

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, gen_a: %4.4f, gen_b: %4.4f, E_da: %4.4f, E_db: %4.4f"  % \
                      (epoch, args.epoch, idx, batch_idxs, time.time() - start_time, \
                       err_gen_a, err_gen_b, err_da, err_db))
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

        test_dir = ["FOGGY1", "FOGGY2", "RAIN1", "RAIN2", "SUNNY1", "SUNNY2"]
        result_dir = os.path.join(args.result_dir, args.method)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for test_epoch in range(1, 21):

            # Initial layer's variables
            self.test_epoch = test_epoch * 100
            self.loadParam(args)
            print("[*] Load network done")


            for dir_index, file_name in enumerate(test_dir):
                
                ## Evaulate test data
                test_files  = glob(os.path.join(args.data_dir, args.dataset, file_name, "*.jpg"))
                test_files.sort()
                test_files = test_files[10:480]
                
                ## Extract Test data code
                start_time = time.time()
                test_code = np.zeros([len(test_files), 512]).astype(np.float32)

                for img_index, file_img in enumerate(test_files):

                    sample = get_image(file_img, args.image_size, is_crop=args.is_crop, \
                                       resize_w=args.output_size, is_grayscale=0)
                    sample_image = np.array(sample).astype(np.float32)
                    sample_image = sample_image.reshape([1,args.output_size,args.output_size,3])
                    print ("Load data {}".format(file_img))
                    feed_dict={self.d_real_A: sample_image}
                    test_code[img_index]  = self.sess.run(self.d_c_A, feed_dict=feed_dict)

    
                print("Test code extraction time: %4.4f"  % (time.time() - start_time))
                Testvector_path = os.path.join(result_dir, str(test_epoch)+'_'+file_name+'_vt.npy')
                np.save(Testvector_path, test_code)



    def makeSample(self, feed_dict, sample_dir, epoch, idx):
        summary, img = self.sess.run([self.summ_merge, self.n_h_A.outputs], feed_dict=feed_dict)

        # update summary
        self.writer.add_summary(summary, self.iter_counter)
        # save image
        img = (np.array(img) + 1) / 2 * 255
        save_images(img, [8, 8],'./{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))


    def loadParam(self, args):
        # load the latest checkpoints
        if args.is_train == True:
            load_de  = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method), \
                                        name='/net_de_%d.npz' % args.c_epoch)
            load_en  = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method), \
                                        name='/net_en_%d.npz' % args.c_epoch)
            load_cls = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method), \
                                        name='/net_cls_%d.npz' % args.c_epoch)
            load_dis = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method), \
                                        name='/net_dis_%d.npz' % args.c_epoch)
            tl.files.assign_params(self.sess, load_en, self.n_c_A)
            tl.files.assign_params(self.sess, load_de, self.n_h_B)
            tl.files.assign_params(self.sess, load_cls, self.n_f_A)
            tl.files.assign_params(self.sess, load_dis, self.n_dis_h_A)
        else:
            load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.method), \
                                        name='/net_de_%d.npz' % self.test_epoch)
            tl.files.assign_params(self.sess, load_en, self.n_c_A)


    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        save_dir = os.path.join(args.checkpoint_dir, args.method)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
   
        print (save_dir)

        # the latest version location
        net_en_name  = os.path.join(save_dir, 'net_en.npz')
        net_de_name  = os.path.join(save_dir, 'net_de.npz')
        net_cls_name = os.path.join(save_dir, 'net_cls.npz')
        net_dis_name = os.path.join(save_dir, 'net_dis.npz')

        # this version is for future re-check and visualization analysis
        net_en_iter_name  = os.path.join(save_dir, 'net_en_%d.npz' % self.iter_counter)
        net_de_iter_name  = os.path.join(save_dir, 'net_de_%d.npz' % self.iter_counter)
        net_cls_iter_name = os.path.join(save_dir, 'net_cls_%d.npz' % self.iter_counter)
        net_dis_iter_name = os.path.join(save_dir, 'net_dis_%d.npz' % self.iter_counter)
        
        tl.files.save_npz(self.n_c_A.all_params,      name=net_en_name, sess=self.sess)
        tl.files.save_npz(self.n_h_B.all_params,      name=net_de_name, sess=self.sess)
        tl.files.save_npz(self.n_f_A.all_params,      name=net_cls_name, sess=self.sess)
        tl.files.save_npz(self.n_dis_h_A.all_params,  name=net_dis_name, sess=self.sess)
        
        tl.files.save_npz(self.n_c_A.all_params,      name=net_en_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_h_B.all_params,      name=net_de_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_f_A.all_params,      name=net_cls_iter_name, sess=self.sess)
        tl.files.save_npz(self.n_dis_h_A.all_params,  name=net_dis_iter_name, sess=self.sess)
