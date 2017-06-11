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

        # ALI approach
        self.model    = args.model_dir
        self.is_train = args.is_train 
        
        # Network module
        self.encoder = encoder
        self.decoder = decoder
        self.discX   = discriminator_X
        self.discZ   = discriminator_Z
        self.discJ   = discriminator_J
        
        # Loss function
        self.lossGAN = mae_criterion
        self.lossCYC = abs_criterion

        # SeqSLAM
        self.vec_D    = Euclidean
        self.getMatch = getMatches

        self._build_model()

    def _build_model(self):
        self.d.real.x = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, \
                                                      self.img_dim], name='real_image')
        self.d.real.z  = tf.placeholder(tf.float32, [self.batch_size, self.code_dim], name="real_code")

        self.n.fake.x, self.d.fake.x = self.decoder(self.d.real.z, is_train=True, reuse=False)
        self.n.cycl.z, self.d.cycl.z = self.encoder(self.d.fake.x, is_train=True, reuse=True )
        self.n.fake.z, self.d.fake.z = self.encoder(self.d.real.x, is_train=True, reuse=False)
        self.n.cycl.x, self.d.cycl.x = self.decoder(self.d.fake.z, is_train=True, reuse=True )

        with tf.name_scope('real'):
            true_image = tf.reshape(real_images, [-1, 64, 64, 3])
            self.summ.image.real = tf.summary.image('real', true_image[0:4], 4)

        with tf.name_scope('fake'):
            fake_image = tf.reshape(cyc_X, [-1, 64, 64, 3])
            self.summ.image.fake = tf.summary.image('fake', fake_image[0:4], 4)

        self.n.dic.x,  self.d.dic.x  = self.discX(self.d.real.x, is_train=True, reuse=False)
        self.n.dic.z,  self.d.dic.z  = self.discZ(self.d.real.z, is_train=True, reuse=False)
        self.n.dic.J,  self.d.dic.J  = self.discJ(self.d.real.J, is_train=True, reuse=False)

        self.n.dic.fx, self.d.dic.fx = self.discX(self.d.fake.x, is_train=True, reuse=True)
        self.n.dic.fz, self.d.dic.fz = self.discY(self.d.fake.y, is_train=True, reuse=True)
        self.n.dic.fJ, self.d.dic.fJ = self.discJ(self.d.fake.J, is_train=True, reuse=True)

        # Apply Loss
        self.loss.encoder = self.param.side_D * self.lossGAN(self.d.dic.fz, 1)
        self.loss.decoder = self.param.side_D * self.lossGAN(self.d.dic.fx, 1)
        self.loss.cycle   = self.param.cycle * (self.criterion_abs(self.d.real.x, self.d.cycl.x) + \
                                                self.criterion_abs(self.d.real.z, self.d.cycl.z))
        self.loss.dicJ    = 0.5 * (self.lossGAN(self.d.dic.J, 1) + self.lossGAN(self.d.dic.fJ, 0))
        self.loss.dicfJ   = 0.5 * (self.lossGAN(self.d.dic.J, 0) + self.lossGAN(self.d.dic.fJ, 1))
        self.loss.dicX    = self.param.side_D*0.5*(self.lossGAN(self.d.dic.x, 1) + \
                                                   self.lossGAN(self.d.dic.fx,0))
        self.loss.dicZ    = self.param.side_D*0.5*(self.lossGAN(self.d.dic.z, 1) + \
                                                   self.lossGAN(self.d.dic.fz,0))
        # Make summary
        self.summ.encoder = tf.summary.scalar('encoder_loss', self.loss.encoder)
        self.summ.decoder = tf.summary.scalar('decoder_loss', self.loss.decoder)
        self.summ.cycle   = tf.summary.scalar('clc_loss',     self.loss.cycle)
        self.summ.dicJ    = tf.summary.scalar('d_J_loss',     self.loss.dicJ)
        self.summ.dicfJ   = tf.summary.scalar('d_fJ_loss',    self.loss.dicfJ)
        self.summ.dicX    = tf.summary.scalar('d_X_loss',     self.loss.dicX)
        self.summ.dicZ    = tf.summary.scalar('d_Z_loss',     self.loss.dicZ)
        self.summ.merge   = tf.summary.merge_all()

        # Extract variables
        self.var.encoder  = tl.layers.get_variables_with_name('ENCODER', True, True)
        self.var.decoder  = tl.layers.get_variables_with_name('DECODER', True, True)
        self.var.dicX     = tl.layers.get_variables_with_name('DISC_X',  True, True)
        self.var.dicZ     = tl.layers.get_variables_with_name('DISC_Z',  True, True)
        self.var.dicJ     = tl.layers.get_variables_with_name('DISC_J',  True, True)
        self.var.gen      = self.var.encoder
        self.var.gen.extend(self.var.decoder)

    def train(self, args):
        
        # Set optimal for nets
        self.optim.dicJ    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                     .minimize(self.loss.dicJ,    var_list=self.var.dicJ)
        self.optim.dicfJ   = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                     .minimize(self.loss.dicfJ,   var_list=self.var.dicfJ)
        if self.model == 'ALI_CYC':
            self.optim.encoder = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss.encoder, var_list=self.var.encoder)
            self.optim.decoder = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss.decoder, var_list=self.var.decoder)
            self.optim.cycle   = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss.cycle,   var_list=self.var.cycle)
            self.optim.dicX    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss.dicX,    var_list=self.var.dicX)
            self.optim.dicZ    = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                                .minimize(self.loss.dicZ,    var_list=self.var.dicZ)

        # Initial global variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        # Initial layer's variables
        tl.layers.initialize_global_variables(self.sess)
        if args.restore == True:
            self.loadParam(args)
            print("[*] Load network done")
        else:
            print("[!] Initial network done")
        
        # Load Data files
        data_files = glob(os.path.join("./data", FLAGS.dataset, "train/*.jpg"))

        # Main loop for Training
        self.iter_counter = 0
        for epoch in range(args.c_epoch+1, args.epoch):
            ## shuffle data
            shuffle(data_files)
            print("[*] Dataset shuffled!")
            
            ## update sample files based on shuffled data
            sample_files = data_files[0:FLAGS.sample_size]
            sample = [get_image(sample_file, FLAGS.image_size, dataset, \
                                is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) \
                      for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            print("[*] Sample images updated!")
            
            ## load image data
            batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
            
            for idx in xrange(0, batch_idxs):
                ### Get datas ###
                batch_files  = data_files[idx*args.batch_size:(idx+1)*args.batch_size]
                ## get real images
                batch        = [get_image(batch_file, args.image_size, dataset, \
                                          is_crop=args.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) \
                                for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                ## get real code
                batch_codes  = np.random.normal(loc=0.0, scale=1.0, \
                                                size=(args.sample_size, z_dim)).astype(np.float32)
                
                ### Update Nets ###
                start_time = time.time()
                feed_dict={real_code: batch_z, real_image: batch_images}
                feed_dict.update(self.n.dic.J.all_drop)
                feed_dict.update(self.n.dic.fJ.all_drop)
                if self.model == 'ALI_CYC':
                    feed_dict.update(self.n.dic.Z.all_drop)
                    feed_dict.update(self.n.dic.fZ.all_drop)
                    errX, _ = self.sess.run([self.loss.dic.x,   self.optim.dicX],    feed_dict=feed_dict)
                    errZ, _ = self.sess.run([self.loss.dic.z,   self.optim.dicZ],    feed_dict=feed_dict)
                    errJ, _ = self.sess.run([self.loss.dic.J,   self.optim.dicJ],    feed_dict=feed_dict)
                    errE, _ = self.sess.run([self.loss.encoder, self.optim.encoder], feed_dict=feed_dict)
                    errD, _ = self.sess.run([self.loss.decoder, self.optim.encoder], feed_dict=feed_dict)
                    ## updates the Joint Generator multi times to avoid Discriminator converge early
                    for _ in range(4):
                        errfJ, _  = self.sess.run([self.loss.dicfJ, self.optim.dicfJ], feed_dict=feed_dict)
                        ## update inverse mapping
                        errClc, _ = self.sess.run([self.loss.cycle, self.optim.cycle], feed_dict=feed_dict)
                        print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f"  % (epoch, args.epoch, idx, batch_idxs, time.time() - start_time))
                elif self.model == 'ALI':
                    errJ, _ = self.sess.run([self.loss.dic.J,   self.optim.dicJ],    feed_dict=feed_dict)
                    ## updates the Joint Generator multi times to avoid Discriminator converge early
                    for _ in range(4):
                        errfJ, _  = self.sess.run([self.loss.dicfJ, self.optim.dicfJ], feed_dict=feed_dict)

                sys.stdout.flush()
                self.iter_counter += 1

                if np.mod(iter_counter, args.sample_step) == 0:
                    self.makeSample(feed_dict, args.sample_dir, epoch, idx)
                    print("[*] Make new sample SUCCESS!")                    
                if np.mod(iter_counter, args.save_step) == 0:
                    self.saveParam(args)
                    print("[*] Saving checkpoints SUCCESS!")

        # Shutdown writer
        self.writer.close()

    def test(self, args):
        # Initial layer's variables
        self.loadParam(args)
        print("[*] Load network done")

        ## Evaulate data
        train_files = glob(os.path.join(args.data_dir, args.dataset, "train/*.jpg"))
        test_files  = glob(os.path.join(args.data_dir, args.dataset, args.test_dir,"*.jpg"))
        train_files.sort()
        test_files.sort()
        
        ## Extract Train data code
        train_code  = np.zeros([args.sample_len, 512]).astype(np.float32)
        for id in range(train_code.shape[0]):
            sample_file = train_files[id]
            sample = get_image(sample_file, FLAGS.image_size, dataset, \
                               is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)
            sample_image = np.array(sample).astype(np.float32)
            sample_image = sample_image.reshape([1,64,64,3])
            print ("Load data {}".format(sample_file))
            feed_dict={real_image: sample_image}
            train_code[id]  = self.sess.run(self.d.fake.z, feed_dict=feed_dict)
        print ("Train code extraction done!")

        ## Extract Test data code
        test_code = np.zeros([sample_len, 512]).astype(np.float32)
        for id in range(test_code.shape[0]):
            sample_file = test_files[id]
            sample = get_image(sample_file, FLAGS.image_size, dataset, \
                               is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)
            sample_image = np.array(sample).astype(np.float32)
            sample_image = sample_image.reshape([1,64,64,3])
            print ("Load data {}".format(sample_file))
            feed_dict={real_image: sample_image}
            test_code[id]  = self.sess.run(self.d.fake.z, feed_dict=feed_dict)
        print ("Test code extraction done!")
        
        ## Measure vector corrcoeffience
        D     = self.vec_D(train_code, test_code)
        match = self.getMatch(DD)
        result_dir = os.path.join(args.result_dir, args.model_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)  
        scipy.misc.imsave(os.path.join(result_dir, args.test_dir+'matrix.jpg'), D * 255)
        
        ## Save matching 
        m = match[:,0]
        thresh = 3
        m[match[:,1] > thresh] = np.nan
        plt.plot(m,'.') 
        plt.title('Matching '+ test_dir)
        plt.savefig(os.path.join(result_dir, args.test_dir+'match.jpg'))

    def makeSample(self, feed_dict, sample_dir, epoch, idx):
        summary, img = sess.run([merged, n_fake_X.outputs], feed_dict=feed_dict)

        # update summary
        self.writer.add_summary(summary, self.iter_counter)
        # save image
        img = (np.array(img) + 1) / 2 * 255
        save_images(img, [8, 8],'./{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))

    def loadParam(self, args):
        # load the latest checkpoints
        if self.method == 'ALI':
            load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                        name='/net_de_%d00.npz' % args.c_epoch)
            load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                        name='/net_en_%d00.npz' % args.c_epoch)
            load_dX = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                        name='/net_dX_%d00.npz' % args.c_epoch)
            load_dZ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                        name='/net_dZ_%d00.npz' % args.c_epoch)
            load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                        name='/net_dJ_%d00.npz' % args.c_epoch)
            tl.files.assign_params(self.sess, load_en, self.n.fake.z)
            tl.files.assign_params(self.sess, load_de, self.n.fake.x)
            tl.files.assign_params(self.sess, load_dX, self.n.dic.x)
            tl.files.assign_params(self.sess, load_dZ, self.n.dic.z)
            tl.files.assign_params(self.sess, load_dJ, self.n.dic.J)
        elif self.method == 'ALI_CYC':
             load_de = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                         name='/net_de_%d00.npz' % args.c_epoch)
             load_en = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                         name='/net_en_%d00.npz' % args.c_epoch)
             load_dJ = tl.files.load_npz(path=os.path.join(args.checkpoint_dir, args.model_dir), \
                                         name='/net_dJ_%d00.npz' % args.c_epoch)
             tl.files.assign_params(self.sess, load_en, self.n.fake.z)
             tl.files.assign_params(self.sess, load_de, self.n.fake.x)
             tl.files.assign_params(self.sess, load_dJ, self.n.dic.J)

    def saveParam(self, args):
        print("[*] Saving checkpoints...")
        save_dir = os.path.join(args.checkpoint_dir, args.model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        

        if self.method == 'ALI':
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
            
            tl.files.save_npz(self.n.fake.x.all_params, name=net_de_name, sess=self.sess)
            tl.files.save_npz(self.n.fake.z.all_params, name=net_en_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.x.all_params,  name=net_dX_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.z.all_params,  name=net_dZ_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.J.all_params,  name=net_dJ_name, sess=self.sess)

            tl.files.save_npz(self.n.fake.x.all_params, name=net_de_iter_name, sess=self.sess)
            tl.files.save_npz(self.n.fake.z.all_params, name=net_en_iter_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.x.all_params,  name=net_dX_iter_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.z.all_params,  name=net_dZ_iter_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.J.all_params,  name=net_dJ_iter_name, sess=self.sess)
        elif self.method == 'ALI':
            # the latest version location
            net_de_name = os.path.join(save_dir, 'net_de.npz')
            net_en_name = os.path.join(save_dir, 'net_en.npz')
            net_dJ_name = os.path.join(save_dir, 'net_dJ.npz')
            # this version is for future re-check and visualization analysis
            net_de_iter_name = os.path.join(save_dir, 'net_de_%d.npz' % self.iter_counter)
            net_en_iter_name = os.path.join(save_dir, 'net_en_%d.npz' % self.iter_counter)
            net_dJ_iter_name = os.path.join(save_dir, 'net_dJ_%d.npz' % self.iter_counter)
            
            tl.files.save_npz(self.n.fake.x.all_params, name=net_de_name, sess=self.sess)
            tl.files.save_npz(self.n.fake.z.all_params, name=net_en_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.J.all_params,  name=net_dJ_name, sess=self.sess)

            tl.files.save_npz(self.n.fake.x.all_params, name=net_de_iter_name, sess=self.sess)
            tl.files.save_npz(self.n.fake.z.all_params, name=net_en_iter_name, sess=self.sess)
            tl.files.save_npz(self.n.dic.J.all_params,  name=net_dJ_iter_name, sess=self.sess)
