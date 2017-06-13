import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model import *
from utils import *

pp = pprint.PrettyPrinter()

# for 3d Drawing
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get dault parameters
FLAGS = defaultParam()

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    if FLAGS.is_train == False:
        FLAGS.batch_size = 1

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.sample_dir):
            os.makedirs(FLAGS.sample_dir)

    z_dim = 512

    ##========================= DEFINE MODEL ===========================##
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')
    
    # Generator: Z --> fake_X
    n_fake_X, fake_X = decoder(z, is_train=True, reuse=False)

    # Encoder:   X --> fake_Z
    n_fake_Z, fake_Z = encoder(real_images, is_train=True, reuse=False)
    
    # X->G(E(X))
    n_cyc_X,  cyc_X  = decoder(n_fake_Z.outputs, is_train=True, reuse=True)

    # Z->E(G(Z))
    n_cyc_Z,  cyc_Z  = encoder(n_fake_X.outputs, is_train=True, reuse=True)


    with tf.name_scope('real'):
        true_image = tf.reshape(real_images, [-1, 64, 64, 3])
        tf.summary.image('real', true_image[0:4], 4)

    with tf.name_scope('fake'):
        fake_image = tf.reshape(cyc_X, [-1, 64, 64, 3])
        tf.summary.image('fake', fake_image[0:4], 4)

    # Discriminator for X
    n_dic_X, dic_X = discriminator_X(real_images, is_train=True, reuse=False)

    # Discriminator for fake_X
    n_dic_fX, dic_fX = discriminator_X(fake_X, is_train=True, reuse=True)
    
    # Discriminator for Z
    n_dic_Z, dic_Z = discriminator_Z(z, is_train=True, reuse=False)

    # Discriminator for fake_Z
    n_dic_fZ, dic_fZ = discriminator_Z(fake_Z, is_train=True, reuse=True)

    # Discriminator for (X, hat_z)
    n_dic_J, dic_J = discriminator_J(real_images, fake_Z, is_train=True, reuse=False)

    # Discriminator for (hat_X, z)
    n_dic_fJ, dic_fJ = discriminator_J(fake_X, z, is_train=True, reuse=True)

    #with tf.name_scope('fake_image'):
    #    fake_image = tf.reshape(net_g.outputs, [-1, 64, 64, 3])
    #    tf.summary.image('fake', fake_image, 64)
        
    ##========================= DEFINE TRAIN OPS =======================##
    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1

    # Apply Different Loss
    with tf.name_scope('discriminator'):
        
        # generator for encoder x -> fake_Z
        #en_loss = tf.reduce_mean((fake_Z - 1)**2) + tf.reduce_mean((dic_J)**2) \
        #        + FLAGS.lamda * tf.reduce_mean(tf.abs(real_images - cyc_X)) \
        #        + FLAGS.lamda * tf.reduce_mean(tf.abs(z - cyc_Z))
        en_loss = FLAGS.side_dic * tf.reduce_mean((dic_fZ - 1)**2)
        tf.summary.scalar('en_loss', en_loss)

        # generator for decoder z -> fake_X
        #de_loss = tf.reduce_mean((fake_X - 1)**2) + tf.reduce_mean((dic_fJ - 1)**2) \
        #        + FLAGS.lamda * tf.reduce_mean(tf.abs(real_images - cyc_X)) \
        #        + FLAGS.lamda * tf.reduce_mean(tf.abs(z - cyc_Z))
        de_loss = FLAGS.side_dic * tf.reduce_mean((dic_fX - 1)**2)
        tf.summary.scalar('de_loss', de_loss)

        clc_loss = FLAGS.lamda * tf.reduce_mean(tf.abs(real_images - cyc_X)) \
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
    
    # optimizers for updating discriminator and generator
    en_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                       .minimize(en_loss, var_list=en_vars)
    de_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                       .minimize(de_loss, var_list=de_vars)
    clc_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                       .minimize(clc_loss, var_list=gen_vars)
    dic_X_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(dic_X_loss, var_list=X_vars)    
    dic_Z_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(dic_Z_loss, var_list=Z_vars)    
    dic_J_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(dic_J_loss, var_list=J_vars)
    dic_fJ_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(dic_fJ_loss, var_list=gen_vars)

    # Limit GPU usage
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    tl.layers.initialize_global_variables(sess)

    model_dir = "loam_ali_clc"
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    tl.files.exists_or_mkdir(save_dir)

    data_files = glob(os.path.join("./data", FLAGS.dataset, "train/*.jpg"))
    # sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    # dataset: 0 for loam, 1 for mnist
    dataset = 0


    if FLAGS.is_train == False:
        ##========================= TEST  MODELS ================================##

        ## load parameters
        load_en = tl.files.load_npz(path=os.path.join("./checkpoint", "loam_ali_clc"), name="/net_en_2200.npz")
        tl.files.assign_params(sess, load_en, n_fake_Z)
        print ("[*] Load NPZ successfully!")

        ## evaulate data
        sample_len = 1000
        test_dir = "test_T15_R1.5"
        train_files = glob(os.path.join("./data", FLAGS.dataset, "train/*.jpg"))
        test_files = glob(os.path.join("./data", FLAGS.dataset, test_dir,"*.jpg"))
        train_files.sort()
        test_files.sort()
        
        ## Extract Train data code
        train_code = np.zeros([sample_len, 512]).astype(np.float32)
        for id in range(train_code.shape[0]):
            sample_file = train_files[id]
            sample = get_image(sample_file, FLAGS.image_size, dataset, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)
            sample_image = np.array(sample).astype(np.float32)
            sample_image = sample_image.reshape([1,64,64,3])
            print ("Load data {}".format(sample_file))
            feed_dict={real_images: sample_image}
            train_code[id]  = sess.run(fake_Z, feed_dict=feed_dict)

        print ("Train code extraction done!")

        ## Extract Test data code
        test_code = np.zeros([sample_len, 512]).astype(np.float32)
        for id in range(test_code.shape[0]):
            sample_file = test_files[id]
            sample = get_image(sample_file, FLAGS.image_size, dataset, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale=0)
            sample_image = np.array(sample).astype(np.float32)
            sample_image = sample_image.reshape([1,64,64,3])
            print ("Load data {}".format(sample_file))
            feed_dict={real_images: sample_image}
            test_code[id]  = sess.run(fake_Z, feed_dict=feed_dict)

        print ("Test code extraction done!")

        #np.set_printoptions(threshold='nan') 
        
        ## Measure the Euclidean Distance
        D_Euclid = np.zeros([train_code.shape[0], test_code.shape[0]])
        for x in range(train_code.shape[0]):
            for y in range(test_code.shape[0]):
                D_Euclid[x,y] = np.linalg.norm(train_code[x]-test_code[y])

        '''
        ## Measure the Manhattan Distance
        D_Manha = np.zeros([H_code.shape[0], H_code.shape[0]])
        for x in range(H_code.shape[0]):
            for y in range(H_code.shape[0]):
                D_Manha[x,y] = np.sum(np.abs(H_code[x]-H_code[y]))
        print ("The vector Manhattan is ")
        print D_Manha

        ## Measure the Chebyshev Distance
        D_Cheby = np.zeros([H_code.shape[0], H_code.shape[0]])
        for x in range(H_code.shape[0]):
            for y in range(H_code.shape[0]):
                D_Cheby[x,y] = np.max(np.abs(H_code[x]-H_code[y]))
        print ("The vector Cheby is ")
        print D_Cheby

        '''
        '''
        ## Measure the Cosine Difference
        D_Cosin = np.zeros([train_code.shape[0], test_code.shape[0]])
        for x in range(train_code.shape[0]):
            for y in range(test_code.shape[0]):
                D_Cosin[x,y] = np.sum(train_code[x]*test_code[y])/(np.linalg.norm(train_code[x])*np.linalg.norm(test_code[y]))
        '''
        ## Measure vector corrcoeffience
        #D_coeff = np.corrcoef([H_code[id] for id in range(H_code.shape[0])])
        #DD = enhanceContrast(D_Euclid)
        DD = D_Euclid
        scipy.misc.imsave(test_dir+'matrix.jpg', DD * 255)
        match = getMatches(DD)
        print ("Extract matching done")
        
        ## show matching 
        print (match)
        m = match[:,0]
        thresh = 10
        m[match[:,1] > thresh] = np.nan
        plt.plot(m,'.') 
        plt.title('Matching '+ test_dir)
        plt.show()
        plt.savefig(test_dir+".jpg")
        print ("Save matching done")
    else:
        ##========================= TRAIN MODELS ================================##
        if FLAGS.is_restore == True:
            # load the latest checkpoints
            load_de = tl.files.load_npz(path=os.path.join(FLAGS.checkpoint_dir, model_dir), \
                                        name='/net_de_%d00.npz' % FLAGS.c_epoch)
            load_en = tl.files.load_npz(path=os.path.join(FLAGS.checkpoint_dir, model_dir), \
                                        name='/net_en_%d00.npz' % FLAGS.c_epoch)
            load_dX = tl.files.load_npz(path=os.path.join(FLAGS.checkpoint_dir, model_dir), \
                                        name='/net_dX_%d00.npz' % FLAGS.c_epoch)
            load_dZ = tl.files.load_npz(path=os.path.join(FLAGS.checkpoint_dir, model_dir), \
                                        name='/net_dZ_%d00.npz' % FLAGS.c_epoch)
            load_dJ = tl.files.load_npz(path=os.path.join(FLAGS.checkpoint_dir, model_dir), \
                                        name='/net_dJ_%d00.npz' % FLAGS.c_epoch)
            tl.files.assign_params(sess, load_en, n_fake_Z)
            tl.files.assign_params(sess, load_de, n_fake_X)
            tl.files.assign_params(sess, load_dX, n_dic_X)
            tl.files.assign_params(sess, load_dZ, n_dic_Z)
            tl.files.assign_params(sess, load_dJ, n_dic_J)

        merged = tf.summary.merge_all()
        logger = tf.summary.FileWriter('./logs', sess.graph)
        tf.global_variables_initializer().run()

        iter_counter = 0
        for epoch in range(FLAGS.c_epoch+1, FLAGS.epoch):
            ## shuffle data
            shuffle(data_files)
            print("[*] Dataset shuffled!")
            
            ## update sample files based on shuffled data
            sample_files = data_files[0:FLAGS.sample_size]
            sample = [get_image(sample_file, FLAGS.image_size, dataset, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            print("[*] Sample images updated!")
            print sample_images.shape
            
            ## load image data
            batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size
            
            for idx in xrange(0, batch_idxs):
                ### Get datas ###
                batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                ## get real images
                # more image augmentation functions in http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html
                batch = [get_image(batch_file, FLAGS.image_size, dataset, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                #batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
                batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
                start_time = time.time()
                
                ### Update Nets ###
                # updates the discriminator
                # print ("update discriminator")
                feed_dict={z: batch_z, real_images: batch_images}
                feed_dict.update(n_dic_J.all_drop)
                feed_dict.update(n_dic_fJ.all_drop)
                feed_dict.update(n_dic_Z.all_drop)
                feed_dict.update(n_dic_fZ.all_drop)
                errX, _ = sess.run([dic_X_loss, dic_X_optim], feed_dict=feed_dict)
                errZ, _ = sess.run([dic_Z_loss, dic_Z_optim], feed_dict=feed_dict)
                errJ, _ = sess.run([dic_J_loss, dic_J_optim], feed_dict=feed_dict)

                errEN, _  = sess.run([en_loss, en_optim], feed_dict=feed_dict)
                errDE, _  = sess.run([de_loss, de_optim], feed_dict=feed_dict)

                # updates the generator, run generator 8 times to make sure 
                # that d_loss does not go to zero (difference from paper)
                for _ in range(4):
                    errfJ, _  = sess.run([dic_fJ_loss, dic_fJ_optim], feed_dict=feed_dict)
            
                errClc, _ = sess.run([clc_loss, clc_optim], feed_dict=feed_dict)

                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f" \
                           % (epoch, FLAGS.epoch, idx, batch_idxs,time.time() - start_time))

                #print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, dX_loss: %.8f, dZ_loss: %.8f, dJ_loss: %.8f, en_loss: %.8f, de_loss: %.8f" \
                #      % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errX, errZ, errJ, errEN, errDE))
                sys.stdout.flush()

                iter_counter += 1
                
                if np.mod(iter_counter, FLAGS.sample_step) == 0:
                    # generate and visualize generated images
                    summary, img, errX, errZ, errJ, errEN, errDE = \
                        sess.run([merged, n_fake_X.outputs, dic_X_loss, dic_Z_loss, dic_J_loss, en_loss, de_loss], feed_dict=feed_dict)
                    #summary, img, errJ, errfJ = sess.run([merged, n_fake_X.outputs, dic_J_loss, dic_fJ_loss], feed_dict=feed_dict)
                    logger.add_summary(summary, iter_counter)
                
                    img255 = (np.array(img) + 1) / 2 * 255
                    #img255 = np.array(img)* 255
                    tl.visualize.images2d(images=img255, second=0, saveable=True,
                                          name='./{}/train_{:02d}_{:04d}'.format(FLAGS.sample_dir, epoch, idx), dtype=None, fig_idx=2838)
    
                    save_images(img, [8, 8],
                                './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
                    
                    #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
                    sys.stdout.flush()

                if np.mod(iter_counter, FLAGS.save_step) == 0:
                    # save current network parameters
                    print("[*] Saving checkpoints...")
                    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # the latest version location
                    net_de_name = os.path.join(save_dir, 'net_de.npz')
                    net_en_name = os.path.join(save_dir, 'net_en.npz')
                    net_dX_name = os.path.join(save_dir, 'net_dX.npz')
                    net_dZ_name = os.path.join(save_dir, 'net_dZ.npz')
                    net_dJ_name = os.path.join(save_dir, 'net_dJ.npz')
                    # this version is for future re-check and visualization analysis
                    net_de_iter_name = os.path.join(save_dir, 'net_de_%d.npz' % iter_counter)
                    net_en_iter_name = os.path.join(save_dir, 'net_en_%d.npz' % iter_counter)
                    net_dX_iter_name = os.path.join(save_dir, 'net_dX_%d.npz' % iter_counter)
                    net_dZ_iter_name = os.path.join(save_dir, 'net_dZ_%d.npz' % iter_counter)
                    net_dJ_iter_name = os.path.join(save_dir, 'net_dJ_%d.npz' % iter_counter)

                    tl.files.save_npz(n_fake_X.all_params, name=net_de_name, sess=sess)
                    tl.files.save_npz(n_fake_Z.all_params, name=net_en_name, sess=sess)
                    tl.files.save_npz(n_dic_X.all_params, name=net_dX_name, sess=sess)
                    tl.files.save_npz(n_dic_Z.all_params, name=net_dZ_name, sess=sess)
                    tl.files.save_npz(n_dic_J.all_params, name=net_dJ_name, sess=sess)

                    tl.files.save_npz(n_fake_X.all_params, name=net_de_iter_name, sess=sess)
                    tl.files.save_npz(n_fake_Z.all_params, name=net_en_iter_name, sess=sess)
                    tl.files.save_npz(n_dic_X.all_params, name=net_dX_iter_name, sess=sess)
                    tl.files.save_npz(n_dic_Z.all_params, name=net_dZ_iter_name, sess=sess)
                    tl.files.save_npz(n_dic_J.all_params, name=net_dJ_iter_name, sess=sess)

                    print("[*] Saving checkpoints SUCCESS!")

        logger.close()

if __name__ == '__main__':
    tf.app.run()
