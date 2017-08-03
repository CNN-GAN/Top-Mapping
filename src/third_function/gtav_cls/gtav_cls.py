import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os, sys, time
import numpy as np
from glob import glob
from scipy.misc import imread, imresize
from data.imagenet_classes import *
from random import shuffle

def conv_layers_simple_api(net_in, is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope('VGG16', reuse=reuse):

        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
        network = Conv2d(net_in, 64, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h0/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h0/batch_norm')

        network = Conv2d(network, 128, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h1/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h1/batch_norm')

        network = Conv2d(network, 256, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h2/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h2/batch_norm')

        network = Conv2d(network, 512, (5, 5), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='h3/conv2d')
        network = BatchNormLayer(network, act=lambda x: tl.act.lrelu(x, 0.2),
                                is_train=is_train, gamma_init=gamma_init, name='h3/batch_norm')

        network = FlattenLayer(network, name='flatten')
        #network = DenseLayer(network, n_units=1024, act=tf.nn.relu, name='fc1_relu')
        network = DenseLayer(network, n_units=3, act=tf.identity, name='fc3_relu')

    return network

def get_img(file_name):
    img = imread(file_name, mode='RGB')
    #img = imresize(img, (224, 224))
    return img

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:

    x = tf.placeholder(tf.float32, [None, 64, 64, 3])
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
    
    net_in = InputLayer(x, name='input')
    network = conv_layers_simple_api(net_in)  # simplified CNN APIs
    
    y = network.outputs
    probs = tf.nn.softmax(y)
    y_op = tf.argmax(probs, 1)
    cost = tl.cost.cross_entropy(y, y_, name='cost')
    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    tl.layers.initialize_global_variables(sess)
    #network.print_params()
    network.print_layers()
    FC_var = tl.layers.get_variables_with_name('VGG16',  True, True)

    '''
    img1 = imread('data/tiger.jpeg', mode='RGB') # test data in github
    img1 = imresize(img1, (224, 224))
    
    # test classification
    start_time = time.time()
    prob = sess.run(probs, feed_dict={x: [img1]})[0]
    print("  End time : %.5ss" % (time.time() - start_time))
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])
    '''

    # Load Data files
    batch_size = 64
    data_bags = []
    for data_id in range(0, 3):
        read_path = os.path.join("./GTAV", 'data'+str(data_id), "*.jpg")
        data_file = glob(read_path)
        data_file = data_file[10:2010]
        data_code = np.ones(len(data_file)) * data_id
        datas = zip(data_file, data_code)
        data_bags = data_bags + datas


    for epoch in range(1, 10):
        
        # simple 1-step training
        shuffle(data_bags)

        ## load image data
        batch_idxs = len(data_bags) // batch_size
        
        for idx in xrange(0, batch_idxs):

            ### Get datas ###
            batch_files  = data_bags[idx*batch_size:(idx+1)*batch_size]
            ## get real images
            batch        = [get_img(batch_file[0]) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32).reshape([-1, 64, 64, 3])
            ## get real code
            batch_codes  = [batch_file[1] for batch_file in batch_files]

            optim = tf.train.GradientDescentOptimizer(0.0001).minimize(cost, var_list=FC_var)
            cost_, cp_, acc_, y_op_, _ = sess.run([cost, correct_prediction, acc, y_op, optim], \
                                           feed_dict={x:batch_images, y_: batch_codes})
            
            # print acc
            if idx%10 == 0:
                print ("Epoch: {}, batch ID: {}, Acc: {}".format(epoch, idx, acc_))

        # save npz
        net_iter_name = os.path.join('net_%d.npz' % epoch)
        tl.files.save_npz(network.all_params, name=net_iter_name, sess=sess)
