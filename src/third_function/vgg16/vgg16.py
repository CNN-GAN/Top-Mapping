import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os, sys, time
import numpy as np
from glob import glob
from scipy.misc import imread, imresize

def conv_layers(net_in):
    with tf.name_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean

    """ conv1 """
    network = Conv2dLayer(net_in, act = tf.nn.relu, shape = [3, 3, 3, 64],      # 64 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool1')
    """ conv2 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool2')
    """ conv3 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool3')
    """ conv4 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool4')
    """ conv5 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool5')
    return network

def conv_layers_simple_api(net_in):
    with tf.name_scope('preprocess') as scope:
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean

    """ conv1 """
    network = Conv2d(net_in, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_1')
    network = Conv2d(network, n_filter=64, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu,padding='SAME', name='conv1_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool1')
    """ conv2 """
    network = Conv2d(network, n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    network = Conv2d(network,n_filter=128, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool2')
    """ conv3 """
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    network = Conv2d(network, n_filter=256, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool3')
    """ conv4 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool4')
    """ conv5 """
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
    network = Conv2d(network, n_filter=512, filter_size=(3, 3),
            strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
    network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool5')
    return network

def fc_layers(net):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc1_relu')
    network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc2_relu')
    network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc3_relu')
    return network

def Seq_VGG(sess, args):

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])

    net_in = InputLayer(x, name='input')
    net_cnn = conv_layers_simple_api(net_in)  # simplified CNN APIs
    network = fc_layers(net_cnn)

    y = network.outputs
    probs = tf.nn.softmax(y)


    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()
    npz = np.load(os.path.join(args.checkpoint_dir, 'vgg16_weights.npz'))
    params = []
    for val in sorted( npz.items() ):
        print("  Loading %s" % str(val[1].shape))
        params.append(val[1])

    tl.files.assign_params(sess, params, network)

    # data iterator
    route_dir = ["Route1", "Route2", "Route3"]
    test_dir = ["FOGGY", "RAIN", "SUNNY"]
    result_dir = os.path.join(args.result_dir, 'VGG16')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

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
            test_code = np.zeros([len(test_files), 1000]).astype(np.float32)

            for img_index, file_img in enumerate(test_files):
                
                img = imread(file_img, mode='RGB') # test data in github
                img = imresize(img, (224, 224))

                prob = sess.run(probs, feed_dict={x: [img]})[0]
                test_code[img_index]  = prob

    
            print("Test code extraction time: %4.4f"  % (time.time() - start_time))
            if route_index==2:
                route_name = "Route3"
                        
            Testvector_path = os.path.join(result_dir, route_name+'_'+file_name+'_vt.npy')
            print ("save path {}".format(Testvector_path))
            np.save(Testvector_path, test_code)








