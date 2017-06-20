from random import shuffle
import pypcd
from pyflann import *
import scipy.misc
import numpy as np
import tensorflow as tf
from tensorlayer.layers import *


## tensorboard function
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

##=================================== function for 3D Layers ================================##
##=================================== function for 3D Layers ================================##
##=================================== function for 3D Layers ================================##
def Conv3d(net, n_filter=32, filter_size=(5, 5, 5), strides=(1, 1, 1), act = None,
           padding='SAME', W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
           W_init_args = {}, b_init_args = {}, use_cudnn_on_gpu = None, data_format = None,name ='conv3d',):

    assert len(strides) == 3, "len(strides) should be 3, Conv3d and Conv3dLayer are different."
    if act is None:
        act = tf.identity
    net = Conv3dLayer(net,
                      act = act,
                      shape = [filter_size[0], filter_size[1], filter_size[2], \
                               int(net.outputs.get_shape()[-1]), n_filter],  # 32 features for each 5x5 patch
                      strides = [1, strides[0], strides[1], strides[2], 1],
                      padding = padding,
                      W_init = W_init,
                      W_init_args = W_init_args,
                      b_init = b_init,
                      b_init_args = b_init_args,
                      name = name)
    return net


def DeConv3d(net, n_out_channel = 32, filter_size=(5, 5, 5),
                   out_size = (30, 30, 30), strides = (2, 2, 2), padding = 'SAME', batch_size = None, act = None,
                   W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
                   W_init_args = {}, b_init_args = {}, name ='decnn3d'):

    assert len(strides) == 3, "len(strides) should be 3, DeConv3d and DeConv3dLayer are different."
    if act is None:
        act = tf.identity
    if batch_size is None:
        batch_size = tf.shape(net.outputs)[0]
    net = DeConv3dLayer(layer = net,
                        act = act,
                        shape = [filter_size[0], filter_size[1], filter_size[2], n_out_channel, int(net.outputs.get_shape()[-1])],
                        output_shape = [batch_size, int(out_size[0]), int(out_size[1]), int(out_size[2]), n_out_channel],
                        strides = [1, strides[0], strides[1], strides[2], 1],
                        padding = padding,
                        W_init = W_init,
                        b_init = b_init,
                        W_init_args = W_init_args,
                        b_init_args = b_init_args,
                        name = name)
    return net
##=================================== function for Dictance ================================##
##=================================== function for Dictance ================================##
##=================================== function for Dictance ================================##
def Euclidean(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.linalg.norm(train[x]-test[y])
    return D

def Manhattan(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.sum(np.abs(train[x]-test[y]))
    return D

def Chebyshev(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.max(np.abs(train[x]-test[y]))
    return D

def Cosine(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.sum(train[x]*test[y])/(np.linalg.norm(train[x])*np.linalg.norm(test[y]))
    return D

##=================================== function for PCD processing   ======================##
##=================================== function for PCD processing   ======================##
##=================================== function for PCD processing   ======================##
def get_pcd(data_file, args):

    pc = pypcd.PointCloud.from_path(data_file)
    data = pc.pc_data

    dx = data['x']
    dy = data['y']
    dz = data['z']
    dz = dz - dz.mean()


    dx = (dx/1.0).astype(int)
    dy = (dy/1.0).astype(int)
    dz = (dz/1.0).astype(int)
    
    pcd = np.array([dx, dy, dz]).transpose()
    keep = (abs(pcd[:,0]) < args.voxel_size) * \
           (abs(pcd[:,1]) < args.voxel_size) * \
           (abs(pcd[:,2]) < 4)
    
    pcd_out = pcd[keep]
    # make up the miss distance, in this case is 4 meters
    pcd_out[:,2] = pcd_out[:,2] + 4

    octree_map = np.ones([args.voxel_size, args.voxel_size, int(args.voxel_size/8)]).astype(float)*(0.0)
    octree_map[pcd_out[:,0],pcd_out[:,1], pcd_out[:,2]] = pcd_out[:,2]
    octree_map = octree_map.reshape([args.voxel_size, args.voxel_size, int(args.voxel_size/8), 1])

    return octree_map

##=================================== function for Image processing ======================##
##=================================== function for Image processing ======================##
##=================================== function for Image processing ======================##
def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    #return np.array(cropped_image)/255.
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        img = scipy.misc.imread(path).astype(np.float)
        return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size,  is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path,  is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def log(x):
    return tf.log(x + 1e-8)


##=================================== function in SeqSLAM ================================##
##=================================== function in SeqSLAM ================================##
##=================================== function in SeqSLAM ================================##
def enhanceContrast(D, enhance):
    # TODO parallelize
    DD = np.zeros(D.shape)
    
    #parfor?
    for i in range(D.shape[0]):
        a=np.max((0, i-enhance/2))
        b=np.min((D.shape[0], i+enhance/2+1))
        
        v = D[a:b, :]
        DD[i,:] = (D[i,:] - np.mean(v, 0)) / np.std(v, 0, ddof=1)
        
    #return DD-np.min(np.min(DD))
    return DD


def getMatches(DD, Ann, args):
    # Load parameters
    v_ds = args.v_ds
    vmax = args.vmax
    vmin = args.vmin
    Rwindow = args.Rwindow
    matches = np.nan*np.ones((DD.shape[1],2))    
    # parfor?
    for N in range(v_ds/2, DD.shape[1]-v_ds/2):
        # find a single match
        
        # We shall search for matches using velocities between
        # params.matching.vmin and params.matching.vmax.
        # However, not every vskip may be neccessary to check. So we first find
        # out, which v leads to different trajectories:
        
        move_min = vmin * v_ds
        move_max = vmax * v_ds    
        
        move = np.arange(int(move_min), int(move_max)+1)
        v = move.astype(float) / v_ds
        
        idx_add = np.tile(np.arange(0, v_ds+1), (len(v),1))
        idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)
        
        # this is where our trajectory starts
        n_start = N + 1 - v_ds/2    
        x= np.tile(np.arange(n_start , n_start+v_ds+1), (len(v), 1))    
        
        #TODO idx_add and x now equivalent to MATLAB, dh 1 indexing
        score = np.zeros(DD.shape[0])    
        
        # add a line of inf costs so that we penalize running out of data
        DD=np.vstack((DD, np.infty*np.ones((1,DD.shape[1]))))
        
        y_max = DD.shape[0]        
        xx = (x-1) * y_max
        
        flatDD = DD.flatten(1)
        for s in range(1, DD.shape[0]):   
            y = np.copy(idx_add+s)
            y[y>y_max]=y_max     
            idx = (xx + y).astype(int)
            ds = np.sum(flatDD[idx-1],1)
            score[s-1] = np.min(ds)
            
            
        # find min score and 2nd smallest score outside of a window
        # around the minimum 
        
        min_idx = np.argmin(score)
        min_value=score[min_idx]
        window = np.arange(np.max((0, min_idx-Rwindow/2)), np.min((len(score), min_idx+Rwindow/2)))
        not_window = list(set(range(len(score))).symmetric_difference(set(window))) #xor
        min_value_2nd = np.min(score[not_window])
        
        match = [min_idx + v_ds/2, min_value / min_value_2nd]
        matches[N,:] = match
        
    return matches

def getAnnMatches(DD, Ann, args):
    # Load parameters
    v_ds = args.v_ds
    vmax = args.vmax
    vmin = args.vmin
    Rwindow = args.Rwindow
    matches = np.nan*np.ones((DD.shape[1],2))

    # We shall search for matches using velocities between
    # params.matching.vmin and params.matching.vmax.
    # However, not every vskip may be neccessary to check. So we first find
    # out, which v leads to different trajectories:
    
    move_min = vmin * v_ds
    move_max = vmax * v_ds    
        
    # Obtain the v steps,
    # in case vmin = 0.8, vmax = 1.1, v equal to
    #     array([ 0.8,  0.9,  1. ,  1.1])
    move = np.arange(int(move_min), int(move_max)+1)
    v = move.astype(float) / v_ds

    # Obtain the addition (v,ds) matrix,
    # in case of v_ds = 10, and v as above, the idx_add is equal to
    # array([[  0.,   0.,   1.,   2.,   3.,   4.,   4.,   5.,   6.,   7.,   8.],
    #        [  0.,   0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
    #        [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.],
    #        [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  11.]]
    idx_add = np.tile(np.arange(0, v_ds+1), (len(v),1))
    idx_add = np.floor(idx_add * np.tile(v, (idx_add.shape[1], 1)).T)

    # Obtain the base (v,ds) matrix,
    # in case of the above setting, and N = 100, x is equal to
    # array([[ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106],
    #        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106],
    #        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106],
    #        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106]])
    n_start = 1 - v_ds/2    
    x= np.tile(np.arange(n_start , n_start+v_ds+1), (len(v), 1))  

    # parfor?
    for N in range(v_ds/2, DD.shape[1]-v_ds/2):
        x_n = x+N
        # add a line of inf costs so that we penalize running out of data
        DD=np.vstack((DD, np.infty*np.ones((1,DD.shape[1]))))
        
        # Extend the base (v,ds) matrix into a flatten based index
        # in case of the data len is 1000, xx is equal to
        # array([[ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000],
        #        [ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000],
        #        [ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000],
        #        [ 95000,  96000,  97000,  98000,  99000, 100000, 101000, 102000, 103000, 104000, 105000]])
        y_max = DD.shape[0]      
        xx = (x_n-1) * y_max
        flatDD = DD.flatten(1)

        print (y_max)
        # Initial Score for K nearest
        score = np.zeros(Ann.shape[1])            

        ## Obtain the score of K nearest points
        for id in range(Ann.shape[1]):
            s = Ann[N, id]
            y = np.copy(idx_add+s)
            y[y>y_max]=y_max     
            idx = (xx + y).astype(int)
            ds = np.sum(flatDD[idx-1],1)
            score[id] = np.min(ds)
            
        # find min score and 2nd smallest score outside of a window
        # around the minimum 
        min_idx   = Ann[N, np.argmin(score)]
        min_value = score[np.argmin(score)]

        a1 = Ann[N,:] > (min_idx + Rwindow/2)
        a2 = Ann[N,:] < (min_idx - Rwindow/2)


        match = [min_idx + v_ds/2, 1. / min_value]
        if match[1] > 1:
            match = 1.0

        '''
        if len(score[a1+a2]) > 0:
            min_value_2nd = np.min(score[a1+a2])
            match = [min_idx + v_ds/2, min_value_2nd / min_value]
        else:
            match = [min_idx + v_ds/2, 0.2]
        '''
        matches[N,:] = match
    
    return matches

def getANN(data, test, k=10):

    flann = FLANN()
    result, dists = flann.nn(data, test, k, algorithm="kmeans", branching=32, iterations=10, checks=16)
    return result, dists
