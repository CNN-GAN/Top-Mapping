from random import shuffle
import scipy.misc
import numpy as np

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

def imread(path, dataset, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        img = scipy.misc.imread(path).astype(np.float)
        if dataset:
            out = np.zeros([img.shape[0], img.shape[0], 3])
            out[:,:,0] = img
            out[:,:,1] = img
            out[:,:,2] = img
            return out
        else:
            return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size, dataset, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, dataset, is_grayscale), image_size, is_crop, resize_w)

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


def getMatches(DD, v_ds, vmax, vmin, Rwindow):
    # TODO parallelize
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

    def Cosine(train, test):
        D = np.zeros([train.shape[0], test.shape[0]])
        for x in range(train.shape[0]):
            for y in range(test.shape[0]):
                D[x,y] = np.sum(train[x]*test[y])/(np.linalg.norm(train[x])*np.linalg.norm(test[y]))

