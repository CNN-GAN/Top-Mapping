from __future__ import division

import os
import sys
import json

from glob import glob

import scipy.misc
from src.util.utils import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import math


def Plot_Caps(args):

    # data iterator
    if args.data_name == "GTAV":
        route_dir = ['Route1']
        test_dir = ['oFOGGY', 'oRAIN', 'oSUNNY']
    if args.data_name == "nordland":
        route_dir = ['']
        test_dir = ['spring', 'summer', 'fall', 'winter']
    if args.data_name == "CMU":
        route_dir = ['test']
        test_dir = ['track1']
                
    data_dir   = os.path.join(args.data_dir, args.data_name)
    result_dir = os.path.join(args.result,  args.method, args.data_name, args.model_time)
    matrix_dir = os.path.join(result_dir, 'MATRIX')
    pr_dir     = os.path.join(result_dir, 'PR')
    match_dir  = os.path.join(result_dir, 'MATCH')
    pair_dir = os.path.join(result_dir, 'PAIR')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       
        
    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   
        
    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    if not os.path.exists(pair_dir):
        os.makedirs(pair_dir)   


    for epoch_id in range(9, 30):
        for route_id, route_name in enumerate(route_dir):
            for w_i in range(len(test_dir)):
                Trainvector_path = os.path.join(result_dir, str(epoch_id), str(w_i+1)+'_vt.npy')
                print (Trainvector_path)
                train_code = np.load(Trainvector_path)
                #train_code = train_code[0:args.test_len]

                print(os.path.join('/data2/Top-Mapping/data/', args.data_name, route_name, test_dir[w_i], "*.JPG"))
                ## Evaulate test data
                train_files  = glob(os.path.join('/data2/Top-Mapping/data/', args.data_name, route_name, test_dir[w_i], "*.JPG"))
                train_files.sort()
                
                test_code = train_code[1]
                train_code = train_code[0]
                
                D = Euclidean(train_code, test_code)
                #D = Cosine(train_code, test_code)
                #D = np.exp(1-D)

                DD = enhanceContrast(D, 30)
                
                file_name = str(epoch_id)
                
                scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_matrix.jpg'), D * 255)
                scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_enhance.jpg'), DD * 255)
                
                ## Extract Video
                match = getMatches(DD, 0, args)
                
                datas = np.zeros([DD.shape[1]])
                pairs = np.zeros([DD.shape[1]])
                for i in range(DD.shape[1]):
                    print ('img {}'.format(i))
                    plt.figure()
                    plt.xlim(0.0, DD.shape[1]*1.0)
                    plt.ylim(0.0, DD.shape[1]*1.0)
                    plt.xlabel('Test Sequence')
                    plt.ylabel('Referend Sequence')
                    if math.isnan(match[i, 0]):
                        pair = i
                    else:
                        pair = match[i, 0]
                        
                    datas[i] = i
                    pairs[i]=pair
                
                    plt.imshow(DD[:, :i+1],  cmap=plt.cm.gray, interpolation='nearest')
                    plt.plot(datas[:i+1], pairs[:i+1], 'r*')
                    plt.savefig(os.path.join(match_dir,  '{}_{:04d}.jpg'.format(epoch_id,i)))
                    plt.close()

