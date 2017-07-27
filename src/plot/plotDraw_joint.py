from __future__ import division

import os
import sys
import json

from src.util.utils import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from parameters import *

def Plot_Joint(args):

    test_dir = ["T1_R0.1", "T1_R0.5", "T1_R1", "T1_R1.5", "T1_R2", "T5_R1", "T10_R1"] 
    
    result_dir = os.path.join(args.result_dir, args.method)
    matrix_dir = os.path.join(result_dir, 'MATRIX')
    pr_dir     = os.path.join(result_dir, 'PR')
    match_dir  = os.path.join(result_dir, 'MATCH')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       

    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   

    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    for id in range(1, 7):

        epoch_id = id*50
        Trainvector_path = os.path.join(result_dir, str(epoch_id)+'_T1_R0.1_vt.npy')
        train_code = np.load(Trainvector_path)
        
        for file_id, file_name in enumerate(test_dir):
            print('Load data epoch:{}, file:{}'.format(epoch_id, file_name)) 
            Testvector_path = os.path.join(result_dir, str(epoch_id)+'_'+file_name+'_vt.npy')
            test_code = np.load(Testvector_path)

            D = Euclidean(train_code, test_code)
            D_sub = D[100:300, 300:500]
            
            print (D.shape)
            scipy.misc.imsave(os.path.join(matrix_dir, str(epoch_id)+'_'+file_name+'_'+'_matrix.jpg'), D * 255)
            DD = enhanceContrast(D, 30)
            DD_sub = DD[100:300, 300:500]
            scipy.misc.imsave(os.path.join(matrix_dir, str(epoch_id)+'_'+file_name+'_'+'_enhance.jpg'), DD * 255)
            
            ## Save matching 
            match = getMatches(DD, 0, args)
            m = match[:,0]
            thresh = 0.95
            matched = match[match[:,1]<thresh, 1]
            score = np.mean(matched)
            m[match[:,1] > thresh] = np.nan
            plt.figure()
            plt.xlabel('Test data')
            plt.ylabel('Stored data')
            plt.text(60, .025, r"score=%4.4f, point=%d" % (score, len(matched)))
            plt.plot(m,'.') 
            plt.title('Epoch_'+str(epoch_id)+'_'+file_name)
            plt.savefig(os.path.join(match_dir, str(epoch_id)+'_'+file_name+'_match.jpg'))
            
            
            ## Caculate Precision and Recall Curve
            np.set_printoptions(threshold='nan')
            match_PR = match[int(args.v_ds/2):int(match.shape[0]-args.v_ds/2), :]
            match_BS = np.array(range(match_PR.shape[0]))+int(int(args.v_ds/2))
            match_EE = np.abs(match_PR[:,0] - match_BS)
            match_PR[match_EE<=args.match_thres, 0] = 1
            match_PR[match_EE> args.match_thres, 0] = 0
            match_PR[np.isnan(match_PR)]=0
            precision, recall, _ = precision_recall_curve(match_PR[:, 0], match_PR[:, 1])
            PR_data = zip(precision, recall) 
            PR_path = os.path.join(pr_dir, str(epoch_id)+'_'+file_name+'_PR.json')
            with open(PR_path, 'w') as data_out:
                json.dump(PR_data, data_out)
            
            plt.figure()
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
            plt.title('PR Curve for Epoch_'+str(epoch_id)+'_'+file_name)
            plt.savefig(os.path.join(pr_dir, str(epoch_id)+'_'+file_name+'_PR.jpg'))
