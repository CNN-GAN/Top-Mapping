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
    
    result_img = os.path.join(args.result_dir, 'ALI')
    result_pcd = os.path.join(args.result_dir, 'ALI_3D')
    result_dir = os.path.join(args.result_dir, 'Joint')
    matrix_dir = os.path.join(result_dir, 'MATRIX')
    pr_dir     = os.path.join(result_dir, 'PR')
    match_dir  = os.path.join(result_dir, 'MATCH')
    
    pcd_epoch = "250"
    img_epoch = "12"    

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       

    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   

    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    Trainvector_img = os.path.join(result_img, img_epoch+'_gt_vt.npy')
    Trainvector_pcd = os.path.join(result_pcd, pcd_epoch+'_gt_vt.npy')
    train_img = np.load(Trainvector_img)
    train_pcd = np.load(Trainvector_pcd)
    train_code = np.concatenate((train_img, train_pcd), axis=1)

    for file_id, file_name in enumerate(test_dir):
        print('Load data file:{}'.format(file_name)) 
        Testvector_img = os.path.join(result_img, img_epoch+'_'+file_name+'_vt.npy')
        Testvector_pcd = os.path.join(result_pcd, pcd_epoch+'_'+file_name+'_vt.npy')
        test_img = np.load(Testvector_img)
        test_pcd = np.load(Testvector_pcd)
        test_code = np.concatenate((test_img, test_pcd), axis=1)        
        
        D = Euclidean(train_code, test_code)
        DD = enhanceContrast(D, 30)
            
        print (D.shape)
        scipy.misc.imsave(os.path.join(matrix_dir, img_epoch+'_'+pcd_epoch+'_'+file_name+'_matrix.jpg'), D * 255)
        scipy.misc.imsave(os.path.join(matrix_dir, img_epoch+'_'+pcd_epoch+'_'+file_name+'_enhance.jpg'), DD * 255)
        
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
        plt.title('Epoch_'+img_epoch+'_'+pcd_epoch+'_'+file_name)
        plt.savefig(os.path.join(match_dir, img_epoch+'_'+pcd_epoch+'_'+file_name+'_match.jpg'))
        
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
        PR_path = os.path.join(pr_dir, img_epoch+'_'+pcd_epoch+'_'+file_name+'_PR.json')
        with open(PR_path, 'w') as data_out:
            json.dump(PR_data, data_out)
            
        plt.figure()
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
        plt.title('PR Curve for Epoch_'+img_epoch+'_'+pcd_epoch+'_'+file_name)
        plt.savefig(os.path.join(pr_dir, img_epoch+'_'+pcd_epoch+'_'+file_name+'_PR.jpg'))
