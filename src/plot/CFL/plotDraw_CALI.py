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

def Plot_CALI(args):

    # data iterator
    if args.data_name == "GTAV":
        route_dir = ['Route1']
        test_dir = ['FOGGY1', 'RAIN1', 'SUNNY1']
    if args.data_name == "nordland":
        route_dir = ['']
        test_dir = ['spring64', 'summer64', 'fall64', 'winter64']
                
    data_dir   = os.path.join(args.data_dir, args.data_name)
    result_dir = os.path.join(args.result, args.method, args.data_name, args.model_time)
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


    for epoch_id in range(11, 12):
        for route_id, route_name in enumerate(route_dir):
            for w_i in range(len(test_dir)):
                Trainvector_path = os.path.join(result_dir, str(epoch_id), str(w_i+1)+'_vt.npy')
                train_code = np.load(Trainvector_path)

                for w_j in range(w_i+1, len(test_dir)):

                    Testvector_path = os.path.join(result_dir, str(epoch_id), str(w_j+1)+'_vt.npy')
                    test_code = np.load(Testvector_path)
                    D = Cosine(train_code, test_code)
                    D = np.exp(1-D)

                    DD = enhanceContrast(D, 30)

                    file_name = str(epoch_id) + '_' + str(w_i+1) + '_' + str(w_j+1)

                    scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_matrix.jpg'), D * 255)
                    scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_enhance.jpg'), DD * 255)
                    
                    ## Evaulate test data
                    test_files  = glob(os.path.join(args.data_dir, 'GTAV', route_name, 'o'+test_dir[w_j], "*.jpg"))
                    test_files.sort()
            
                    ## Extract Video
                    match = getMatches(DD, 0, args)
                    #print (match)

                    ## Save Matches
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
                    plt.title('Epoch_'+file_name)
                    plt.savefig(os.path.join(match_dir, file_name+'_match.jpg'))
                    plt.close()

                    ## Caculate Precision and Recall Curve
                    np.set_printoptions(threshold='nan')
                    match_PR = match[int(args.v_ds/2):int(match.shape[0]-args.v_ds/2), :]
                    match_BS = np.array(range(match_PR.shape[0]))+int(int(args.v_ds/2))
                    match_EE = np.abs(match_PR[:,0] - match_BS)
                    match_PR[match_EE<=args.match_thres, 0] = 1
                    match_PR[match_EE> args.match_thres, 0] = 0
                    match_PR[np.isnan(match_PR)]=0
                    match_path = os.path.join(pr_dir, file_name+'_match.json')
                    print (match_path)
                    with open(match_path, 'w') as data_out:
                        json.dump(match_PR.tolist(), data_out)

                    precision, recall, _ = precision_recall_curve(match_PR[:, 0], match_PR[:, 1])
                    PR_data = zip(precision, recall) 
                    PR_path = os.path.join(pr_dir, file_name+'_PR.json')
                    with open(PR_path, 'w') as data_out:
                        json.dump(PR_data, data_out)
                
                    fpr, tpr, _ = roc_curve(match_PR[:, 0], match_PR[:, 1])
                    roc_auc     = auc(fpr, tpr)
                    
                    ROC_data = zip(fpr, tpr) 
                    PR_path = os.path.join(pr_dir, file_name+'_ROC.json')
                    with open(PR_path, 'w') as data_out:
                        json.dump(PR_data, data_out)

                    plt.figure()
                    plt.xlim(0.0, 1.0)
                    plt.ylim(0.0, 1.0)
                    plt.xlabel('Recall/FPR')
                    plt.ylabel('Precision/TPR')
                    plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
                    plt.plot(fpr, tpr, lw=2, color='deeppink', label='ROC curve')
                    plt.title('PR Curve for Epoch_'+file_name+'  (area={0:0.2f})'.format(roc_auc))
                    plt.savefig(os.path.join(pr_dir, file_name+'_PR.jpg'))
                    plt.close()
                    plt.clf()
