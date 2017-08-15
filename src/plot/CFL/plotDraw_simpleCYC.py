from __future__ import division

import os
import sys
import json

from src.util.utils import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from parameters import *

def Plot_simpleCYC(args):

    route_dir = ["Route1", "Route2"]
    test_dir = ["FOGGY", "RAIN", "SUNNY"]
    
    result_dir = os.path.join(args.result_dir, 'simpleCYC')
    matrix_dir = os.path.join(result_dir, 'MATRIX')
    pr_dir = os.path.join(result_dir, 'PR')
    match_dir = os.path.join(result_dir, 'MATCH')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       
        
    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   
        
    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    for epoch_id in range(29, 30):
        for route_id, route_name in enumerate(route_dir):
            for w_i in range(len(test_dir)):
                Trainvector_path = os.path.join(result_dir, \
                                                str(epoch_id)+'_'+route_name+'_'+test_dir[w_i]+'_vt.npy')
                train_code = np.load(Trainvector_path)
                #train_code = train_code[0:args.test_len]
            
                for w_j in range(w_i+1, len(test_dir)):

                    file_name = str(epoch_id)+'_'+route_name+'_'+test_dir[w_i]+'_'+test_dir[w_j]
                    print('Load data epoch:{}, file:{}'.format(epoch_id, test_dir[w_j])) 
                    Testvector_path = os.path.join(result_dir, str(epoch_id)+'_'+route_name+'_'+test_dir[w_j]+'_vt.npy')
                    test_code = np.load(Testvector_path)
                    #test_code = test_code[0:args.test_len*args.frame_skip:args.frame_skip]
                    D = Euclidean(train_code, test_code)
                    DD = enhanceContrast(D, 30)

                    scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_matrix.jpg'), D * 255)
                    scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_enhance.jpg'), DD * 255)
                    
                    #D_sub = D[100:300, 300:500]
                    #scipy.misc.imsave(os.path.join(matrix_dir, \
                    #            str(epoch_id)+'_'+route_name+'_'+file_name+'_'+Base_cmp+'_matrix.jpg'), D * 255)

                    #DD_sub = DD[100:300, 300:500]
                    #scipy.misc.imsave(os.path.join(matrix_dir, \
                    #            str(epoch_id)+'_'+route_name+'_'+file_name+'_'+Base_cmp+'_enhance.jpg'), DD * 255)

            
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




                '''
                ## Save sub matrix
                plt.figure()
                fig, ax = plt.subplots()
                ax.imshow(D_sub, cmap=plt.cm.gray, interpolation='nearest')
                #ax.set_title('Difference Matrix')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                plt.savefig(os.path.join(matrix_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_FOGGY1_sub_matrix.jpg'))
            
                ## Save sub enhance
                plt.figure()
                fig, ax = plt.subplots()
                ax.imshow(DD_sub, cmap=plt.cm.gray, interpolation='nearest')
                #ax.set_title('Enhance Matrix')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                plt.savefig(os.path.join(matrix_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_FOGGY1_sub_enhance.jpg'))
                '''
