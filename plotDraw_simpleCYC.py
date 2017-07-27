from __future__ import division

import os
import sys
import json

from utils import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from parameters import *

method = 'simpleCYC'
route_dir = ["Route1", "Route2", "Route3", "Route4"]
test_dir = ["FOGGY1", "FOGGY2", "RAIN1", "RAIN2", "SUNNY1", "SUNNY2"]
#test_dir = ["JOINT"]

result_dir = os.path.join('results', method)
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

args = Param()

Base_cmp = "SUNNY1"

for epoch_id in range(18, 23):

    for route_id, route_name in enumerate(route_dir):
        Testvector_path = os.path.join(result_dir, str(epoch_id)+'_'+route_name+'_'+Base_cmp+'_vt.npy')
        train_code = np.load(Testvector_path)

        for file_id, file_name in enumerate(test_dir):

            print('Load data epoch:{}, file:{}'.format(epoch_id, file_name)) 
            Testvector_path = os.path.join(result_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_vt.npy')
            test_code = np.load(Testvector_path)
            D = Euclidean(train_code, test_code)
            D_sub = D[100:300, 300:500]
            print (D.shape)
            scipy.misc.imsave(os.path.join(matrix_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_'+Base_cmp+'_matrix.jpg'), D * 255)
            DD = enhanceContrast(D, 30)
            DD_sub = DD[100:300, 300:500]
            scipy.misc.imsave(os.path.join(matrix_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_'+Base_cmp+'_enhance.jpg'), DD * 255)

            
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
            plt.title('Epoch_'+str(epoch_id)+'_'+route_name+'_'+file_name)
            plt.savefig(os.path.join(match_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_match.jpg'))
            
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
            PR_path = os.path.join(pr_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_PR.json')
            with open(PR_path, 'w') as data_out:
                json.dump(PR_data, data_out)
    
            plt.figure()
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
            plt.title('PR Curve for Epoch_'+str(epoch_id)+'_'+route_name+'_'+file_name)
            plt.savefig(os.path.join(pr_dir, str(epoch_id)+'_'+route_name+'_'+file_name+'_PR.jpg'))




        '''
    plt.figure()
    legend = []
    for id in range(1, 20):
            with open(os.path.join('results', model[0], test_data[test_id]+'_'+str(id)+'_N_PR.json'), 'r') as data_file:
                data = json.load(data_file)
                legend.append('Epoch ' + str(id))
                results = np.array(data) # precision, recall
                plt.plot(results[:,1], results[:,0], lw=2, label='Precision-Recall curve')

                ## Measure vector corrcoeffience
                start_time = time.time()
                D          = self.vec_D(train_code, test_code)
                #D          = enhanceContrast(D, args.enhance)
                print("Distance Matrix time: %4.4f"  % (time.time() - start_time))
                
                ## Estimate matches
                start_time = time.time()
                match      = self.getMatch(D, Ann, args)
                print("Match search time: %4.4f"  % (time.time() - start_time))
                
                ## Save Matrix image
                    save_path = args.method
                    result_dir = os.path.join(args.result_dir, save_path)
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                        if not os.path.exists(os.path.join(result_dir, 'MATRIX')):
                            os.makedirs(os.path.join(result_dir, 'MATRIX'))
                            scipy.misc.imsave(os.path.join(result_dir, 'MATRIX', \
                                                           test_dir[dir_id]+'_'+str(test_epoch)+'_matrix.jpg'), D * 255)


    #plt.legend(['Cycle SeqSLAM', 'Cycle SeqSLAM(ANN)', 'Enhanced SeqSLAM', 'Enhanced SeqSLAM(ANN)','SeqSLAM'], loc='lower left')
    #plt.gca().set_color_cycle(['red', 'green', 'blue'])
    #plt.legend(['Enhanced SeqSLAM', 'Enhanced SeqSLAM (ANN)', 'SeqSLAM'], loc='lower left')
    #plt.gca().set_color_cycle(['red', 'green', 'blue'])
    
    plt.legend(legend, loc='lower left')
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.savefig(test_data[test_id]+'_PR.jpg')

    print ('plot done')
        '''
