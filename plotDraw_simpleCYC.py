from __future__ import division

import os
import sys
import json

from utils import *
import numpy as np
from matplotlib import pyplot as plt
from parameters import *

method = 'simpleCYC'
test_dir = ["FOGGY1", "FOGGY2", "RAIN1", "RAIN2", "SUNNY1", "SUNNY2"]

result_dir = os.path.join('results', method)
matrix_dir = os.path.join(result_dir, 'MATRIX')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)   
    
if not os.path.exists(matrix_dir):
    os.makedirs(matrix_dir)       

args = Param()

for epoch_id in range(10, 30):

    Testvector_path = os.path.join(result_dir, str(epoch_id)+'_SUNNY1_vt.npy')
    train_code = np.load(Testvector_path)

    for file_id, file_name in enumerate(test_dir):

        print('Load data epoch:{}, file:{}'.format(epoch_id, file_name)) 
        Testvector_path = os.path.join(result_dir, str(epoch_id)+'_'+file_name+'_vt.npy')
        test_code = np.load(Testvector_path)
        D = Euclidean(train_code, test_code)
        #D = enhanceContrast(D, 10)
        scipy.misc.imsave(os.path.join(matrix_dir, str(epoch_id)+'_'+file_name+'_SUNNY1_matrix.jpg'), D * 255)

        ## Save matching 
        match = getMatches(D, 0, args)
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
        plt.savefig(os.path.join(result_dir, str(epoch_id)+'_'+file_name+'_match.jpg'))
                

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
