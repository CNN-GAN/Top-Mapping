from __future__ import division

import os
import sys
import json

import numpy as np
from matplotlib import pyplot as plt

#model = ['ALI_CLC', 'ALI', 'Seq']
model = ['ALI', 'ALI']
#test_data = ['test_T1_R0.1', 'test_T1_R1', 'test_T1_R1.5', 'test_T1_R2', \
#             'test_T10_R1', 'test_T10_R2', 'test_T10_R2', 'test_T20_R2', 'test_T20_R2.5']
#json_file = ['_17_PR.json', '_16_PR.json', '_PR.json']
#json_file = ['_16_N_PR.json', '_16_N_PR.json', '_16_N_PR.json', '_16_N_PR.json', '_16_N_PR.json',]
test_data = ["test_T1_R0.1","test_T1_R1", "test_T1_R2", "test_T5_R0.5",\
            "test_T10_R1", "test_T10_R2", "test_T15_R1.5", "test_T20_R2"]

for test_id in range(len(test_data)):

    plt.figure()
    legend = []

    for id in range(3, 20):
        if id%3 != 0:
            continue
        #with open(os.path.join('results', model[id], test_data[test_id]+json_file[id]), 'r') as data_file:
        #    data = json.load(data_file)
        with open(os.path.join('results', model[0], test_data[test_id]+'_'+str(id)+'_N_PR.json'), 'r') as data_file:
            data = json.load(data_file)
        legend.append('Epoch ' + str(id))
        results = np.array(data) # precision, recall
        plt.plot(results[:,1], results[:,0], lw=2, label='Precision-Recall curve')

    #plt.legend(['A-OP', 'A', 'Seq'], loc='lower left')
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
