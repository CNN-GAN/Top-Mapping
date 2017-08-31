import sys
import json

from src.util.utils import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from parameters import *

def Plot_Paper3(args):

    routes = ["Route1", "Route2"]
    weathers = ["FOGGY_RAIN", "FOGGY_SUNNY", "RAIN_SUNNY"]

    linestyle = ['-', '--', '-.', ':']

    result_seqGTAV = os.path.join(args.result_dir, 'SeqGTAV', 'PR')
    result_seqCYC  = os.path.join(args.result_dir, 'simpleCYC', args.log_name, 'PR')
    result_seqVGG  = os.path.join(args.result_dir, 'VGG16', 'PR')

    result_dir = [result_seqGTAV, result_seqCYC, result_seqVGG]
    
    simple_epoch = '29'

    method_dir = ['', simple_epoch+'_', '']
    methods = ['SeqGTAV', 'simpleCYC', 'VGG16']
    change_method = ['CFL', 'SeqSLAM', 'DNN']

    ROC = np.zeros([len(routes), len(methods), len(weathers)]).astype('float')

    for route_id, route_name in enumerate(routes):
        plt.figure()
        f, axarr = plt.subplots(3)
        for method_id, method_name in  enumerate(methods):
            
            legend = []
            for weather_id, weather_name in  enumerate(weathers):
                file_path = os.path.join(result_dir[method_id], method_dir[method_id]+route_name+'_'+weather_name+'_match.json')
                print (file_path)
                with open(file_path) as data_file:
                    data = json.load(data_file)

                match = np.array(data)
                fpr, tpr, _ = roc_curve(match[:, 0], match[:, 1])
                roc_auc     = auc(fpr, tpr)
                if method_id ==0:
                    roc_auc += 0.0
                else:
                    roc_auc += 0.0                  
                ROC[route_id, method_id, weather_id] = roc_auc
                
                precision, recall, _ = precision_recall_curve(match[:, 0], match[:, 1])
                axarr[method_id].plot(recall, precision, lw=2, linestyle=linestyle[weather_id%3], label='Precision-Recall curve')
                axarr[method_id].set_title('PR Curve for '+change_method[method_id])
                axarr[method_id].set_xlim(0.0, 1.0)
                axarr[method_id].set_ylim(0.0, 1.0)
                legend.append(weather_name)

        plt.legend(legend, loc='lower left')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(route_name+'_'+'_PR.jpg')
        plt.close()


    plt.figure()
    f, axarr = plt.subplots(2)
    plt.xlabel("Weather condition")
    plt.ylabel("AUC score")
    ## plot in figure
    w = 1.2
    method = 5
    dim = len(weathers)
    dimw = w/method

    for route_id, route_name in enumerate(routes):

        x = np.arange(len(weathers))
        b1 = axarr[route_id].bar(x,        ROC[route_id, 0],  dimw, color='g', label=(('CFL')), bottom=0.001)
        b2 = axarr[route_id].bar(x+dimw*1, ROC[route_id, 1],  dimw, color='r', label=(('SeqSLAM')), bottom=0.001)
        b2 = axarr[route_id].bar(x+dimw*2, ROC[route_id, 2],  dimw, color='b', label=(('DNN')), bottom=0.001)
        axarr[route_id].set_xticklabels([])
        axarr[route_id].set_ylim(0.0, 1.0)

    plt.xticks(x + dimw*2, weathers)
    plt.legend(loc='center left', bbox_to_anchor=(-0.1, 1.3))
    #plt.ylim(0.0, 1.0)
    
    plt.savefig('AUC_score.jpg')
    plt.close()
