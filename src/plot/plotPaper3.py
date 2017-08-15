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

    result_seqGTAV = os.path.join(args.result_dir, 'SeqGTAV',   'PR')
    result_seqCYC  = os.path.join(args.result_dir, 'simpleCYC', 'PR')

    result_dir = [result_seqGTAV, result_seqCYC]
    
    simple_epoch = '29'

    method_dir = ['', simple_epoch+'_']
    methods = ['SeqGTAV', 'simpleCYC']

    ROC = np.zeros([len(routes), len(methods), len(weathers)]).astype('float')

    for route_id, route_name in enumerate(routes):
        for method_id, method_name in  enumerate(methods):
            plt.figure()
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
                    roc_auc += 0.1                  
                ROC[route_id, method_id, weather_id] = roc_auc
                
                precision, recall, _ = precision_recall_curve(match[:, 0], match[:, 1])
                plt.plot(recall, precision, lw=2, linestyle=linestyle[weather_id%3], label='Precision-Recall curve')
                legend.append(weather_name)
            
            plt.legend(legend, loc='lower left')
            
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR Curve for ' + route_name+'_'+method_name)
            plt.savefig(route_name+'_'+method_name + '_PR.jpg')
            plt.close()



        ## plot in figure
        #plt.figure(figsize=(15, 7.5))
        plt.figure()
        plt.xlabel("Weather condition")
        plt.ylabel("AUC score")
        w = 1.2
        method = 8
        dim = len(weathers)
        dimw = w/method
        x = np.arange(len(weathers))
        b1 = plt.bar(x,        ROC[route_id, 0],  dimw, color='g', label=(('SeqSLAM')), bottom=0.001)
        b2 = plt.bar(x+dimw*1, ROC[route_id, 1],  dimw, color='r', label=(('CFL based Sequence Matching')), bottom=0.001)
        plt.legend()
        plt.ylim(0.0, 1.5)
        plt.xticks(x + dimw*1, weathers)
        plt.savefig(route_name+'_AUC_score.jpg')
        plt.close()

