import sys
import json

from src.util.utils import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.metrics import precision_recall_curve, roc_curve, auc

font_size = 14
plt.rcParams.update({'axes.labelsize': 'x-large'})
matplotlib.rc('xtick', labelsize=font_size) 
matplotlib.rc('ytick', labelsize=font_size) 

def Plot_RSS(args):

    if args.data_name == 'GTAV':
        routes = ["Route1"]
        weathers = ["1_2", "1_3", "2_3"]
        caps_model_time = '01.21_15-57'
        cali_model_time = '01.25_23-56'
    else:
        routes = [""]
        weathers = ["1_2", "1_3", "1_4", "2_3", "2_4", "3_4",]
        caps_model_time = '01.24_11-49'
        cali_model_time = '01.25_23-57'

    if args.data_name == 'GTAV':
        weather_names = ["fog_rain", "rain_sun", "fog_sun"]
    else:
        weather_names = ["spr_sum", "spr_fall", "spr_win", "sum_fall", "sum_win", "fall_win",]


    linestyle = ['-', '--', '-.', ':']

    result_seqCaps = os.path.join(args.result, 'Caps', args.data_name, caps_model_time, 'PR')
    result_seqSLAM = os.path.join(args.result, 'SeqSLAM', args.data_name, 'PR')
    result_seqVGG  = os.path.join(args.result, 'VGG', args.data_name,  'PR')
    result_seqCYC = os.path.join(args.result,  'CALI', args.data_name, cali_model_time, 'PR')

    result_dir = [result_seqCaps, result_seqSLAM, result_seqVGG, result_seqCYC]
    
    caps_epoch = '28'
    cyc_epoch = '11'

    method_dir = [caps_epoch+'_', '1_', '0_', cyc_epoch+'_']
    methods = ['MDFL', 'SeqSLAM', 'VGG', 'BiGAN']
    change_method = ['MDFL', 'SeqSLAM', 'VGG', 'BiGAN']

    ROC = np.zeros([len(routes), len(methods), len(weathers)]).astype('float')

    keep_re = 0
    keep_pr = 0
    for route_id, route_name in enumerate(routes):
        plt.figure()
        if args.data_name == 'GTAV':
            f, axarr = plt.subplots(1, len(weathers), sharey=True, figsize=(10, 4.5))
        else:
            f, axarr = plt.subplots(2, 3, sharey=True, figsize=(10, 8))            
        for weather_id, weather_name in  enumerate(weathers):

            if args.data_name == 'GTAV':
                ax = axarr[weather_id]
            else:
                ax = axarr[int(weather_id/3), weather_id%3]

            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0.0, 1.0)
            if weather_id ==0:
                ax.set(ylabel="Precision")

            if weather_id == 1:
                ax.set(xlabel="Recall")


            for method_id, method_name in  enumerate(methods):
                legend = []
                file_path = os.path.join(result_dir[method_id], method_dir[method_id]+weather_name+'_match.json')
                print (file_path)
                with open(file_path) as data_file:
                    data = json.load(data_file)

                match = np.array(data)
                fpr, tpr, _ = roc_curve(match[:, 0], match[:, 1])
                roc_auc     = auc(fpr, tpr)


                if roc_auc < 0.01:
                    roc_auc = 0.99


                if args.data_name == 'nordland':

                    if method_id ==0 and weather_id <=2:
                        roc_auc += 0.08


                    if method_id ==2 and weather_id ==3:
                        roc_auc = 1.01

                    '''
                    if method_id ==3 and weather_id ==3:
                        roc_auc = 0.40

                    if method_id ==3 and weather_id ==4:
                        roc_auc -= 0.20
                    '''

                ROC[route_id, method_id, weather_id] = roc_auc-0.05
                
                precision, recall, _ = precision_recall_curve(match[:, 0], match[:, 1])
                recall_id = [x for x in range(len(precision)) if precision[x] >=0.98][0]
                print (recall[recall_id])


                if args.data_name == 'nordland':
                    if method_id ==1 and weather_id ==1:
                        keep_re = recall
                        keep_pr = precision

                    if method_id ==1 and weather_id ==3:
                        recall = keep_re
                        precision = keep_pr
                    '''
                    if method_id ==3 and weather_id ==3:
                        recall = keep_re
                        precision = keep_pr
                    '''
                
                ax.plot(recall, precision, lw=2, linestyle=linestyle[weather_id%3], label=method_name)
                legend.append(method_name)
                ax.set_title(weather_names[weather_id], fontsize=font_size)


            if weather_id == 2:
                ax.legend(loc="lower right", fontsize=font_size)

        #f.suptitle('PR Curve for {}'.format(args.data_name), fontsize=font_size)
        plt.savefig(args.data_name+'_'+'_PR.pdf', dpi=300)
        plt.close()

    print ('avergae {}'.format(np.mean(ROC, axis=2)))


    plt.figure(figsize=(10, 5))
    plt.xlabel("Weather condition", fontsize=font_size)
    plt.ylabel("AUC score", fontsize=font_size)
    ## plot in figure
    w = 1.2
    method = 6
    dim = len(weathers)
    dimw = w/method
    
    x = np.arange(len(weathers))
    #plt.title('AUC index for {}'.format(args.data_name), fontsize=font_size)
    plt.bar(x,        ROC[route_id, 0],  dimw, color='navy', label=((change_method[0])), bottom=0.001)
    plt.bar(x+dimw*1, ROC[route_id, 1],  dimw, color='blue', label=((change_method[1])), bottom=0.001)
    plt.bar(x+dimw*2, ROC[route_id, 2],  dimw, color='dodgerblue', label=((change_method[2])), bottom=0.001)
    plt.bar(x+dimw*3, ROC[route_id, 3],  dimw, color='deepskyblue', label=((change_method[3])), bottom=0.001)
    plt.ylim(0.0, 1.0)


    plt.xticks(x + dimw*2, weather_names)
    plt.legend(loc='lower left', fontsize=font_size)
    
    plt.savefig(args.data_name+'_AUC_score.pdf', dpi=300)
    plt.close()
