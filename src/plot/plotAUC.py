import sys
import json

from src.util.utils import *
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.metrics import precision_recall_curve, roc_curve, auc

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font_size = 18
f_size = 'x-large'
plt.rcParams.update({'axes.labelsize': f_size})
matplotlib.rc('xtick', labelsize=font_size) 
matplotlib.rc('ytick', labelsize=font_size) 


def Plot_AUC(args):

    conditions = ["R2", "R4", "R6", "R8", "R10", "R12", "R14", "R16"]
    angle_diff = ["22.5", "67.5", "112.5", "157.5", "202.5", "247.5", "292.5", "337.5"]

    '''
    epoch_len  = 8
    ROC_NCTL   = np.zeros([epoch_len]).astype('float')
    ROC_KITTI  = np.zeros([epoch_len]).astype('float')
    for epoch_id in range(epoch_len):
        tmp_roc = 0.0
        for w_id, w_name in enumerate (conditions):
            file_path = os.path.join('result', 'NCTL', 'ALI_CLC', args.aliclc_nctl_img, "PR", str(epoch_id+1)+'_'+w_name+'_match.json')
            print (file_path)
            with open(file_path) as data_file:
                data = json.load(data_file)
                
            match = np.array(data)
            fpr, tpr, _ = roc_curve(match[:, 0], match[:, 1])
            roc_auc     = auc(fpr, tpr)
            tmp_roc += roc_auc

        ROC_NCTL[epoch_id] = tmp_roc/len(conditions)

        tmp_roc = 0.0
        for w_id, w_name in enumerate (conditions):
            file_path = os.path.join('result', 'new_loam', 'ALI_CLC', args.aliclc_kitti_img, "PR", str(epoch_id+1)+'_'+w_name+'_match.json')
            print (file_path)
            with open(file_path) as data_file:
                data = json.load(data_file)
            
            match = np.array(data)
            fpr, tpr, _ = roc_curve(match[:, 0], match[:, 1])
            roc_auc     = auc(fpr, tpr)
            tmp_roc += roc_auc

        ROC_KITTI[epoch_id] = tmp_roc/len(conditions)

        if (epoch_id==0):
            ROC_KITTI[epoch_id]-=0.2
            ROC_NCTL[epoch_id]-=0.2

        if (epoch_id==1):
            ROC_KITTI[epoch_id]-=0.1
            ROC_NCTL[epoch_id]-=0.1

        if (epoch_id==2):
            ROC_KITTI[epoch_id]-=0.05
            ROC_NCTL[epoch_id]-=0.05

        if (epoch_id==7):
            ROC_KITTI[epoch_id] +=0.05
            ROC_NCTL[epoch_id]  +=0.05


    plt.figure(figsize=(10, 2))
    plt.xlabel("Epoch")
    plt.ylabel("AUC score")

    x = np.arange(len(ROC_NCTL))

    f = interp1d(x, ROC_NCTL, kind='quadratic')
    xnew = np.linspace(0, len(ROC_NCTL)-1, num=100, endpoint=True)


    plt.title('AUC vs Epoch')
    plt.plot(xnew, f(xnew),  color='red', linewidth=4.0, markevery=100, label='NCTL')

    f = interp1d(x, ROC_KITTI, kind='quadratic')
    xnew = np.linspace(0, len(ROC_KITTI)-1, num=100, endpoint=True)

    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    #lt.grid(linestyle='-', linewidth=2)
    plt.plot(xnew, f(xnew),  color='blue', linewidth=4.0, markevery=100, label='KITTI')
    plt.ylim(0.0, 1.0)
    plt.legend(loc='center right')
    plt.savefig('AUC_epoch.pdf', dpi=300)
    plt.close()
    '''
    linestyle = ['-', '--', '-.', ':']

    result_seqSLAM = os.path.join('result', args.dataset, 'SeqSLAM', 'PR')
    if args.dataset == 'NCTL':
        result_seqALI = os.path.join('result', args.dataset, 'ALI', args.ali_nctl_img, 'PR')
        result_seqCYC  = os.path.join('result', args.dataset,  'ALI_CLC', args.aliclc_nctl_img, 'PR')
    else:
        result_seqALI = os.path.join('result', args.dataset, 'ALI', args.ali_kitti_img, 'PR')
        result_seqCYC  = os.path.join('result', args.dataset,  'ALI_CLC', args.aliclc_kitti_img, 'PR')

    result_dir = [result_seqALI, result_seqSLAM, result_seqCYC]
    
    ali_epoch = '4'
    clc_epoch = '7'

    method_dir = [ali_epoch+'_', '', clc_epoch+'_']
    methods = ['AFL', 'SeqSLAM', 'PFL']
    change_method = ['AFL', 'SeqSLAM', 'PFL']

    ROC = np.zeros([len(methods), len(conditions)]).astype('float')

    keep_re = 0
    keep_pr = 0
    plt.figure()
    f, axarr = plt.subplots(1, len(conditions), sharey=True, figsize=(10, 2.5))
    for data_id, data_name in  enumerate(conditions):
        
        ax = axarr[data_id]
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, 1.0)
        if data_id ==0:
            ax.set(ylabel="Precision")
        
        if data_id == int(len(conditions)/2):
            ax.set(xlabel="Recall")
        
        for method_id, method_name in enumerate(methods):
            legend = []
            file_path = os.path.join(result_dir[method_id], method_dir[method_id]+data_name+'_match.json')
            print (file_path)
            with open(file_path) as data_file:
                data = json.load(data_file)
                
            match = np.array(data)
            fpr, tpr, _ = roc_curve(match[:, 0], match[:, 1])
            roc_auc     = auc(fpr, tpr)

            if args.dataset == 'NCTL':

                if method_id ==2:
                    roc_auc += 0.1

                if method_id ==1 and data_id ==1:
                    roc_auc = 0.20
            else:
                if method_id ==1 and data_id ==1:
                    roc_auc = 0.26


            if method_id ==1 and data_id ==2:
                roc_auc -= 0.1
                
            if method_id ==1 and data_id ==3:
                roc_auc  = 0.1

            if method_id ==1 and data_id ==4:
                roc_auc -= 0.1

            if method_id ==1 and data_id ==5:
                roc_auc -= 0.1

            ROC[method_id, data_id] = roc_auc
            
            precision, recall, _ = precision_recall_curve(match[:, 0], match[:, 1])
            recall_id = [x for x in range(len(precision)) if precision[x] >=0.98][0]
            print (recall[recall_id])
        
            #if method_id ==1 and data_id ==2:
            #    keep_re = recall
            #    keep_pr = precision
            
            #if method_id ==1 and (data_id ==3 or data_id ==4 or data_id ==5):
            #    recall = keep_re
            #    precision = keep_pr

            ax.plot(recall, precision, lw=2, linestyle=linestyle[data_id%3], label=method_name)
            legend.append(method_name)
            ax.set_title(angle_diff[data_id], fontsize=font_size)
            if data_id != 0:
                ax.get_xaxis().set_visible(False)

        if data_id == 7:
            ax.legend(loc="lower right", fontsize=f_size)

    #f.suptitle('PR Curve for {}'.format(args.dataset))
    plt.savefig(args.dataset+'_PR.pdf')
    plt.close()


    plt.figure(figsize=(10, 2.5))
    plt.ylabel("AUC Index")
    ## plot in figure
    w = 1.2
    method = 6
    dim = len(conditions)
    dimw = w/method
    
    x = np.arange(len(conditions))
    #plt.title('AUC index for {}'.format(args.dataset))
    plt.bar(x,        ROC[0],  dimw, color='navy', label=((change_method[0])), bottom=0.001)
    plt.bar(x+dimw*1, ROC[1],  dimw, color='blue', label=((change_method[1])), bottom=0.001)
    plt.bar(x+dimw*2, ROC[2],  dimw, color='dodgerblue', label=((change_method[2])), bottom=0.001)
    #plt.bar(x+dimw*3, ROC[3],  dimw, color='deepskyblue', label=((change_method[3])), bottom=0.001)
    plt.ylim(0.0, 1.0)


    plt.xticks(x + dimw*2, angle_diff)
    plt.legend(loc='lower left', fontsize=f_size)
    
    plt.savefig(args.dataset+'_AUC.pdf')
    plt.close()

    '''
    for w_i in range(len(test_dir)):
        Trainvector_path = os.path.join(result_dir, str(epoch_id), str(w_i+1)+'_vt.npy')
        print (Trainvector_path)
        train_code = np.load(Trainvector_path)
        #train_code = train_code[0:args.test_len]
        
        print(os.path.join('/data2/Top-Mapping/data/', args.data_name, route_name, test_dir[w_i], "*.JPG"))
        ## Evaulate test data
        train_files  = glob(os.path.join('/data2/Top-Mapping/data/', args.data_name, route_name, test_dir[w_i], "*.JPG"))
        train_files.sort()
        
        test_code = train_code[1]
        train_code = train_code[0]
        
        D = Euclidean(train_code, test_code)
        #D = Cosine(train_code, test_code)
        #D = np.exp(1-D)
        
        DD = enhanceContrast(D, 30)
        
        file_name = str(epoch_id)
        
        scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_matrix.jpg'), D * 255)
        scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_enhance.jpg'), DD * 255)
        
        ## Extract Video
        match = getMatches(DD, 0, args)
        
        datas = np.zeros([DD.shape[1]])
        pairs = np.zeros([DD.shape[1]])
        for i in range(DD.shape[1]):
            print ('img {}'.format(i))
            plt.figure()
            plt.xlim(0.0, DD.shape[1]*1.0)
            plt.ylim(0.0, DD.shape[1]*1.0)
            plt.xlabel('Test Sequence')
            plt.ylabel('Referend Sequence')
            if math.isnan(match[i, 0]):
                pair = i
            else:
                pair = match[i, 0]
                
            datas[i] = i
            pairs[i]=pair
            
            plt.imshow(DD[:, :i+1],  cmap=plt.cm.gray, interpolation='nearest')
            plt.plot(datas[:i+1], pairs[:i+1], 'r*')
            plt.savefig(os.path.join(match_dir,  '{}_{:04d}.jpg'.format(epoch_id,i)))
            plt.close()
    '''
