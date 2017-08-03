from src.third_function.pyseqslam.parameters import defaultParameters
from src.third_function.pyseqslam.utils import AttributeDict
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os
import scipy.misc

from sklearn.metrics import precision_recall_curve, roc_curve, auc
import json

from seqslam import *

def Seq(args):

    # set the parameters
    test_dir = ["T1_R1", "T1_R1.5", "T1_R2", "T5_R1", "T5_R1.5", "T5_R2",  "T10_R1", "T10_R1.5", "T10_R2", "T20_R1", "T20_R1.5", "T20_R2"]

    # start with default parameters
    params = defaultParameters()    
    
    # dirs
    result_dir = os.path.join(args.result_dir, 'SeqSLAM')
    matrix_dir = os.path.join(result_dir, 'MATRIX')
    pr_dir     = os.path.join(result_dir, 'PR')
    match_dir  = os.path.join(result_dir, 'MATCH')
    data_dir   = os.path.join(args.data_dir, 'new_loam', '00')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       

    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   

    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    Trainvector_pose = os.path.join(data_dir, 'gt', 'pose.txt')
    train_pose = np.loadtxt(Trainvector_pose)
    train_pose = train_pose[0:args.test_len*args.frame_skip:args.frame_skip, 1:3]        

    # Nordland spring dataset
    ds = AttributeDict()
    ds.name = 'train'
    ds.imagePath = os.path.join(data_dir, 'gt', 'img')
    
    ds.imageSkip = args.frame_skip  # use every n-nth image
    ds.imageIndices = range(10, 1010, ds.imageSkip)    
    ds.savePath = 'results'
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])
    
    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 0 #1
    #ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[]
        
    train=ds

    for file_id, file_name in enumerate(test_dir):

        Testvector_pose = os.path.join(data_dir, file_name, 'pose.txt')
        test_pose = np.loadtxt(Testvector_pose)
        test_pose = test_pose[0:args.test_len*args.frame_skip:args.frame_skip, 1:3]
        
        ds2 = deepcopy(ds)
        # Nordland winter dataset
        ds2.name = 'test'
        
        ds2.imagePath = os.path.join(data_dir, file_name, 'img')
        ds2.saveFile = '%s-%d-%d-%d' % (ds2.name, ds2.imageIndices[0], ds2.imageSkip, ds2.imageIndices[-1])
        ds2.crop=[]
        
        test=ds2      
        
        params.dataset = [train, test]
    
        # load old results or re-calculate?
        params.differenceMatrix.load = 0
        params.contrastEnhanced.load = 0
        params.matching.load = 0
        
        # where to save / load the results
        params.savePath='results'
        
        ## now process the dataset
        ss = SeqSLAM(params, args) 
        t1=time.time()
        results = ss.run()
        t2=time.time()          
        print "time taken: "+str(t2-t1)
    
        scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_'+'_matrix.jpg'), results.D * 255)
        scipy.misc.imsave(os.path.join(matrix_dir, file_name+'_'+'_enhance.jpg'), results.DD * 255)

        ## show some results
        if len(results.matches) > 0:
            plt.figure()
            m = results.matches[:,0] # The LARGER the score, the WEAKER the match.
            thresh=0.95  # you can calculate a precision-recall plot by varying this threshold
            m[results.matches[:,1]>thresh] = np.nan # remove the weakest matches
            plt.plot(m,'.')      # ideally, this would only be the diagonal
            plt.title('Matchings')   
            plt.title('Matching '+ file_name)
            plt.savefig(os.path.join(match_dir, file_name+'_match.jpg'))
        
            ## Caculate Precision and Recall Curve
            np.set_printoptions(threshold='nan')
            #match_PR = results.matches
            match_PR = results.matches[(int(args.v_ds)+10):int(results.matches.shape[0]-args.v_ds-10), :]

            for match_id in range(len(match_PR)):
                train_id = int(match_PR[match_id, 0])
                test_id  = match_id+int(int(args.v_ds))+10
                distance = np.linalg.norm(train_pose[train_id]-test_pose[test_id])

                if distance <= args.match_distance:
                    match_PR[match_id,0] = 1
                else:
                    match_PR[match_id,0] = 0

            # print
            match_PR[np.isnan(match_PR)]=0
            match_path = os.path.join(pr_dir, file_name+'_match.json')
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
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
            plt.plot(fpr, tpr, lw=2, color='deeppink', label='ROC curve')
            plt.title('PR Curve for Epoch_'+file_name+'  (area={0:0.2f})'.format(roc_auc))
            fig_path = os.path.join(pr_dir, file_name+'_PR.jpg')
            plt.savefig(fig_path)
        else:
            print "Zero matches"
