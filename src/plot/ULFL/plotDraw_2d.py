from __future__ import division

import os
import sys
import json

from src.util.utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from parameters import *
from glob import glob
import scipy.misc
import math

def Plot_2D(args):

    #test_dir = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16"]
    test_dir = ["R2", "R4", "R6", "R8", "R10", "R12", "R14", "R16"]
    #, "T1", "T2", "T3", "T4", "T5"

    # For new_loam dataset
    if args.dataset == 'new_loam':
        sequence_name = '00'

    # For NCTL dataset            
    if args.dataset == 'NCTL':
        sequence_name = '2012-02-02'

    if args.is_one == True:
        PR_path = 'PR_one'
        Matrix_path = 'MATRIX_one'
        Match_path = 'MATCH_one'
    else:
        PR_path = 'PR'
        Matrix_path = 'MATRIX'
        Match_path = 'MATCH'
    
    result_dir = args.result_dir
    matrix_dir = os.path.join(result_dir,      Matrix_path)
    pr_dir     = os.path.join(result_dir,      PR_path)
    match_dir  = os.path.join(result_dir,      Match_path)
    pose_dir   = os.path.join(args.data_dir,   args.dataset, sequence_name)
    
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       

    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   

    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    for epoch_id in range(14, 29):
        Trainvector_img = os.path.join(result_dir, str(epoch_id)+'_joint_vt.npy')
        train_img = np.load(Trainvector_img)

        Trainvector_pose = os.path.join(pose_dir, 'R1', 'pose.txt')
        train_pose = np.loadtxt(Trainvector_pose)
        train_pose = train_pose[args.test_base:args.test_len*args.frame_skip:args.frame_skip, 1:3]

        train_path = os.path.join(pose_dir, 'R1', "img/*.jpg")
        train_files = glob(train_path)
        train_files.sort()

        for file_id, file_name in enumerate(test_dir):
            print('Load data file:{}'.format(file_name)) 

            cur_file_dir = os.path.join(matrix_dir, file_name)
            if not os.path.exists(cur_file_dir):
                os.makedirs(cur_file_dir)

            Testvector_img = os.path.join(result_dir, str(epoch_id)+'_'+file_name+'_vt.npy')
            test_img = np.load(Testvector_img)

            Testvector_pose = os.path.join(pose_dir, file_name, 'pose.txt')
            test_pose = np.loadtxt(Testvector_pose)
            test_pose = test_pose[0:args.test_len*args.frame_skip:args.frame_skip, 1:3]

            test_path  = os.path.join(pose_dir, file_name, "img/*.jpg")
            test_files = glob(test_path)
            test_files.sort()

            print (train_img.shape)
            print (test_img.shape)
            print('Load data done :{}'.format(file_name))
            #D = N2One_Euclidean(train_img, test_img)
            D = Euclidean(train_img[:,4], test_img)
            print('Done Euclidean :{}'.format(file_name))
            DD = enhanceContrast(D, 30)

            scipy.misc.imsave(os.path.join(matrix_dir, file_name, str(epoch_id)+'_matrix.jpg'), D * 255)
            scipy.misc.imsave(os.path.join(matrix_dir, file_name, str(epoch_id)+'_enhance.jpg'), DD * 255)
        
            ## Save matching 
            tmp_D = np.transpose(D)
            N = 3
            min_arr = np.zeros([len(test_img), N])
            Inf=100000000
            for i in range(len(tmp_D)):
                temp = []
                for _ in range(N):
                    if tmp_D[i, np.argmin(tmp_D[i])] < 17.3:
                        temp.append(np.argmin(tmp_D[i]))
                        tmp_D[i, np.argmin(tmp_D[i])] = Inf
                    else:
                        temp.append(0)                        

                min_arr[i] = np.array(temp)

            '''
            plt.figure()
            plt.xlabel('Test data')
            plt.ylabel('Stored data')
            for i in range (len(min_arr)):
                for j in range(N):
                    if min_arr[i,j] > 0:
                        plt.plot(i, min_arr[i,j],'b.') 

            
            plt.title('Epoch_'+str(epoch_id)+'_'+file_name)
            cur_file_dir = os.path.join(match_dir, file_name)
            if not os.path.exists(cur_file_dir):
                os.makedirs(cur_file_dir)

            plt.savefig(os.path.join(match_dir, file_name, str(epoch_id)+'_ann.jpg'))
            plt.close()
            '''

            match = getMatches(DD, 0, args)
            m = match[:,0]
            epoch_dir = os.path.join(match_dir, file_name, str(epoch_id))
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)       


            ## Extract Video
            datas = np.zeros([DD.shape[1]])
            pairs = np.zeros([DD.shape[1]])
            for i in range(D.shape[1]):
                fig, ax = plt.subplots()
                plt.xlim(0.0, DD.shape[1]*1.0)
                plt.ylim(0.0, DD.shape[1]*1.0)
                plt.xlabel('Test Sequence')
                plt.ylabel('Referend Sequence')
                if math.isnan(match[i, 0]):
                    pair = i
                else:
                    pair = match[i, 0]
                
                datas[i] = i
                pairs[i]=int(pair)
                plt.imshow(DD[:, :i+1],  cmap=plt.cm.gray, interpolation='nearest')
                plt.plot(datas[:i+1], pairs[:i+1], 'r*')

                plt_path = os.path.join(match_dir, file_name, str(epoch_id), "Match_{:04d}.jpg".format(i))
                print (plt_path)
                fig.savefig(plt_path)
                plt.close(fig) 

            ## Extract the color image
            fig, ax = plt.subplots()
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            plt.imshow(DD[:, :i+1],  cmap=plt.cm.gray, interpolation='nearest')
            plt.plot(datas[:i+1], pairs[:i+1], 'r*')
            plt_path = os.path.join(match_dir, file_name, str(epoch_id)+"_heatmap.jpg")
            fig.savefig(plt_path,  bbox_inches='tight')
            plt.close(fig) 

            im_in = cv2.imread(plt_path, cv2.IMREAD_GRAYSCALE)
            heatmap = cv2.applyColorMap(im_in, cv2.COLORMAP_JET)
            cv2.imwrite(plt_path, heatmap)

            count_id = 1
            for test_id, match_id in enumerate(m):

                if np.isnan(match_id):
                    continue

                print ("plot for test id {}".format(test_id))
                
                print (test_id)
                print (match_id)
                print (test_files[int(args.test_base + test_id*args.frame_skip)])
                print (train_files[int(args.test_base + match_id*args.frame_skip)])
                img1 = cv2.imread(test_files[int(args.test_base + test_id*args.frame_skip)])
                img2 = cv2.imread(train_files[int(args.test_base + match_id*args.frame_skip)])
                
                vis = np.concatenate((img1, img2), axis=1)
                cv2.imwrite(os.path.join(epoch_dir, str(count_id).zfill(5)+'.jpg'), vis)
                count_id += 1


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
            plt.savefig(os.path.join(match_dir, file_name, str(epoch_id)+'_match.jpg'))
            plt.close()

            ## Caculate Precision and Recall Curve
            np.set_printoptions(threshold='nan')
            match_PR = match[int(args.v_ds/2):int(match.shape[0]-args.v_ds/2), :]
            for match_id in range(len(match_PR)):
                train_id = int(match_PR[match_id, 0])
                test_id  = match_id+int(int(args.v_ds/2))
                distance = np.linalg.norm(train_pose[train_id]-test_pose[test_id])
                if distance <= args.match_distance:
                    match_PR[match_id,0] = 1
                else:
                    match_PR[match_id,0] = 0

            #print (match_PR)
            match_PR[np.isnan(match_PR)]=0
            match_path = os.path.join(pr_dir, str(epoch_id)+'_'+file_name+'_match.json')
            with open(match_path, 'w') as data_out:
                json.dump(match_PR.tolist(), data_out)

            precision, recall, _ = precision_recall_curve(match_PR[:, 0], match_PR[:, 1])
            PR_data = zip(precision, recall) 
            PR_path = os.path.join(pr_dir, str(epoch_id)+'_'+file_name+'_PR.json')
            with open(PR_path, 'w') as data_out:
                json.dump(PR_data, data_out)
            
            fpr, tpr, _ = roc_curve(match_PR[:, 0], match_PR[:, 1])
            roc_auc     = auc(fpr, tpr)
            
            ROC_data = zip(fpr, tpr) 
            PR_path = os.path.join(pr_dir, str(epoch_id)+'_'+file_name+'_ROC.json')
            with open(PR_path, 'w') as data_out:
                json.dump(PR_data, data_out)

            plt.figure()
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Recall/FPR')
            plt.ylabel('Precision/TPR')
            plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
            plt.plot(fpr, tpr, lw=2, color='deeppink', label='ROC curve')
            plt.title('PR Curve for Epoch_'+str(epoch_id)+'_'+file_name+'  (area={0:0.2f})'.format(roc_auc))
            plt.savefig(os.path.join(pr_dir, str(epoch_id)+'_'+file_name+'_PR.jpg'))
            plt.close()
            plt.clf()
