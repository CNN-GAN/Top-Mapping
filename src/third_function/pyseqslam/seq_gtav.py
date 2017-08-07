from src.parameters import defaultParameters
from src.utils import AttributeDict
from src.seqslam import *
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os

from sklearn.metrics import precision_recall_curve
import json

def Seq_GTAV(args):

    # start with default parameters
    params = defaultParameters()    
    
    # dirs
    data_dir   = os.path.join(args.data_dir, 'GTAV')
    result_dir = os.path.join(args.result_dir, 'SeqGTAV')
    matrix_dir = os.path.join(result_dir, 'MATRIX')
    pr_dir     = os.path.join(result_dir, 'PR')
    match_dir  = os.path.join(result_dir, 'MATCH')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)   
        
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)       

    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)   

    if not os.path.exists(match_dir):
        os.makedirs(match_dir)   

    img_base = os.path.join(data_dir, 'GTAV')
    route = ['Route1', 'Route2', 'Route3']
    weather = ['FOGGY1', 'FOGGY2', 'RAIN1', 'RAIN2', 'SUNNY1', 'SUNNY2']

    for r_id, route_name in enumerate(route):
        for w_i in range(len(weather)):
            # Train dataset
            ds = AttributeDict()
            ds.name = 'train'
            ds.imagePath = os.path.join(data_dir, route_name, weather[w_i])
            
            ds.preprocessing = AttributeDict()
            ds.preprocessing.save = 1
            ds.preprocessing.load = 0 #1
            ds.crop=[]
            
            train=ds
            
            for w_j in range(w_i+1, len(weather)):

                # Test dataset
                ds2 = deepcopy(ds)
                ds2.name = 'test'
                ds2.imagePath = os.path.join(data_dir, route_name, weather[w_j])
            
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
        
                ## show some results
                plt.figure()
                m = results.matches[:,0] # The LARGER the score, the WEAKER the match.
                thresh=0.95  # you can calculate a precision-recall plot by varying this threshold
                m[results.matches[:,1]>thresh] = np.nan # remove the weakest matches
                plt.plot(m,'.')      # ideally, this would only be the diagonal
                plt.title('Matchings')   
                plt.title('Matching '+ route_name + '_' + weather[w_i] + '_' + weather[w_j])
                plt.savefig(os.path.join(match_dir, route_name+'_'+weather[w_i]+'_'+weather[w_j]+'_match.jpg'))
                plt.close()

                # save match matrix
                match = results.matches
                match_PR = match[int(args.v_ds/2):int(match.shape[0]-args.v_ds/2), :]
                match_BS = np.array(range(match_PR.shape[0]))+int(int(args.v_ds/2))
                match_EE = np.abs(match_PR[:,0] - match_BS)
                match_PR[match_EE<=args.match_thres, 0] = 1
                match_PR[match_EE> args.match_thres, 0] = 0
                match_PR[np.isnan(match_PR)]=0
                match_path = os.path.join(pr_dir, route_name+'_'+weather[w_i]+'_'+weather[w_j]+'_match.json')
                print (match_path)
                with open(match_path, 'w') as data_out:
                    json.dump(match_PR.tolist(), data_out)
