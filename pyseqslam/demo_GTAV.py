from parameters import defaultParameters
from utils import AttributeDict
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os

from sklearn.metrics import precision_recall_curve
import json

from seqslam import *

def demo():

    # set the parameters

    # start with default parameters
    params = defaultParameters()    
    
    # Nordland spring dataset
    ds = AttributeDict()
    ds.name = 'train'
    ds.imagePath = '../data/GTAV/SUNNY1'
    
    ds.prefix=''
    ds.extension='.jpg'
    ds.suffix=''
    ds.imageSkip = 1     # use every n-nth image
    ds.imageIndices = range(20, 700, ds.imageSkip)    
    ds.savePath = 'results'
    ds.saveFile = '%s-%d-%d-%d' % (ds.name, ds.imageIndices[0], ds.imageSkip, ds.imageIndices[-1])
    
    ds.preprocessing = AttributeDict()
    ds.preprocessing.save = 1
    ds.preprocessing.load = 0 #1
    #ds.crop=[1 1 60 32]  # x0 y0 x1 y1  cropping will be done AFTER resizing!
    ds.crop=[]
    
    train=ds

    ds2 = deepcopy(ds)
    # Nordland winter dataset
    ds2.name = 'test'
    ds2.imageSkip = 1     # use every n-nth image
    ds2.imageIndices = range(20, 700, ds.imageSkip)    

    test_name = 'SUNNY2'
    ds2.imagePath = '../data/GTAV/'+test_name
    ds2.saveFile = '%s-%d-%d-%d' % (ds2.name, ds2.imageIndices[0], ds2.imageSkip, ds2.imageIndices[-1])
    # ds.crop=[5 1 64 32]
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
    ss = SeqSLAM(params)  
    t1=time.time()
    results = ss.run()
    t2=time.time()          
    print "time taken: "+str(t2-t1)
    
    ## show some results
    if len(results.matches) > 0:
        m = results.matches[:,0] # The LARGER the score, the WEAKER the match.
        thresh=0.95  # you can calculate a precision-recall plot by varying this threshold
        m[results.matches[:,1]>thresh] = np.nan # remove the weakest matches
        plt.plot(m,'.')      # ideally, this would only be the diagonal
        plt.title('Matchings')   
        plt.title('Matching '+ test_name)
        plt.savefig(test_name+'.jpg')

        match_PR = results.matches[int(params.matching.ds/2):int(results.matches.shape[0]-params.matching.ds/2), :]
        match_BS = np.array(range(match_PR.shape[0]))+int(int(params.matching.ds/2))
        match_EE = np.abs(match_PR[:,0] - match_BS)
        match_PR[match_EE<=80, 0] = 1
        match_PR[match_EE> 80, 0] = 0
        match_PR[np.isnan(match_PR)]=0
        precision, recall, _ = precision_recall_curve(match_PR[:, 0], match_PR[:, 1])
        PR_data = zip(precision, recall)
        result_dir = '../results/SeqGTAV'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        json_path = os.path.join(result_dir, test_name+'_PR.json')
        with open(json_path, 'w') as data_out:
            json.dump(PR_data, data_out)
                
        plt.figure()
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot(recall, precision, lw=2, color='navy', label='Precision-Recall curve')
        plt.title('PR Curve for Epoch_'+test_name)
        fig_path = os.path.join(result_dir, test_name+'_PR.jpg')
        plt.savefig(fig_path)
    else:
        print "Zero matches"          


if __name__ == "__main__":
    demo()
