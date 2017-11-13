
import numpy as np
from scipy.stats import mode



if __name__ == '__main__':
    path = '/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/'
    filenames = ['error_analysis_0.866598984772.txt', 'error_analysis_0.866497461929.txt','error_analysis_0.865989847716.txt','error_analysis_0.865888324873.txt',
                 'error_analysis_0.865482233503.txt','error_analysis_0.865279187817.txt' #0.870964467005
#                  'error_analysis_0.864467005076.txt',
#                  'error_analysis_0.864365482234.txt',
#                  'error_analysis_0.864263959391.txt',
#                  'error_analysis_0.864162436548.txt',
                 ]
    preds=[]
    grounds=[]
    for i, file in enumerate(filenames):
        preds_i=[]
        print 'load ', path+file
        readfile=open(path+file, 'r')
        for line in readfile:
            parts=line.strip().split('\t')
            preds_i.append(int(parts[0]))
            if i==0:
                grounds.append(int(parts[1]))
        preds.append(preds_i)
        readfile.close()
    
    majority_preds = np.asarray(preds, dtype='int32')
    majority_ys= mode(np.transpose(majority_preds), axis=-1)[0][:,0]
    gold_ys = np.asarray(grounds, dtype='int32')
    majority_acc =1.0-np.not_equal(gold_ys, majority_ys).sum()*1.0/len(gold_ys)
    print 'majority: ', majority_acc
        