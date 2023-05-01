import os
import numpy as np
import decodingSetup
import sys
import pickle
import itertools
from itertools import product

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir

import neuronsSample
from preprocessing_funcs import get_spikes_with_history
from sklearn.model_selection import KFold

def make_name(s,t,d,m,o,nm,nf,bn,fo,fi,r):
    #SET,session,timesPrePost,binwidth,model,output,numMT,numFEF,binsPrePost,outerFolds, innerFolds, numRepeats
    #0      1          1         50      1     0     20    20        1          5            5          100
    return "s{:02d}-t{}-d{:03d}-m{:02d}-o{}-nm{:02d}-nf{:02d}-bn{}-fo{:02d}-fi{:02d}-r{:04d}".format(s,t,d,m,o,nm,nf,bn,fo,fi,r)

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def get_params(i):
    line = np.loadtxt('params_mlproject.txt')[i]
    s = int(line[1])
    t = int(line[2])
    d = int(line[3])
    m = int(line[4])
    o = int(line[5])
    nm = int(line[6])
    nf = int(line[7])
    bn = int(line[8])
    fo = int(line[9])
    fi = int(line[10]) 
    r = int(line[11])
    return s,t,d,m,o,nm,nf,bn,fo,fi,r

def make_directory(jobname):
    cwd = os.getcwd()
    f="/runs/"+jobname
    if not os.path.isdir(cwd+f):
       os.makedirs(cwd+f)
    return f

def get_session(j,t,d):
    sessions = ['pa29dir4A']
    times = [[500,300]]
    return sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])+'-dt'+str(d),sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])

def get_bins(bn):
    bins = [[6,1,6],[6,1,0]]
    return bins[bn][0],bins[bn][1],bins[bn][2]

def get_data(i,repeat,outer_fold):
    neurons_perRepeat = neuronsSample.get_neuronRepeats(i)
    s,t,d,m,o,nm,nf,bn,fo,fi,r = get_params(i)
    sess,sess_nodt = get_session(s,t,d)
    [bins_before,bins_current,bins_after] = get_bins(bn)

    with open(cwd+'/datasets/vars-'+sess+'.pickle','rb') as f:
        neural_data,pos_binned,vel_binned,acc_binned=pickle.load(f,encoding='latin1')

    neural_data2= neural_data[:,neurons_perRepeat[repeat]]

    X = get_spikes_with_history(neural_data2,bins_before,bins_after,bins_current)
    X = X[range(bins_before,X.shape[0]-bins_after),:,:]
    num_examples=X.shape[0]
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))

    if o==0:
        y=pos_binned
    elif o==1:
        y=vel_binned
    elif o==2:
        y=acc_binned

    y = y[range(bins_before,y.shape[0]-bins_after),:]

    outer_cv = KFold(n_splits=fo, random_state=None, shuffle=False)
    outer_folds = outer_cv.split(X)

    trainTest_index = next(itertools.islice(outer_cv.split(X), outer_fold, None))

    X_train0 = X[trainTest_index[0],:,:]
    X_flat_train0 = X_flat[trainTest_index[0],:]
    y_train0 = y[trainTest_index[0],:]
    X_test = X[trainTest_index[1],:,:]
    X_flat_test = X_flat[trainTest_index[1],:]
    y_test = y[trainTest_index[1],:]

    return X_train0,X_flat_train0,y_train0,X_test,X_flat_test,y_test,neurons_perRepeat[repeat]

def get_foldneuronPairs(i):
    # initialize lists
    s,t,d,m,o,nm,nf,bn,fo,fi,r = get_params(i)
    unique_combinations = []

    pairs = list(product(range(fo), range(r)))
    return pairs

def normalize_trainTest(X_train,X_flat_train,X_test,X_flat_test,y_train,y_test):
    X_test=(X_test-np.nanmean(X_train,axis=0))/(np.nanstd(X_train,axis=0))
    X_train=(X_train-np.nanmean(X_train,axis=0))/(np.nanstd(X_train,axis=0))
    X_flat_test=(X_flat_test-np.nanmean(X_flat_train,axis=0))/(np.nanstd(X_flat_train,axis=0))
    X_flat_train=(X_flat_train-np.nanmean(X_flat_train,axis=0))/(np.nanstd(X_flat_train,axis=0))
    y_test=y_test-np.mean(y_train,axis=0)
    y_train=y_train-np.mean(y_train,axis=0)
    y_zscore_test=y_test/(np.nanstd(y_train,axis=0))
    y_zscore_train=y_train/(np.nanstd(y_train,axis=0))

    return X_train,X_flat_train,X_test,X_flat_test,y_train,y_test,y_zscore_train,y_zscore_test
