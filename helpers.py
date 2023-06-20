import os
import numpy as np
#import decodingSetup
import sys
import pickle
import itertools
from itertools import product

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir
params = 'params.txt'
data_folder     = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/datasets/'

import neuronsSample
from preprocessing_funcs import get_spikes_with_history
from sklearn.model_selection import KFold
from random import shuffle
import glob 

def make_name(s,t,d,m,o,nm,nf,bn,fo,fi,r):
    return "s{:02d}-t{}-d{:03d}-m{:02d}-o{}-nm{:02d}-nf{:02d}-bn{}-fo{:02d}-fi{:02d}-r{:04d}".format(s,t,d,m,o,nm,nf,bn,fo,fi,r)

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def get_params(i):
    line = np.loadtxt(params)[i]
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
       os.makedirs(cwd+f,exist_ok=True)
    return f

def get_session(j,t,d):
    session = 'pa'+str(j)+'dir4A'
    times = [[500,300]]
    return session+'-pre'+str(times[t][0])+'-post'+str(times[t][1])+'-dt'+str(d),session+'-pre'+str(times[t][0])+'-post'+str(times[t][1])

def get_bins(bn):
    bins = [[6,1,6],[6,1,0],[0,1,6],[1,1,0],[2,1,0],[3,1,0],[1,0,0],[2,0,0],[3,0,0]]
    return bins[bn][0],bins[bn][1],bins[bn][2]

def get_data(line,repeat,outer_fold,shuffle):
    neurons_perRepeat = neuronsSample.get_neuronRepeats(line)
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
    
    sess,sess_nodt = get_session(s,t,d)
    [bins_before,bins_current,bins_after] = get_bins(bn)
    with open(cwd+'/datasets/vars/vars-'+sess+'.pickle','rb') as f:
        neural_data,pos_binned,vel_binned,acc_binned=pickle.load(f,encoding='latin1')
    
    neural_data2 = neural_data[:,neurons_perRepeat[repeat]]
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

    if shuffle==1:
        y2 = y.T
        idx = np.random.rand(*y2.shape).argsort(axis=1)
        y2 = np.take_along_axis(y2,idx,axis=1)
        y = y2.T

    valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],
                 [.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]
    testing_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],
                     [.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]
    training_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],
                       [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]

    testing_range=testing_range_all[outer_fold]
    testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+bins_before,int(np.round(testing_range[1]*num_examples))-bins_after)
    valid_range=valid_range_all[outer_fold]
    valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+bins_before,int(np.round(valid_range[1]*num_examples))-bins_after)

    training_ranges=training_range_all[outer_fold]
    for j in range(len(training_ranges)): 
        training_range=training_ranges[j]
        if j==0: #If it's the first portion of the training set, make it the training set
            training_set=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples))-bins_after)
        if j==1: #If it's the second portion of the training set, concatentate it to the first
            training_set_temp=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples))-bins_after)
            training_set=np.concatenate((training_set,training_set_temp),axis=0)
                
    X_train=X[training_set,:,:]
    X_flat_train=X_flat[training_set,:]
    y_train=y[training_set,:]
    X_test=X[testing_set,:,:]
    X_flat_test=X_flat[testing_set,:]
    y_test=y[testing_set,:]
    X_valid=X[valid_set,:,:]
    X_flat_valid=X_flat[valid_set,:]
    y_valid=y[valid_set,:]

    X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid = normalize_trainTest(X_train,X_flat_train,X_test,X_flat_test,X_valid,X_flat_valid,y_train,y_test,y_valid)

    return X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,neurons_perRepeat[repeat] 

def get_foldneuronPairs(i):
    s,t,d,m,o,nm,nf,bn,fo,fi,r = get_params(i)
    pairs = list(product(range(fo), range(r)))
    return pairs

def get_neuronCombos(i):
    s,t,d,m,o,nm,nf,bn,fo,fi,r = get_params(i)
    pairs = list(product(range(0,nm,2), range(0,nf,2)))
    return pairs

def get_sessConditions(s):
    contrasts,speeds,directions = [],[],[]
    for idx, x in enumerate(glob.glob(data_folder+'vars/vars-pa{:0>2d}dir4A-*-c*.mat'.format(s))):
        contrasts.append(int(x[-7:-4]))
    for idx, x in enumerate(glob.glob(data_folder+'vars/vars-pa{:0>2d}dir4A-*-sp*.mat'.format(s))):
        speeds.append(int(x[-6:-4]))
    for idx, x in enumerate(glob.glob(data_folder+'vars/vars-pa{:0>2d}dir4A-*-d*.mat'.format(s))):
        directions.append(int(x[-7:-4]))

    return sorted(contrasts), sorted(speeds), sorted(directions)

def normalize_trainTest(X_train,X_flat_train,X_test,X_flat_test,X_valid,X_flat_valid,y_train,y_test,y_valid):
    #Z-score "X" inputs. 
    X_train_mean=np.nanmean(X_train,axis=0) #Mean of training data
    X_train_std=np.nanstd(X_train,axis=0) #Stdev of training data
    X_train=(X_train-X_train_mean)/X_train_std #Z-score training data
    X_test=(X_test-X_train_mean)/X_train_std #Preprocess testing data in same manner as training data
    X_valid=(X_valid-X_train_mean)/X_train_std #Preprocess validation data in same manner as training data

    #Z-score "X_flat" inputs. 
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)
    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    X_flat_valid=(X_flat_valid-X_flat_train_mean)/X_flat_train_std

    #Zero-center outputs
    y_train_mean=np.nanmean(y_train,axis=0) #Mean of training data outputs
    y_train=y_train-y_train_mean #Zero-center training output
    y_test=y_test-y_train_mean #Preprocess testing data in same manner as training data
    y_valid=y_valid-y_train_mean #Preprocess validation data in same manner as training data
    
    #Z-score outputs (for SVR)
    y_train_std=np.nanstd(y_train,axis=0)
    y_zscore_train=y_train/y_train_std
    y_zscore_test=y_test/y_train_std
    y_zscore_valid=y_valid/y_train_std  

    return X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid

