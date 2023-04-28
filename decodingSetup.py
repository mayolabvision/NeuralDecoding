import numpy as np
import sys
from scipy import io
from scipy import stats
import pickle
import time
import pandas as pd
import os.path
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#folder = '/jet/home/knoneman/NeuralDecoding/'
#folder = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/'
cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir

from preprocessing_funcs import get_spikes_with_history
from matlab_funcs import mat_to_pickle
from sklearn.model_selection import KFold
import itertools

import helpers

def get_dataParams(linenum):
    s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(linenum))
    jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)
    sess,sess_nodt = helpers.get_session(s,t,d)

    if not os.path.isfile(cwd+'/datasets/vars-'+sess+'.pickle'):
        if os.path.isfile(cwd+'/datasets/vars-'+sess_nodt+'.mat'):
            mat_to_pickle('vars-'+sess_nodt,d)
            print('preprocessed file has been properly pickled, yay')
        else:
            print('you need to go run data_formatting.m within MATLAB, then come right back (:')

    with open(cwd+'/datasets/vars-'+sess+'.pickle','rb') as f:
        neural_data,pos_binned,vel_binned,acc_binned=pickle.load(f,encoding='latin1')

    if o==0: 
        out_binned = pos_binned
    elif o==1:
        out_binned = vel_binned
    elif o==2:
        out_binned = acc_binned

    [bins_before,bins_current,bins_after] = helpers.get_bins(bn)

    ############ getting all possible combinations of neurons #############
    units = pd.read_csv(cwd+'/datasets/units-'+sess_nodt+'.csv')

    X_train0 = [[] for i in range(fo)] 
    X_flat_train0 = [[] for i in range(fo)]
    y_train0 = [[] for i in range(fo)]
    X_test = [[] for i in range(fo)]
    X_flat_test = [[] for i in range(fo)]
    y_test = [[] for i in range(fo)]
    neurons_perRepeat = []
    for r in range(num_repeats):
        mt_inds = sorted(np.random.choice(units[units['BrainArea'] == 'MT'].index, nm, replace=False))
        fef_inds = sorted(np.random.choice(units[units['BrainArea'] == 'FEF'].index, nf, replace=False))

        if nm==0:
            neuron_inds = np.array((fef_inds))
        elif nf==0:
            neuron_inds = np.array((mt_inds))
        else:
            neuron_inds = sorted(np.concatenate((np.array((mt_inds)),np.array((fef_inds)))))
       
        neurons_perRepeat.append(neuron_inds)
        neural_data2= neural_data[:,neuron_inds]
        

        X = get_spikes_with_history(neural_data2,bins_before,bins_after,bins_current)
        X = X[range(bins_before,X.shape[0]-bins_after),:,:]
        num_examples=X.shape[0]
        X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
        y=out_binned
        y = y[range(bins_before,y.shape[0]-bins_after),:]
       
        outer_cv = KFold(n_splits=fo, random_state=None, shuffle=False)
        
        for q, (train0_index, test_index) in enumerate(outer_cv.split(X)):
            X_train0[q].append(X[train0_index,:,:])
            X_flat_train0[q].append(X_flat[train0_index,:])
            y_train0[q].append(y[train0_index,:])
            X_test[q].append(X[test_index,:,:])
            X_flat_test[q].append(X_flat[test_index,:])
            y_test[q].append(y[test_index,:])

    return X_train0, X_flat_train0, y_train0, X_test, X_flat_test, y_test, neurons_perRepeat 
