import numpy as np
import sys
from scipy import io
from scipy import stats
import pickle
import time
import pandas as pd
import os.path
import os
import random

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
    s,t,d,m,o,nm,nf,bn,fo,fi = helpers.get_params(int(linenum))
    jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi)
    pfile = helpers.make_directory(jobname)

    sess,sess_nodt = helpers.get_session(s,t,d)

    if not os.path.isfile(cwd+'/datasets/vars-'+sess+'.pickle'):
        if os.path.isfile(cwd+'/datasets/vars-'+sess_nodt+'.mat'):
            mat_to_pickle('vars-'+sess_nodt,d)
            print('preprocessed file has been properly pickled, yay')
        else:
            print('you need to go run data_formatting.m within MATLAB, then come right back (:')

    with open(cwd+'/datasets/vars-'+sess+'.pickle','rb') as f:
        neural_data,out_binned=pickle.load(f,encoding='latin1')

    units = pd.read_csv(cwd+'/datasets/units-'+sess_nodt+'.csv')

    [bins_before,bins_current,bins_after] = helpers.get_bins(bn)

    ############ getting all possible combinations of neurons #############

    mt = sorted(units.loc[units['BrainArea'] == 'MT'].index)
    fef = sorted(units.loc[units['BrainArea'] == 'FEF'].index)

    mt_inds = []
    fef_inds = []

    for row in range(10):
        mt_inds.append(sorted(np.random.choice(mt, nm, replace=False)))
        fef_inds.append(sorted(np.random.choice(fef, nf, replace=False)))


    #mt_set = itertools.combinations(mt, nm)
    #fef_set = itertools.combinations(fef, nf)

    #for i in mt_set:
    #    mt_inds.append(i)
    #for i in fef_set:
    #    fef_inds.append(i)

    out_binned = out_binned[:,helpers.define_outputs(o)]
    outer_cv = KFold(n_splits=fo, random_state=None, shuffle=False)

    X_train0 = [[] for i in range(fo)]
    X_flat_train0 = [[] for i in range(fo)]
    y_train0 = [[] for i in range(fo)]
    X_test = [[] for i in range(fo)]
    X_flat_test = [[] for i in range(fo)]
    y_test = [[] for i in range(fo)]
    for r in range(len(fef_inds)):
        neural_data2 = neural_data[:,sorted(np.concatenate((np.array((mt_inds[r])),np.array((fef_inds[r])))))];
        X = get_spikes_with_history(neural_data2,bins_before,bins_after,bins_current)
        num_examples=X.shape[0]
        X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
        y=out_binned

        for i, (train0_index, test_index) in enumerate(outer_cv.split(X)):
            X_train0[i].append(X[train0_index,:,:])
            X_flat_train0[i].append(X_flat[train0_index,:])
            y_train0[i].append(y[train0_index,:])
            X_test[i].append(X[test_index,:,:])
            X_flat_test[i].append(X_flat[test_index,:])
            y_test[i].append(y[test_index,:])


    return X_train0, X_flat_train0, y_train0, X_test, X_flat_test, y_test
