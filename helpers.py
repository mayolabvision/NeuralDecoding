import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import itertools
from itertools import product
import pandas as pd
from math import ceil

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir

import neuronsSample
from preprocessing_funcs import get_spikes_with_history
from sklearn.model_selection import KFold
from random import shuffle
import glob 

def get_params(i,params):
    line = np.loadtxt(params)[i]
    print(line)
    s   = int(line[1])          # session number
    t   = int(line[2])          # time configuration around target motion onset 
    dto = int(line[3])          # output bin width
    df  = int(line[4])          # downsample factor
    wi  = int(line[5])          # input time window
    dti = int(line[6])          # input bin width
    nn  = int(line[7])          # number of total neurons
    nm  = int(line[8])          # number of MT neurons
    nf  = int(line[9])          # number of FEF neurons
    fo  = int(line[10])          # number of outer cross-validation folds 
    tp  = float(line[11])/100   # proportion of training data to train model on
    o   = int(line[12])         # output type (0 = pos, 1 = vel, 2 = acc)
    m   = int(line[13])         # model type
    r   = int(line[14])         # number of repeats
    return s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,r

def get_jobArray(*args):
    prep_args = []
    for i in range(len(args)):
        if isinstance(args[i], int):
            prep_args.append(range(args[i]))
        elif isinstance(args[i], list):
            prep_args.append(args[i])

    jobs = list(product(*prep_args))
    return jobs 

def get_session(j,t,dto,df,wi,dti):
    session = 'pa'+str(j)+'dir4A'
    times = [[500,300]]
    return session+'-pre{}-post{}-dto{:03d}-df{}-wi{:03d}-dti{:03d}'.format(times[t][0],times[t][1],dto,df,wi,dti), session+'-pre{}-post{}'.format(times[t][0],times[t][1])

def remove_overlapBins(cond,wi,dto):
    num_bins = round(int(wi)/int(dto))
    trials = np.unique(cond[:, 0])

    inbt_trials = np.where(np.modf(cond[:, 0])[0] != 0)[0]

    toss_inds = []
    for t,trial in enumerate(trials[1:]):
        if trial.is_integer():
            ind_thisTrial = np.where(cond[:, 0] == trial)[0]
            toss_inds.append(ind_thisTrial[:num_bins-1])

    toss_inds = np.array(toss_inds).flatten()
    if inbt_trials.size != 0:
        toss_inds = np.sort(np.concatenate((toss_inds,inbt_trials)))
        print('bitch')

    return toss_inds

def plot_first_column_lines(*arrays):
    # Take the first 100 rows of the first column of each array
    data = [arr[:100, 0] for arr in arrays]
        
    # Create a plot
    plt.figure(figsize=(10, 6)) 
        
    # Plot the first column of each array as lines with specified colors
    plt.plot(data[0], label='Array 1', color='black')
    for i, arr_data in enumerate(data[1:], start=2):
        plt.plot(arr_data, label=f'Array {i}', color=f'C{i}')
        
    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('First Column of Arrays (First 100 Rows)')
    plt.legend()
        
    # Show the plot
    plt.show()

def avgEye_perCondition(train_cond,train_kin,test_cond,test_kin):
    tbs = np.sort(np.unique(train_cond[:,4]))
    unique_conditions = np.sort(np.unique(train_cond[:, 5])).astype(int)
    
#plot_first_column_lines(train_kin)

    # Find the average trace for each condition
    train_kin_matrices = [] 
    for condition in unique_conditions:
        condition_mask = train_cond[:, 5] == condition
        condition_trials = np.unique(train_cond[condition_mask, 0].astype(int))

        condition_matrix = np.full((len(condition_trials), len(tbs), train_kin.shape[1]), np.nan)
        for i, trial in enumerate(condition_trials):
            trial_mask = train_cond[:,0] == trial
            trial_tbs = train_cond[trial_mask, 4]
            condition_matrix[i,np.where(np.isin(tbs,trial_tbs))[0],:] = train_kin[trial_mask,:]
            
        train_kin_matrices.append(np.nanmean(condition_matrix,axis=0))

    # Create test trace made up of the average kin for each condition
    avg_trace = np.zeros_like(test_kin)
    for trl in np.unique(test_cond[:,0]):
        trl_mask = test_cond[:, 0] == trl
        trl_cond = test_cond[trl_mask,:]
        avg_eye = train_kin_matrices[trl_cond[0,5].astype(int)-1]
       
        avg_trace[trl_mask,:] = avg_eye[np.where(np.isin(tbs, trl_cond[:,4])),:]
    
    return avg_trace

def get_data(X,o,pos_binned,vel_binned,acc_binned,cond,fo,outer_fold,bn,m,condition='all',trCo=0,teCo=0):
    if m!=2:
        if o==0:
            y = pos_binned
        elif o==1:
            y = vel_binned
        elif o==2:
            y = acc_binned
    else: #KF
        y = np.concatenate((pos_binned,vel_binned,acc_binned),axis=1)
       
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    num_examples=X.shape[0]
    
    if condition=='all':
        training_set,testing_set,valid_set = get_fold(outer_fold,bn,num_examples,m)
    else:
        training_set,testing_set,valid_set = get_foldX(outer_fold,bn,num_examples,cond,condition,int(trCo),int(teCo))

    if m!=2:
        X_train=X[training_set,:,:]
        X_flat_train=X_flat[training_set,:]
        y_train=y[training_set,:]
        X_test=X[testing_set,:,:]
        X_flat_test=X_flat[testing_set,:]
        y_test=y[testing_set,:]
        X_valid=X[valid_set,:,:]
        X_flat_valid=X_flat[valid_set,:]
        y_valid=y[valid_set,:]
        c_train=cond[training_set,:].astype(int)
        c_test=cond[testing_set,:].astype(int)

        X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid = normalize_trainTest(X_train,X_flat_train,X_test,X_flat_test,X_valid,X_flat_valid,y_train,y_test,y_valid)

    else:
        X_train=X[training_set,:]
        y_train=y[training_set,:]
        X_valid=X[valid_set,:]
        y_valid=y[valid_set,:]
        X_test=X[testing_set,:]
        y_test=y[testing_set,:]
        c_test = cond[testing_set,:]

        X_train,X_test,X_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid = normalize_trainTestKF(X_train,X_test,X_valid,y_train,y_test,y_valid)

        X_flat_train=X_train
        X_flat_test=X_test
        X_flat_valid=X_valid

    
    # calculate baseline eye traces (averaged within each condition)
    y_test_avg = avgEye_perCondition(c_train,y_train,c_test,y_test)
    y_test_zscore_avg = avgEye_perCondition(c_train,y_zscore_train,c_test,y_zscore_test)

    return X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test,y_test_avg,y_test_zscore_avg 

def make_name(l,s,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,r):
    return "l{:05d}-s{:02d}-dto{:03d}-df{}-wi{:03d}-dti{:03d}-nn{:02d}-nm{:02d}-nf{:02d}-fo{:02d}-tp{:03d}-o{}-m{:02d}-r{:04d}".format(l,s,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,r)

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def make_directory(jobname,nameOnly):
    cwd = os.getcwd()
    f="/runs/"+jobname
    if nameOnly==0:
        if not os.path.isdir(cwd+f):
           os.makedirs(cwd+f,exist_ok=True)
    return f

def get_bins(bn):
    #bins = [[6,1,6],[3,1,0],[0,1,0]]#,[6,1,0],[0,1,6],[1,1,0],[2,1,0],[3,1,0],[1,0,0],[2,0,0],[3,0,0]]
    #return bins[bn][0],bins[bn][1],bins[bn][2]
    bins = [bn,1,0]
    return bins[0],bins[1],bins[2]

def get_bins_fromTime(d,bn):
    binsPre = int(bn/d)
    binsCur = int(1)
    binsPost = int(0)
    return binsPre,binsCur,binsPost

def get_foldneuronPairs(i,params):
    s,t,dto,df,o,wi,dti,m,nm,nf,fo,fi,r = get_params(i,params)
    pairs = list(product(range(fo), range(r)))
    return pairs

def get_foldneuronmodelPairs(fo,r,mdls):
    pairs = list(product(range(fo), range(r), range(mdls)))
    return pairs

def get_foldneuronmodelrepeatPairs_twoBrainAreas(fo,r,nm,nf,mdls):
    if nm>0 and nf>0:
        nPairs = list(product(range(fo), range(0,nm,2), range(0,nf,2), range(1,r+1), range(mdls)))
    elif nm==0:
        nPairs = list(product(range(fo), [0], range(0,nf,2), range(1,r+1), range(mdls))) 
    elif nf==0:
        nPairs = list(product(range(fo), range(0,nm,2), [0], range(1,r+1), range(mdls)))
   
    # Filter out pairs where values in index 1 and 2 both equal 0
    nPairs = [pair for pair in nPairs if not (pair[1] == 0 and pair[2] == 0)]

    return nPairs

def get_modelneuronfoldrepeatPairs(fo,r,nm,nf,mdls):
    nPairs = list(product(range(mdls), range(0,nm+nf+1,5), range(fo), range(1,r+1)))
    nPairs = [pair for pair in nPairs if not (pair[1] == 0)]

    return nPairs

def get_neuronCombos(i,params):
    s,t,d,m,o,nm,nf,bn,fo,fi,r = get_params(i,params)
    if nm>0 & nf>0:
        pairs = list(product(range(0,nm,2), range(0,nf,2)))
    elif nm==0:
        pairs = list(product([0],range(0,nf,2))) 
    elif nf==0:
        pairs = list(product(range(0,nm,2), [0])) 
    return pairs

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

def normalize_trainTestKF(X_train,X_test,X_valid,y_train,y_test,y_valid):
    #Z-score "X" inputs. 
    X_train_mean=np.nanmean(X_train,axis=0) #Mean of training data
    X_train_std=np.nanstd(X_train,axis=0) #Stdev of training data
    X_train=(X_train-X_train_mean)/X_train_std #Z-score training data
    X_test=(X_test-X_train_mean)/X_train_std #Preprocess testing data in same manner as training data
    X_valid=(X_valid-X_train_mean)/X_train_std #Preprocess validation data in same manner as training data

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

    return X_train,X_test,X_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid

def get_fold(outer_fold,bins_before,num_examples,m):
    bins_before = int(bins_before)

    valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],
                 [.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]
    testing_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],
                     [.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]
    training_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],
                       [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]
        
    testing_range=testing_range_all[outer_fold]
    valid_range=valid_range_all[outer_fold]
    training_ranges=training_range_all[outer_fold]

    if m!=2:
        testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+bins_before,int(np.round(testing_range[1]*num_examples)))
        valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+bins_before,int(np.round(valid_range[1]*num_examples)))

        for j in range(len(training_ranges)): 
            training_range=training_ranges[j]
            if j==0: #If it's the first portion of the training set, make it the training set
                training_set=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples)))
            if j==1: #If it's the second portion of the training set, concatentate it to the first
                training_set_temp=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples)))
                training_set=np.concatenate((training_set,training_set_temp),axis=0)
    else:
        testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+1,int(np.round(testing_range[1]*num_examples))-1)
        valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+1,int(np.round(valid_range[1]*num_examples))-1)

        for j in range(len(training_ranges)): #Go through different separated portions of the training set
            training_range=training_ranges[j]
            if j==0: #If it's the first portion of the training set, make it the training set
                training_set=np.arange(int(np.round(training_range[0]*num_examples))+1,int(np.round(training_range[1]*num_examples))-1)
            if j==1: #If it's the second portion of the training set, concatentate it to the first
                training_set_temp=np.arange(int(np.round(training_range[0]*num_examples))+1,int(np.round(training_range[1]*num_examples))-1)
                training_set=np.concatenate((training_set,training_set_temp),axis=0)
    
    return training_set,testing_set,valid_set

def get_foldX(outer_fold,bins_before,num_examples,cond,condition,trCo,teCo):
    trCo = int(trCo)
    teCo = int(teCo)
    bins_before = int(bins_before)

    num_trls = np.unique(cond[:,0]).shape[0]

    if condition=='speed':
        spds = [10,20,10000]
        conds = cond[:,2]
        
        c1 = ceil(np.shape(conds[conds==10])[0]/(ceil(cond.shape[0]/num_trls)))
        c2 = ceil(np.shape(conds[conds==20])[0]/(ceil(cond.shape[0]/num_trls)))
        csize = np.array([c1,c2]).min()
        
        trTrls = np.unique(cond[(conds<=spds[trCo]),0])
        teTrls = np.unique(cond[(conds<=spds[teCo]),0])

    elif condition=='contrast':
        spds = [12,100,10000]
        conds = cond[:,1]
        
        c1 = ceil(np.shape(conds[conds==12])[0]/(ceil(cond.shape[0]/num_trls)))
        c2 = ceil(np.shape(conds[conds==100])[0]/(ceil(cond.shape[0]/num_trls)))
        csize = np.array([c1,c2]).min()
        
        trTrls = np.unique(cond[(conds<=spds[trCo]),0])
        teTrls = np.unique(cond[(conds<=spds[teCo]),0])

    elif condition=='ud' or condition=='ic': 
        conds = cond[:,3]
       
        if condition=='ud':
        # 0 = up, 1 = down, 2 = all 
            c1 = ceil(np.shape(conds[conds<=180])[0] / (ceil(cond.shape[0] / num_trls)))
            c2 = ceil(np.shape(conds[conds>180])[0] / (ceil(cond.shape[0] / num_trls)))
            csize = np.array([c1,c2]).min()

            if trCo==0:
                trTrls = np.unique(cond[(conds<=180), 0])
            elif trCo==1:
                trTrls = np.unique(cond[(conds>180), 0])
            else:
                trTrls = np.unique(cond[(conds < 1000), 0])
            
            if teCo==0:
                teTrls = np.unique(cond[(conds<=180), 0])
            elif teCo==1:
                teTrls = np.unique(cond[(conds>180), 0])
            else:
                teTrls = np.unique(cond[(conds<1000), 0])

        elif condition=='ic':
        # 0 = ipsi, 1 = contra, 2 = all
            c1 = ceil(np.shape(conds[(conds <= 90) | (conds > 270)])[0] / (ceil(cond.shape[0] / num_trls)))
            c2 = ceil(np.shape(conds[(conds > 90) & (conds <= 270)])[0] / (ceil(cond.shape[0] / num_trls)))
            csize = np.array([c1,c2]).min()

            if trCo==0:
                trTrls = np.unique(cond[(conds <= 90) | (conds > 270), 0])
            elif trCo==1:
                trTrls = np.unique(cond[(conds > 90) & (conds <= 270), 0])
            else:
                trTrls = np.unique(cond[(conds < 1000), 0])
            
            if teCo==0:
                teTrls = np.unique(cond[(conds <= 90) | (conds > 270), 0])
            elif teCo==1:
                teTrls = np.unique(cond[(conds > 90) & (conds <= 270), 0])
            else:
                teTrls = np.unique(cond[(conds < 1000), 0])

    trTrls = np.array(sorted(np.random.choice(trTrls, size=csize, replace=False)))
    teTrls = np.array(sorted(np.random.choice(teTrls, size=csize, replace=False)))
    
    trInds = np.array(np.where(np.isin(cond[:, 0], trTrls))).T
    teInds = np.array(np.where(np.isin(cond[:, 0], teTrls))).T

    num_examples = trInds.shape[0]

    ##############################################################
    valid_range_all = [[0, .2], [.2, .4], [.4, .6], [.6, .8], [.8, 1]]
    testing_range_all = [[.2, .4], [.4, .6], [.6, .8], [.8, 1], [0, .2]]
    training_range_all = [[[.4, 1]], [[0, .2], [.6, 1]], [[0, .4], [.8, 1]], [[0, .6], [0.2, 1]], [[0, .8]]]

    testing_range=testing_range_all[outer_fold]
    valid_range=valid_range_all[outer_fold]
    training_ranges=training_range_all[outer_fold]
    
    testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+bins_before,int(np.round(testing_range[1]*num_examples)))
    valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+bins_before,int(np.round(valid_range[1]*num_examples)))

    for j in range(len(training_ranges)): 
        training_range=training_ranges[j]
        if j==0: #If it's the first portion of the training set, make it the training set
            training_set=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples)))
        if j==1: #If it's the second portion of the training set, concatentate it to the first
            training_set_temp=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples)))
            training_set=np.concatenate((training_set,training_set_temp),axis=0)

    training_set = trInds[training_set].squeeze()
    testing_set = teInds[testing_set].squeeze()
    valid_set = trInds[valid_set].squeeze()

    return training_set,testing_set,valid_set

def get_data_classifier(X,o,pos_binned,vel_binned,acc_binned,cond,fo,fi,outer_fold,bn,m,condition='all',trCo=0,teCo=0):
    trCo = int(trCo)
    teCo = int(teCo)

    if m!=2:
        if o==0:
            y = pos_binned
        elif o==1:
            y = vel_binned
        elif o==2:
            y = acc_binned
    else: #KF
        y = np.concatenate((pos_binned,vel_binned,acc_binned),axis=1)
       
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    num_examples=X.shape[0]
    
    if condition=='all':
        training_set,testing_set,valid_set = get_fold(outer_fold,bn,num_examples,m)
    else:
        training_set,testing_set,valid_set = get_foldX(outer_fold,bn,num_examples,cond,condition,trCo,teCo)

    X_train=X[training_set,:,:]
    X_flat_train=X_flat[training_set,:]
    y_train=cond[training_set,1:4]
    X_test=X[testing_set,:,:]
    X_flat_test=X_flat[testing_set,:]
    y_test=cond[testing_set,1:4]
    X_valid=X[valid_set,:,:]
    X_flat_valid=X_flat[valid_set,:]
    y_valid=cond[valid_set,1:4]
    c_test=cond[testing_set,1:4]

    X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,_,_,_,_,_,_ = normalize_trainTest(X_train,X_flat_train,X_test,X_flat_test,X_valid,X_flat_valid,y_train,y_test,y_valid)

    return X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,c_test 


def get_data_Xconditions(line,repeat,outer_fold,shuffle,condition,trCo,teCo):

    if condition=='contrast':
        conds = cond_binned[:,0]
        c1 = ceil(np.shape(conds[conds==12])[0]/32)
        c2 = ceil(np.shape(conds[conds==100])[0]/32)
        csize = np.array([c1,c2]).min()-1

        if trCo==0:
            Xtr = neural_data2[conds==12,:]
            ytr = y[conds==12,:]
        elif trCo==1:
            Xtr = neural_data2[conds==100,:]
            ytr = y[conds==100,:]
        elif trCo==2:
            X0 = neural_data2[conds==12,:]
            y0 = y[conds==12,:]
            X1 = neural_data2[conds==100,:]
            y1 = y[conds==100,:]

        if teCo==0:
            Xte = neural_data2[conds==12,:]
            yte = y[conds==12,:]
        elif teCo==1:
            Xte = neural_data2[conds==100,:]
            yte = y[conds==100,:]

    else:
        conds = cond_binned[:,1]
        c1 = ceil(np.shape(conds[conds==10])[0]/32)
        c2 = ceil(np.shape(conds[conds==20])[0]/32)
        csize = np.array([c1,c2]).min()-1

        if trCo==0:
            Xtr = neural_data2[conds==10,:]
            ytr = y[conds==10,:]
        elif trCo==1:
            Xtr = neural_data2[conds==20,:]
            ytr = y[conds==20,:]
        elif trCo==2:
            X0 = neural_data2[conds==10,:]
            y0 = y[conds==10,:]
            X1 = neural_data2[conds==20,:]
            y1 = y[conds==20,:]
        
        if teCo==0:
            Xte = neural_data2[conds==10,:]
            yte = y[conds==10,:]
        elif teCo==1:
            Xte = neural_data2[conds==20,:]
            yte = y[conds==20,:]

    if trCo<2 and teCo<2:
        Xtr_select = sorted(np.random.choice(np.arange(0,Xtr.shape[0]-32,32), size=csize, replace=False))
        all_inds = []
        for sel in Xtr_select:
            all_inds.append(np.arange(sel,sel+32,1))
        Xtr = Xtr[np.concatenate(all_inds),:] 
        ytr = ytr[np.concatenate(all_inds),:]

        Xte_select = sorted(np.random.choice(np.arange(0,Xte.shape[0]-32,32), size=csize, replace=False))
        all_inds = []
        for sel in Xte_select:
            all_inds.append(np.arange(sel,sel+32,1))
        Xte = Xte[np.concatenate(all_inds),:] 
        yte = yte[np.concatenate(all_inds),:]

    else:
        X0_select = sorted(np.random.choice(np.arange(0,X0.shape[0]-32,32), size=csize, replace=False))
        all_inds0 = []
        for sel in X0_select:
            all_inds0.append(np.arange(sel,sel+32,1))
        X1_select = sorted(np.random.choice(np.arange(0,X1.shape[0]-32,32), size=csize, replace=False))
        all_inds1 = []
        for sel in X1_select:
            all_inds1.append(np.arange(sel,sel+32,1))
    
        X0 = X0[all_inds0,:]
        X1 = X1[all_inds1,:]
        y0 = y0[all_inds0,:]
        y1 = y1[all_inds1,:]

        shuf_array0 = np.arange(X0.shape[0])
        shuf_inds = np.random.choice(shuf_array0.shape[0], size=int(shuf_array0.shape[0]/2), replace=False)
        Xtr0 = X0[shuf_inds,:,:]
        Xte0 = np.delete(X0, shuf_inds, axis=0)
        ytr0 = y0[shuf_inds,:,:]
        yte0 = np.delete(y0, shuf_inds, axis=0)
        
        shuf_array1 = np.arange(X1.shape[0])
        shuf_inds = np.random.choice(shuf_array1.shape[0], size=int(shuf_array1.shape[0]/2), replace=False)
        Xtr1 = X1[shuf_inds,:,:]
        Xte1 = np.delete(X1, shuf_inds, axis=0)
        ytr1 = y1[shuf_inds,:,:]
        yte1 = np.delete(y1, shuf_inds, axis=0)

        Xtr = np.concatenate((Xtr0, Xtr1), axis=0)
        shuf_ind = np.random.permutation(Xtr.shape[0])
        Xtr = Xtr[shuf_ind,:,:]
        Xtr = np.reshape(Xtr, (Xtr.shape[0]*Xtr.shape[1], Xtr.shape[2]))
        ytr = np.concatenate((ytr0, ytr1), axis=0)
        ytr = ytr[shuf_ind,:,:]
        ytr = np.reshape(ytr, (ytr.shape[0]*ytr.shape[1], ytr.shape[2]))
       
        Xte = np.concatenate((Xte0, Xte1), axis=0)
        shuf_ind = np.random.permutation(Xte.shape[0])
        Xte = Xte[shuf_ind,:,:]
        Xte = np.reshape(Xte, (Xte.shape[0]*Xte.shape[1], Xte.shape[2]))
        yte = np.concatenate((yte0, yte1), axis=0)
        yte = yte[shuf_ind,:,:]
        yte = np.reshape(yte, (yte.shape[0]*yte.shape[1], yte.shape[2]))

    #print(np.mean(Xtr, axis=None))
    #print(np.mean(Xte, axis=None))

    Xtr = get_spikes_with_history(Xtr,bins_before,bins_after,bins_current)
    Xtr = Xtr[range(bins_before,Xtr.shape[0]-bins_after),:,:]
    Xte = get_spikes_with_history(Xte,bins_before,bins_after,bins_current)
    Xte = Xte[range(bins_before,Xte.shape[0]-bins_after),:,:]
    ytr = ytr[range(bins_before,ytr.shape[0]-bins_after),:]
    yte = yte[range(bins_before,yte.shape[0]-bins_after),:]
    
    num_examples=Xtr.shape[0]
    X_flat_tr=Xtr.reshape(Xtr.shape[0],(Xtr.shape[1]*Xtr.shape[2]))
    X_flat_te=Xte.reshape(Xte.shape[0],(Xte.shape[1]*Xte.shape[2]))

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
                
    X_train=Xtr[training_set,:,:]
    X_flat_train=X_flat_tr[training_set,:]
    y_train=ytr[training_set,:]
    X_test=Xte[testing_set,:,:]
    X_flat_test=X_flat_te[testing_set,:]
    y_test=yte[testing_set,:]
    X_valid=Xtr[valid_set,:,:]
    X_flat_valid=X_flat_tr[valid_set,:]
    y_valid=ytr[valid_set,:]

    X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid = normalize_trainTest(X_train,X_flat_train,X_test,X_flat_test,X_valid,X_flat_valid,y_train,y_test,y_valid)

    return X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,these_neurons 



