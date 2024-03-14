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

from handy_functions import dataSampling
from preprocessing_funcs import get_spikes_with_history
from sklearn.model_selection import KFold, train_test_split
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
    st  = int(line[14])         # evaluation metric
    pc  = int(line[15])         # whether PCA is done to input data (0 = none, 1 = seperate, 2 = together)
    r   = int(line[16])         # number of repeats
    j   = int(line[17])         # jobID multiplier
    return s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,st,pc,r,j

def make_name(l,s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,st,pc,r):
    return "{:05d}-s{:02d}-t{}-dto{:03d}-df{}-wi{:03d}-dti{:03d}-nn{:02d}-nm{:02d}-nf{:02d}-fo{:02d}-tp{:03d}-o{}-m{:02d}-stl{}-pc{}-r{:04d}".format(l,s,t,dto,df,wi,dti,nn,nm,nf,fo,int(tp*100),o,m,st,pc,r)

def make_directory(jobname,nameOnly):
    cwd = os.getcwd()
    f="/runs/"+jobname
    if nameOnly==0:
        if not os.path.isdir(cwd+f):
           os.makedirs(cwd+f,exist_ok=True)
    return f

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

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def get_possibleOuts(pos,vel,acc,cond):
    r_eye,p_eye = cart2pol(pos[:,0],pos[:,1])
    pos_pol = np.concatenate((r_eye[:,np.newaxis],p_eye[:,np.newaxis]), axis=1)
    rr_eye,pp_eye = cart2pol(vel[:,0],vel[:,1])
    vel_pol = np.concatenate((rr_eye[:,np.newaxis],pp_eye[:,np.newaxis]), axis=1)
    rrr_eye,ppp_eye = cart2pol(acc[:,0],acc[:,1])
    acc_pol = np.concatenate((rrr_eye[:,np.newaxis],ppp_eye[:,np.newaxis]), axis=1)
    
    print(cond[0:10,:])


    print(pos_pol.shape)
    print(blah)

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

def get_data(X,o,pos_binned,vel_binned,acc_binned,cond,fo,outer_fold,bn,condition='all',trCo=0,teCo=0):
    if o==0:
        y = pos_binned
    elif o==1:
        y = vel_binned
    elif o==2:
        y = acc_binned
       
    X_flat=X.reshape(X.shape[0],(X.shape[1]*X.shape[2]))
    num_examples=X.shape[0]
    
    if condition=='all':
        training_set,valid_set,testing_set = get_fold(outer_fold,bn,cond)
    else:
        training_set,valid_set,testing_set = get_foldX(outer_fold,bn,num_examples,cond,condition,int(trCo),int(teCo))

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

    return X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test 

def get_fold(outer_fold, bins_before, cond):
    trials = np.unique(cond[:,0])

    fold_size = len(trials) // 10
    fold_rem = len(trials) % 10  # Determine the remainder

    training_set, validation_set, testing_set = [], [], []
    order = [1,2,3,4,5,6,7,8,9,0]
    for i in order:
        fold_start = i * fold_size + min(i, fold_rem)
        fold_end = (i + 1) * fold_size + min(i + 1, fold_rem)
        
        test_trls = [int(num) for num in trials[fold_start:fold_end]]
        remaining_trls =  [int(num) for num in list(set(trials) - set(test_trls))] 
        train_trls, valid_trls = train_test_split(remaining_trls, test_size=0.24, shuffle=True, random_state=42)
        
        training_set.append(np.where(np.isin(cond[:,0], train_trls))[0]) 
        validation_set.append(np.where(np.isin(cond[:,0], valid_trls))[0]) 
        testing_set.append(np.where(np.isin(cond[:,0], test_trls))[0]) 

    return training_set[outer_fold], validation_set[outer_fold], testing_set[outer_fold]

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

##########################################################################

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

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



