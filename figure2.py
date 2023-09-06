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
from sklearn.utils import shuffle
from scipy.stats import ttest_1samp
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', 'Solver terminated early.*')

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir
params = 'params/params_fig2a.txt'

from metrics import get_R2
from metrics import get_rho
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
from psutil import cpu_count
import helpers
import neuronsSample

line = np.loadtxt(params)[int(sys.argv[1])]
print(line)
s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
foldneuron_pairs = helpers.get_foldneuronPairs(int(sys.argv[1]))

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    neuron_fold = foldneuron_pairs[int(sys.argv[3])]
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    neuron_fold = foldneuron_pairs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

outer_fold = neuron_fold[0]
repeat = neuron_fold[1]

[neurons_perRepeat,nm,nf] = neuronsSample.get_neuronRepeats(s,t,d,num_repeats)
these_neurons = neurons_perRepeat[repeat]

sess,sess_nodt = helpers.get_session(s,t,d)
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)

with open(cwd+'/datasets/vars/vars-'+sess+'.pickle','rb') as f:
    neural_data,pos_binned,vel_binned,acc_binned,cond_binned=pickle.load(f,encoding='latin1')

if o==0:
    y = pos_binned
elif o==1:
    y = vel_binned
elif o==2:
    y = acc_binned

if m==8:
    y = vel_binned

############ training ################
X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test = helpers.get_data(neural_data,these_neurons,y,cond_binned,outer_fold,bn,d,m)

init_points = 10
n_iter = 10
num_permutations = 1000

t1=time.time()
##################### Wiener Filter Decoder ############################
if m == 0:
    from decoders import WienerFilterDecoder
    model=WienerFilterDecoder()
    coeffs,intercept = model.fit(X_flat_train,y_train)
    y_test_predicted=model.predict(X_flat_test)   
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)

    print("R2 = {}".format(r2))

##################### Wiener Cascade Decoder ###########################
if m == 1:
    from decoders import WienerCascadeDecoder
    def wc_evaluate(degree):
        model_wc=WienerCascadeDecoder(degree) 
        model_wc.fit(X_flat_train,y_train) 
        y_valid_predicted_wc=model_wc.predict(X_flat_valid) 
        return np.mean(get_R2(y_valid,y_valid_predicted_wc))
    BO = BayesianOptimization(wc_evaluate, {'degree': (1, 5.01)}, verbose=0, allow_duplicate_points=True)    
    BO.maximize(init_points=10, n_iter=10) 
    params = max(BO.res, key=lambda x:x['target'])
    degree = params['params']['degree']
    
    model=WienerCascadeDecoder(degree) #Declare model
    model.fit(X_flat_train,y_train) #Fit model on training data
    y_test_predicted=model.predict(X_flat_test)   
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)
   
    coeffs, intercept = model.get_coefficients_intercepts(0) 

    print("R2 = {}".format(r2))
    
##################### XGBoost Decoder #########################
if m == 2:
    from decoders import XGBoostDecoder
    def xgb_evaluate(max_depth,num_round,eta):
        max_depth=int(max_depth) 
        num_round=int(num_round) 
        eta=float(eta) 
        model_xgb=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta) 
        model_xgb.fit(X_flat_train,y_train) 
        y_valid_predicted_xgb=model_xgb.predict(X_flat_valid) 
        return np.mean(get_R2(y_valid,y_valid_predicted_xgb)) 
    BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=0, allow_duplicate_points=True) 
    BO.maximize(init_points=5, n_iter=5) 
    params = max(BO.res, key=lambda x:x['target'])
    num_round = int(params['params']['num_round'])
    max_depth = int(params['params']['max_depth'])
    eta = params['params']['eta']
    
    model=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta) 
    coeffs,intercept = model.fit(X_flat_train,y_train) 
    y_test_predicted=model.predict(X_flat_test) 
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)
    
    print("R2 = {}".format(r2))

######################### SVR Decoder #########################
if m == 3:
    from decoders import SVRDecoder
    max_iter=4000
    def svr_evaluate(C):
        model_svr=SVRDecoder(C=C, max_iter=max_iter)
        model_svr.fit(X_flat_train,y_zscore_train) 
        y_valid_predicted_svr=model_svr.predict(X_flat_valid)
        return np.mean(get_R2(y_zscore_valid,y_valid_predicted_svr))
    BO = BayesianOptimization(svr_evaluate, {'C': (.5, 10)}, verbose=1, allow_duplicate_points=True)    
    BO.maximize(init_points=5, n_iter=5)
    params = max(BO.res, key=lambda x:x['target'])
    C = params['params']['C']

    model=SVRDecoder(C=C, max_iter=max_iter)
    coeffs,intercept = model.fit(X_flat_train,y_zscore_train) 
    y_test_predicted=model.predict(X_flat_test) 
    r2 = get_R2(y_zscore_test,y_test_predicted)
    rho = get_rho(y_zscore_test,y_test_predicted)
    
    print("R2 = {}".format(r2))

####################### DNN #######################
if m == 4:
    from decoders import DenseNNDecoder
    def dnn_evaluate(num_units,frac_dropout,n_epochs):
        num_units=int(num_units)
        frac_dropout=float(frac_dropout)
        n_epochs=int(n_epochs)
        model_dnn=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
        model_dnn.fit(X_flat_train,y_train)
        y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)
        return np.mean(get_R2(y_valid,y_valid_predicted_dnn))
    BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
    BO.maximize(init_points=10, n_iter=10)
    params = max(BO.res, key=lambda x:x['target'])
    frac_dropout=float(params['params']['frac_dropout'])
    n_epochs=int(params['params']['n_epochs'])
    num_units=int(params['params']['num_units'])

    model=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
    coeffs = model.fit(X_flat_train,y_train) 
    intercept = np.nan
    y_test_predicted=model.predict(X_flat_test) 
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)

    print("R2 = {}".format(r2))
    
########################## RNN ##############################3
if m == 5:
    from decoders import SimpleRNNDecoder
    def rnn_evaluate(num_units,frac_dropout,n_epochs):
        num_units=int(num_units)
        frac_dropout=float(frac_dropout)
        n_epochs=int(n_epochs)
        model_rnn=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
        model_rnn.fit(X_train,y_train)
        y_valid_predicted_rnn=model_rnn.predict(X_valid)
        return np.mean(get_R2(y_valid,y_valid_predicted_rnn))
    BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
    BO.maximize(init_points=10, n_iter=10)
    params = max(BO.res, key=lambda x:x['target'])
    frac_dropout=float(params['params']['frac_dropout'])
    n_epochs=int(params['params']['n_epochs'])
    num_units=int(params['params']['num_units'])

    model=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
    coeffs = model.fit(X_train,y_train)
    intercept = np.nan
    y_test_predicted=model.predict(X_test)
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)
    
    print("R2 = {}".format(r2))

######################### GRU Decoder ################################
if m == 6:
    from decoders import GRUDecoder
    def gru_evaluate(num_units,frac_dropout,n_epochs):
        num_units=int(num_units)
        frac_dropout=float(frac_dropout)
        n_epochs=int(n_epochs)
        model_gru=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
        model_gru.fit(X_train,y_train)
        y_valid_predicted_gru=model_gru.predict(X_valid)
        return np.mean(get_R2(y_valid,y_valid_predicted_gru))
    BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
    BO.maximize(init_points=3, n_iter=3)
    params = max(BO.res, key=lambda x:x['target'])
    frac_dropout=float(params['params']['frac_dropout'])
    n_epochs=int(params['params']['n_epochs'])
    num_units=int(params['params']['num_units'])
    
    model=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
    coeffs = model.fit(X_train,y_train)
    intercept = np.nan
    y_test_predicted=model.predict(X_test)
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)
    
    print("R2 = {}".format(r2))
    
######################### LSTM Decoder ############################
if m == 7:
    from decoders import LSTMDecoder
    def lstm_evaluate(num_units,frac_dropout,n_epochs):
        num_units=int(num_units)
        frac_dropout=float(frac_dropout)
        n_epochs=int(n_epochs)
        model_lstm=LSTMDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
        model_lstm.fit(X_train,y_train)
        y_valid_predicted_lstm=model_lstm.predict(X_valid)
        return np.mean(get_R2(y_valid,y_valid_predicted_lstm))
    BO = BayesianOptimization(lstm_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
    BO.maximize(init_points=3, n_iter=3)
    params = max(BO.res, key=lambda x:x['target'])
    frac_dropout=float(params['params']['frac_dropout'])
    n_epochs=int(params['params']['n_epochs'])
    num_units=int(params['params']['num_units'])

    model=LSTMDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
    coeffs = model.fit(X_train,y_train)
    intercept = np.nan
    y_test_predicted=model.predict(X_test)
    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)
    
    print("R2 = {}".format(r2))

######################### Kalman Filter ############################
if m == 8:
    from decoders import KalmanFilterDecoder
   
    [bins_before,bins_current,bins_after] = helpers.get_bins(bn)

    valid_lags=np.arange(-1*bins_before,bins_after)
    num_valid_lags=valid_lags.shape[0] 
    lag_results=np.empty(num_valid_lags) #Array to store validation R2 results for each lag
    C_results=np.empty(num_valid_lags) #Array to store the best hyperparameter for each lag

    def kf_evaluate_lag(lag,X_train,y_train,X_valid,y_valid):    
        if lag<0:
            y_train=y_train[-lag:,:]
            X_train=X_train[:lag,:]
            y_valid=y_valid[-lag:,:]
            X_valid=X_valid[:lag,:]
        if lag>0:
            y_train=y_train[0:-lag,:]
            X_train=X_train[lag:,:]
            y_valid=y_valid[0:-lag,:]
            X_valid=X_valid[lag:,:]
            
        def kf_evaluate(C):
            model=KalmanFilterDecoder(C=C) #Define model
            model.fit(X_train,y_train) #Fit model
            y_valid_predicted=model.predict(X_valid,y_valid) #Get validation set predictions
            
            return np.mean(get_R2(y_valid,y_valid_predicted)[2:4]) #Velocity is components 2 and 3
        
        #Do Bayesian optimization!
        BO = BayesianOptimization(kf_evaluate, {'C': (.5, 20)}, verbose=0) #Define Bayesian optimization, and set limits of hyperparameters
        BO.maximize(init_points=10, n_iter=10) #Set number of initial runs and subsequent tests, and do the optimization
        params = max(BO.res, key=lambda x:x['target'])
        C=float(params['params']['C'])

        model=KalmanFilterDecoder(C=C) #Define model
        model.fit(X_train,y_train) #Fit model
        y_valid_predicted=model.predict(X_valid,y_valid) #Get validation set predictions
        
        return [np.mean(get_R2(y_valid,y_valid_predicted)[2:4]), C] #Velocity is components 2 and 3

    for j in range(num_valid_lags):
        valid_lag=valid_lags[j] #Set what lag you're using
        #Run the wrapper function, and put the R2 value and corresponding C (hyperparameter) in arrays
        [lag_results[j],C_results[j]]=kf_evaluate_lag(valid_lag,X_train,y_train,X_valid,y_valid)

        print(lag_results[j])

    lag=valid_lags[np.argmax(lag_results)] #The lag
    C=C_results[np.argmax(lag_results)] #The hyperparameter C 

    #Re-align data to take lag into account
    if lag<0:
        y_train=y_train[-lag:,:]
        X_train=X_train[:lag,:]
        y_test=y_test[-lag:,:]
        X_test=X_test[:lag,:]
        y_valid=y_valid[-lag:,:]
        X_valid=X_valid[:lag,:]
    if lag>0:
        y_train=y_train[0:-lag,:]
        X_train=X_train[lag:,:]
        y_test=y_test[0:-lag,:]
        X_test=X_test[lag:,:]
        y_valid=y_valid[0:-lag,:]
        X_valid=X_valid[lag:,:]
    
    model=KalmanFilterDecoder(C=C) #Define model
    coeffs = model.fit(X_train,y_train) #Fit model
    intercept = np.nan
    y_test_predicted=model.predict(X_test,y_test) #Get test set predictions

    r2 = get_R2(y_test,y_test_predicted)
    rho = get_rho(y_test,y_test_predicted)

    print("R2 = {}".format(r2))

######################### WMP ############################
if m==9:
    from decoders import WeightedMovingAverage
    def wmp_evaluate(window_size):
        window_size=int(window_size)
        model = WeightedMovingAverage(window_size=window_size,n_outputs=y_train.shape[1])
        y_valid_predicted = model.predict(X_flat_valid)
        return np.mean(get_R2(y_valid, y_valid_predicted))

    BO = BayesianOptimization(wmp_evaluate, {'window_size': (1, 20.01)}, verbose=0, allow_duplicate_points=True)
    BO.maximize(init_points=10, n_iter=20)
    params = max(BO.res, key=lambda x: x['target'])
    window_size = int(params['params']['window_size'])

    model = WeightedMovingAverage(window_size=window_size,n_outputs=y_train.shape[1])
    y_test_predicted = model.predict(X_flat_test)
    r2 = get_R2(y_test, y_test_predicted)
    rho = get_rho(y_test, y_test_predicted)

    coeffs = np.nan
    intercept = np.nan

    print("R2 = {}".format(r2))
    print("rho = {}".format(rho))

#################### OMP ########################
if m==10:
    from decoders import OrthogonalMatchingPursuitDecoder

    def omp_evaluate(n_nonzero_coefs):
        n_nonzero_coefs = int(n_nonzero_coefs)
        model = OrthogonalMatchingPursuitDecoder(n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])
        model.fit(X_flat_train, y_train)
        y_valid_predicted = model.predict(X_flat_valid)
        return np.mean(get_R2(y_valid, y_valid_predicted))

    BO = BayesianOptimization(omp_evaluate, {'n_nonzero_coefs': (1, 20.01)}, verbose=0, allow_duplicate_points=True)
    BO.maximize(init_points=10, n_iter=20)
    params = max(BO.res, key=lambda x: x['target'])
    n_nonzero_coefs = int(params['params']['n_nonzero_coefs'])

    model = OrthogonalMatchingPursuitDecoder(n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])
    coeffs,intercept = model.fit(X_flat_train, y_train)
    y_test_predicted = model.predict(X_flat_test)
    r2 = get_R2(y_test, y_test_predicted)
    rho = get_rho(y_test, y_test_predicted)

    print("R2 = {}".format(r2))
    print("rho = {}".format(rho))

############### cascade OMP ####################
if m==11:
    from decoders import CascadeOrthogonalMatchingPursuitDecoder
    
    def cascade_omp_evaluate(n_stages,n_nonzero_coefs):
        n_stages = int(n_stages)
        n_nonzero_coefs = int(n_nonzero_coefs)
        model = CascadeOrthogonalMatchingPursuitDecoder(n_stages=n_stages, n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])  # You can adjust n_nonzero_coefs
        model.fit(X_flat_train, y_train)
        y_valid_predicted = model.predict(X_flat_valid)
        return np.mean(get_R2(y_valid, y_valid_predicted))

    BO = BayesianOptimization(cascade_omp_evaluate, {'n_stages': (1, 100.01), 'n_nonzero_coefs': (1,20.01)}, verbose=0, allow_duplicate_points=True)
    BO.maximize(init_points=5, n_iter=10)  # You can adjust the number of initial points and iterations
    params = max(BO.res, key=lambda x: x['target'])
    n_stages = int(params['params']['n_stages'])
    n_nonzero_coefs = int(params['params']['n_nonzero_coefs'])

    model = CascadeOrthogonalMatchingPursuitDecoder(n_stages=n_stages, n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])  # You can adjust n_nonzero_coefs
    coeffs,intercept = model.fit(X_flat_train, y_train)
    y_test_predicted = model.predict(X_flat_test)
    r2 = get_R2(y_test, y_test_predicted)
    rho = get_rho(y_test, y_test_predicted)

    print("R2 = {}".format(r2))
    print("rho = {}".format(rho))


#######################################################################################################################################
time_elapsed = time.time()-t1
print("time elapsed = {} mins".format(time_elapsed/60))

if o==0:
    output = 'position'
elif o==1:
    output = 'velocity'
elif o==2:
    output = 'acceleration'

result = [s,output,repeat,outer_fold,nm,nf,m,r2,rho,coeffs,intercept,time_elapsed]     

pfile = helpers.make_directory('Main/'+(jobname[:-6]),0)
if s==29:
    with open(cwd+pfile+'/fold{:0>2d}'.format(outer_fold)+'.pickle','wb') as p:
        pickle.dump([result,c_test,y_test,y_test_predicted],p)
else:
    with open(cwd+pfile+'/fold{:0>2d}'.format(outer_fold)+'.pickle','wb') as p:
        pickle.dump([result,c_test],p)
