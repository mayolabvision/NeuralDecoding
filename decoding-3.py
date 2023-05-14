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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir

from metrics import get_R2
from metrics import get_rho
from bayes_opt import BayesianOptimization

import argparse
import logging
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
from psutil import cpu_count
from sklearn.externals.joblib import Parallel, parallel_backend
from sklearn.externals.joblib import register_parallel_backend
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import cpu_count
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend


import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

import helpers
import decodingSetup

############## if on local computer ################
#num_cores = multiprocessing.cpu_count() 
#outer_fold = int(sys.argv[2])

############# if on cluster ########################
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", default="ipy_profile",
                 help="Name of IPython profile to use")
args = parser.parse_args()
profile = args.profile
logging.basicConfig(filename=os.path.join(cwd,profile+'.log'),
                    filemode='w',
                    level=logging.DEBUG)
logging.info("number of CPUs found: {0}".format(cpu_count()))
logging.info("args.profile: {0}".format(profile))

#prepare the engines
c = Client(profile=profile)
#The following command will make sure that each engine is running in
# the right working directory to access the custom function(s).
c[:].map(os.chdir, [cwd]*len(c))
logging.info("c.ids :{0}".format(str(c.ids)))
bview = c.load_balanced_view()
register_parallel_backend('ipyparallel',
                          lambda : IPythonParallelBackend(view=bview))


#num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
#outer_fold = int(os.environ["SLURM_ARRAY_TASK_ID"])

#print('number of cores = {}'.format(num_cores))
#print('outer fold = {}'.format(outer_fold))

########### model evaluations #############
def wc_evaluate(degree): #1
    model=WienerCascadeDecoder(degree) #Define model
    model.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted=model.predict(X_flat_valid) #Validation set predictions
    return np.mean(get_R2(y_valid,y_valid_predicted)) #R2 value of validation set (mean over x and y position/velocity)

def xgb_evaluate(max_depth,num_round,eta): #2
    model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta)) #Define model
    model.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted=model.predict(X_flat_valid) #Get validation set predictions
    return np.mean(get_R2(y_valid,y_valid_predicted)) #Return mean validation set R2

def svr_evaluate(C): #3
    model=SVRDecoder(C=C, max_iter=2000)
    model.fit(X_flat_train,y_zscore_train) #Note for SVR that we use z-scored y values
    y_valid_predicted=model.predict(X_flat_valid)
    return np.mean(get_R2(y_zscore_valid,y_valid_predicted))

def dnn_evaluate(num_units,frac_dropout,n_epochs): #4
    model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_flat_train,y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

def rnn_evaluate(num_units,frac_dropout,n_epochs):
    model=SimpleRNNDecoder(units=int(num_units),dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

def gru_evaluate(num_units,frac_dropout,n_epochs):
    model=GRUDecoder(units=int(num_units),dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

def lstm_evaluate(num_units,frac_dropout,n_epochs):
    model=LSTMDecoder(units=int(num_units),dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

############ training ################
X_train0, X_flat_train0, y_train0, X_test, X_flat_test, y_test, neurons_perRepeat = helpers.get_outerfold(int(sys.argv[1]),outer_fold)

s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)

inner_cv = KFold(n_splits=fi, random_state=None, shuffle=False)
print(inner_cv)

t1=time.time()
y_train_predicted,y_test_predicted,mean_R2,mean_rho,time_elapsed,max_params,neuron_inds = [],[],[],[],[],[],[]
def trainTest_perRepeat(r): 
    hp_tune = []
    for j, (train_index, valid_index) in enumerate(inner_cv.split(X_train0[r])):
        X_train = X_train0[r][train_index,:,:]
        X_flat_train = X_flat_train0[r][train_index,:]
        y_train = y_train0[r][train_index,:]

        X_valid = X_train0[r][valid_index,:,:]
        X_flat_valid = X_flat_train0[r][valid_index,:]
        y_valid = y_train0[r][valid_index,:]

        X_valid=(X_valid-np.nanmean(X_train,axis=0))/(np.nanstd(X_train,axis=0))
        X_train=(X_train-np.nanmean(X_train,axis=0))/(np.nanstd(X_train,axis=0))
        X_flat_valid=(X_flat_valid-np.nanmean(X_flat_train,axis=0))/(np.nanstd(X_flat_train,axis=0))
        X_flat_train=(X_flat_train-np.nanmean(X_flat_train,axis=0))/(np.nanstd(X_flat_train,axis=0))
        y_valid=y_valid-np.mean(y_train,axis=0)
        y_train=y_train-np.mean(y_train,axis=0) 
        y_zscore_valid=y_valid/(np.nanstd(y_train,axis=0))
        y_zscore_train=y_train/(np.nanstd(y_train,axis=0))


        # Wiener Filter Decoder
        if m == 0:
            from decoders import WienerFilterDecoder

            if j==fi-1:
                X_flat_testf=(X_flat_test[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                X_flat_train0f=(X_flat_train0[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 

                model=WienerFilterDecoder() #Define model
                model.fit(X_flat_train0f,y_train0f) #Fit model
                max_params = 0

                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                print(mean_R2)

        # Wiener Cascade Decoder
        if m == 1:
            from decoders import WienerCascadeDecoder

            BO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.99)}, verbose=1,allow_duplicate_points=True)
            BO.maximize(init_points=10, n_iter=10) #Set number of initial runs and subsequent tests, and do the optimization
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([(round(((BO.res[key]['params']['degree'])*2))/2) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','degree'])
                df_mn = df.groupby(['degree']).agg(['count','mean'])
                deg = df_mn['R2']['mean'].idxmax()
                max_params.append([deg])

                X_flat_testf=(X_flat_test[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                X_flat_train0f=(X_flat_train0[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 
                
                # Run model w/ above hyperparameters
                model=WienerCascadeDecoder(deg) #Define model
                model.fit(X_flat_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))

        # XGBoost Decoder
        if m == 2:
            from decoders import XGBoostDecoder
    
            BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=1)
            BO.maximize(init_points=3, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['max_depth']) for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['num_round']) for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['eta'],2) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','max_depth','num_round','eta'])
                df = df.sort_values(by=['R2'])
                df_max = df.groupby(['max_depth','num_round']).mean()
                df_max = df_max.reset_index()
                best_params = df_max.iloc[df_max['R2'].idxmax()]
                max_depth = best_params['max_depth']
                num_round = best_params['num_round']
                eta = best_params['eta']
                
                max_params.append([max_depth,num_round,eta])

                X_flat_testf=(X_flat_test[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                X_flat_train0f=(X_flat_train0[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 

                model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta))
                model.fit(X_flat_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))

        # SVR Decoder
        if m == 3:
            from decoders import SVRDecoder

            BO = BayesianOptimization(svr_evaluate, {'C': (0.5, 10)}, verbose=1, allow_duplicate_points=True)
            BO.maximize(init_points=5, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['C'],1) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','C'])
                df_mn = df.groupby(['C']).agg(['count','mean'])
                C = df_mn['R2']['mean'].idxmax()
                max_params.append([C])
               
                X_flat_testf=(X_flat_test[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                X_flat_train0f=(X_flat_train0[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 
                
                y_zscore_test=y_testf/(np.nanstd(y_train0f,axis=0))
                y_zscore_train0=y_train0f/(np.nanstd(y_train0f,axis=0))

                model=SVRDecoder(C=C, max_iter=2000)
                model.fit(X_flat_train0f,y_zscore_train0) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_zscore_test,y_test_predicted))
                mean_rho = np.mean(get_rho(y_zscore_test,y_test_predicted))
                
        # DNN
        if m == 4:
            from decoders import DenseNNDecoder

            BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
            BO.maximize(init_points=3, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['num_units']) for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['n_epochs']) for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['frac_dropout'],2) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','num_units','n_epochs','frac_dropout'])
                df = df.sort_values(by=['R2'])
                df_max = df.groupby(['num_units','n_epochs']).mean()
                df_max = df_max.reset_index()
                best_params = df_max.iloc[df_max['R2'].idxmax()]
                num_units = best_params['num_units']
                n_epochs = best_params['n_epochs']
                frac_dropout = best_params['frac_dropout']
                max_params.append([num_units,n_epochs,frac_dropout])

                X_flat_testf=(X_flat_test[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                X_flat_train0f=(X_flat_train0[r]-np.nanmean(X_flat_train0[r],axis=0))/(np.nanstd(X_flat_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 

                model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_flat_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))

        # RNN
        if m == 5:
            from decoders import SimpleRNNDecoder

            BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
            BO.maximize(init_points=3, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['num_units']) for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['n_epochs']) for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['frac_dropout'],2) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','num_units','n_epochs','frac_dropout'])
                df = df.sort_values(by=['R2'])
                df_max = df.groupby(['num_units','n_epochs']).mean()
                df_max = df_max.reset_index()
                best_params = df_max.iloc[df_max['R2'].idxmax()]
                num_units = best_params['num_units']
                n_epochs = best_params['n_epochs']
                frac_dropout = best_params['frac_dropout']
                max_params.append([num_units,n_epochs,frac_dropout])

                X_testf=(X_test[r]-np.nanmean(X_train0[r],axis=0))/(np.nanstd(X_train0[r],axis=0))
                X_train0f=(X_train0[r]-np.nanmean(X_train0[r],axis=0))/(np.nanstd(X_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 

                model=SimpleRNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))

        # GRU Decoder
        if m == 6:
            from decoders import GRUDecoder

            BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
            BO.maximize(init_points=3, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['num_units']) for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['n_epochs']) for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['frac_dropout'],2) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','num_units','n_epochs','frac_dropout'])
                df = df.sort_values(by=['R2'])
                df_max = df.groupby(['num_units','n_epochs']).mean()
                df_max = df_max.reset_index()
                best_params = df_max.iloc[df_max['R2'].idxmax()]
                num_units = best_params['num_units']
                n_epochs = best_params['n_epochs']
                frac_dropout = best_params['frac_dropout']
                max_params.append([num_units,n_epochs,frac_dropout])
                
                X_testf=(X_test[r]-np.nanmean(X_train0[r],axis=0))/(np.nanstd(X_train0[r],axis=0))
                X_train0f=(X_train0[r]-np.nanmean(X_train0[r],axis=0))/(np.nanstd(X_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 

                model=GRUDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))

        # LSTM Decoder
        if m == 7:
            from decoders import LSTMDecoder

            BO = BayesianOptimization(lstm_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
            BO.maximize(init_points=3, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['num_units']) for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['n_epochs']) for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['frac_dropout'],2) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','num_units','n_epochs','frac_dropout'])
                df = df.sort_values(by=['R2'])
                df_max = df.groupby(['num_units','n_epochs']).mean()
                df_max = df_max.reset_index()
                best_params = df_max.iloc[df_max['R2'].idxmax()]
                num_units = best_params['num_units']
                n_epochs = best_params['n_epochs']
                frac_dropout = best_params['frac_dropout']
                max_params.append([num_units,n_epochs,frac_dropout])
                
                X_testf=(X_test[r]-np.nanmean(X_train0[r],axis=0))/(np.nanstd(X_train0[r],axis=0))
                X_train0f=(X_train0[r]-np.nanmean(X_train0[r],axis=0))/(np.nanstd(X_train0[r],axis=0))
                y_testf=y_test[r]-np.mean(y_train0[r],axis=0)
                y_train0f=y_train0[r]-np.mean(y_train0[r],axis=0) 

                model=LSTMDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
    
    neuron_inds = neurons_perRepeat[r]
    return [y_train_predicted, y_test_predicted, mean_R2, mean_rho, max_params, neuron_inds] 

#results = Parallel(n_jobs=num_cores)(delayed(trainTest_perRepeat)(r) for r in range(num_repeats))
with parallel_backend('ipyparallel'):
    results = Parallel(n_jobs=len(c))(delayed(trainTest_perRepeat)(r)



time_elapsed = time.time()-t1
print("time elapsed: %.3f seconds" % time_elapsed)

y_train_predicted = [i[0] for i in results]
y_test_predicted = [i[1] for i in results]
mean_R2 = [i[2] for i in results]
mean_rho = [i[3] for i in results]
max_params = [i[4] for i in results]
neuron_inds = [i[5] for i in results]

pfile = helpers.make_directory(jobname)
with open(cwd+pfile+'/fold_'+str(outer_fold)+'.pickle','wb') as p:
    pickle.dump([y_train0,y_test,y_train_predicted,y_test_predicted,mean_R2,mean_rho,time_elapsed,max_params,neuron_inds],p)
