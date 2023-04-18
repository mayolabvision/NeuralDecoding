import numpy as np
import sys
from scipy import io
from scipy import stats
import pickle
import time
import pandas as pd
import os.path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', 'Solver terminated early.*')

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir

from preprocessing_funcs import get_spikes_with_history
from matlab_funcs import mat_to_pickle
from metrics import get_R2
from metrics import get_rho
from decoders import WienerCascadeDecoder
from decoders import WienerFilterDecoder
from decoders import DenseNNDecoder
from decoders import SimpleRNNDecoder
from decoders import GRUDecoder
from decoders import LSTMDecoder
from decoders import XGBoostDecoder
from decoders import SVRDecoder
from sklearn import linear_model 
from sklearn.svm import SVR 
from sklearn.svm import SVC 
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import itertools

import helpers
import decodingSetup

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

def svr_evaluate(C,max_iter): #3
    model=SVRDecoder(C=int(C), max_iter=int(max_iter))
    model.fit(X_flat_train,y_zscore_train) #Note for SVR that we use z-scored y values
    y_valid_predicted=model.predict(X_flat_valid)
    return np.mean(get_R2(y_zscore_valid,y_valid_predicted))

def dnn_evaluate(num_units,frac_dropout,n_epochs): #4
    model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_flat_train,y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

def rnn_evaluate(num_units,frac_dropout,n_epochs): #5
    model=SimpleRNNDecoder(units=int(num_units),dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

def gru_evaluate(num_units,frac_dropout,n_epochs): #6
    model=GRUDecoder(units=int(num_units),dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

def lstm_evaluate(num_units,frac_dropout,n_epochs): #7
    model=LSTMDecoder(units=int(num_units),dropout=float(frac_dropout),num_epochs=int(n_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted))

############ training ################
X_train0, X_flat_train0, y_train0, X_test, X_flat_test, y_test = decodingSetup.get_dataParams(int(sys.argv[1]))

s,t,d,m,o,nm,nf,bn,fo,fi = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi)
pfile = helpers.make_directory(jobname)

outer_fold = int(sys.argv[2])
inner_cv = KFold(n_splits=fi, random_state=None, shuffle=False)

t1=time.time()
hp_tune = []
########### run through each inner fold #############3
for j, (train_index, valid_index) in enumerate(inner_cv.split(X_train0[outer_fold])):
    print(j)
    X_train = X_train0[outer_fold][train_index,:,:]
    X_flat_train = X_flat_train0[outer_fold][train_index,:]
    y_train = y_train0[outer_fold][train_index,:]

    X_valid = X_train0[outer_fold][valid_index,:,:]
    X_flat_valid = X_flat_train0[outer_fold][valid_index,:]
    y_valid = y_train0[outer_fold][valid_index,:]

    ##### PREPROCESS DATA #####
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
        if j==fi-1:
            X_flat_test=(X_flat_test[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            X_flat_train0=(X_flat_train0[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            model=WienerFilterDecoder() #Define model
            model.fit(X_flat_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_flat_train0) #Validation set predictions
            y_test_predicted=model.predict(X_flat_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

    # Wiener Cascade Decoder
    if m == 1:
        BO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.99)}, verbose=1,allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10) #Set number of initial runs and subsequent tests, and do the optimization
        params = max(BO.res, key=lambda x:x['target'])
        hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([(round(((BO.res[key]['params']['degree'])*2))/2) for key in range(len(BO.res))]))).T)

        if j==fi-1:
            df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','degree'])
            df_mn = df.groupby(['degree']).agg(['count','mean'])
            deg = df_mn['R2']['mean'].idxmax()
            
            X_flat_test=(X_flat_test[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            X_flat_train0=(X_flat_train0[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            # Run model w/ above hyperparameters
            model=WienerCascadeDecoder(deg) #Define model
            model.fit(X_flat_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_flat_train0) #Validation set predictions
            y_test_predicted=model.predict(X_flat_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))

            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

    # XGBoost Decoder
    if m == 2:
        BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=1)
        BO.maximize(init_points=10, n_iter=10)
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
            
            X_flat_test=(X_flat_test[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            X_flat_train0=(X_flat_train0[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta))
            model.fit(X_flat_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_flat_train0) #Validation set predictions
            y_test_predicted=model.predict(X_flat_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))


    # SVR Decoder
    if m == 3:
        BO = BayesianOptimization(svr_evaluate, {'C': (2, 6.99),'max_iter': (100,600.99)}, verbose=1, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10)
        params = max(BO.res, key=lambda x:x['target'])
        hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([int(BO.res[key]['params']['C']) for key in range(len(BO.res))]), np.array([int(BO.res[key]['params']['max_iter']) for key in range(len(BO.res))]))).T)

        if j==fi-1:
            df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','C','max_iter'])
            df = df.sort_values(by=['R2'])
            df_max = df.groupby(['C','max_iter']).mean()
            df_max = df_max.reset_index()
            best_params = df_max.iloc[df_max['R2'].idxmax()]
            C = best_params['C']
            max_iter = best_params['max_iter']
            
            X_flat_test=(X_flat_test[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            X_flat_train0=(X_flat_train0[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 
            y_zscore_test=y_test/(np.nanstd(y_train0,axis=0))
            y_zscore_train0=y_train0/(np.nanstd(y_train0,axis=0))

            model=SVRDecoder(C=C, max_iter=max_iter)
            model.fit(X_flat_train0,y_zscore_train0) #Fit model
            y_train_predicted=model.predict(X_flat_train0) #Validation set predictions
            y_test_predicted=model.predict(X_flat_test) #Validation set predictions

            print(np.mean(get_R2(y_zscore_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

    # DNN Decoder
    if m == 4:
        BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
        BO.maximize(init_points=20, n_iter=20)
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

            X_flat_test=(X_flat_test[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            X_flat_train0=(X_flat_train0[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
            model.fit(X_flat_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_flat_train0) #Validation set predictions
            y_test_predicted=model.predict(X_flat_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

    # RNN Decoder
    if m == 5:
        BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
        BO.maximize(init_points=20, n_iter=20)
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

            X_test=(X_test[outer_fold]-np.nanmean(X_train0[outer_fold],axis=0))/(np.nanstd(X_train0[outer_fold],axis=0))
            X_train0=(X_train0[outer_fold]-np.nanmean(X_train0[outer_fold],axis=0))/(np.nanstd(X_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            model=SimpleRNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
            model.fit(X_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_train0) #Validation set predictions
            y_test_predicted=model.predict(X_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

    # GRU Decoder
    if m == 6:
        BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
        BO.maximize(init_points=20, n_iter=20)
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

            X_test=(X_test[outer_fold]-np.nanmean(X_train0[outer_fold],axis=0))/(np.nanstd(X_train0[outer_fold],axis=0))
            X_train0=(X_train0[outer_fold]-np.nanmean(X_train0[outer_fold],axis=0))/(np.nanstd(X_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            model=GRUDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
            model.fit(X_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_train0) #Validation set predictions
            y_test_predicted=model.predict(X_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

    # LSTM Decoder
    if m == 7:
        BO = BayesianOptimization(lstm_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
        BO.maximize(init_points=20, n_iter=20)
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

            X_test=(X_test[outer_fold]-np.nanmean(X_train0[outer_fold],axis=0))/(np.nanstd(X_train0[outer_fold],axis=0))
            X_train0=(X_train0[outer_fold]-np.nanmean(X_train0[outer_fold],axis=0))/(np.nanstd(X_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 

            model=LSTMDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
            model.fit(X_train0,y_train0) #Fit model
            y_train_predicted=model.predict(X_train0) #Validation set predictions
            y_test_predicted=model.predict(X_test) #Validation set predictions

            print(np.mean(get_R2(y_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

time_elapsed=time.time()-t1 #How much time has passed
print("time elapsed: %.3f seconds" % time_elapsed)

with open(cwd+pfile+'/fold_'+str(outer_fold)+'.pickle','wb') as p:
    pickle.dump([y_train0,y_test,y_train_predicted,y_test_predicted,mean_R2,mean_rho,time_elapsed],p)

