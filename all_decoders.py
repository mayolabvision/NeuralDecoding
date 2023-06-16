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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', 'Solver terminated early.*')

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir
params = 'params_allNeurons.txt'

from metrics import get_R2
from metrics import get_rho
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
from psutil import cpu_count
import helpers

line = np.loadtxt(params)[int(sys.argv[1])]
print(line)
s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)
foldneuron_pairs = helpers.get_foldneuronPairs(int(sys.argv[1]))

############## if on local computer ################
#num_cores = multiprocessing.cpu_count() 
#neuron_fold = foldneuron_pairs[int(sys.argv[2])]

############# if on cluster ########################
#num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
neuron_fold = foldneuron_pairs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

outer_fold = neuron_fold[0]
repeat = neuron_fold[1]

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
results,params_all,times_all = [],[],[]
X_train0,X_flat_train0,y_train0,X_test,X_flat_test,y_test,X_shuf,X_flat_shuf,y_shuf,X_null,X_flat_null,y_null,_,trainTest_index = helpers.get_data(line,repeat,outer_fold)

inner_cv = KFold(n_splits=fi, random_state=None, shuffle=False)
models = [5,6,7]
for m in models:
    print(m)
    t1=time.time()
    hp_tune = []
    for j, (train_index, valid_index) in enumerate(inner_cv.split(X_train0)):
        #print('\n')
        X_train = X_train0[train_index,:,:]
        X_flat_train = X_flat_train0[train_index,:]
        y_train = y_train0[train_index,:]

        X_valid = X_train0[valid_index,:,:]
        X_flat_valid = X_flat_train0[valid_index,:]
        y_valid = y_train0[valid_index,:]

        X_train,X_flat_train,X_valid,X_flat_valid,y_train,y_valid,y_zscore_train,y_zscore_valid=helpers.normalize_trainTest(X_train,X_flat_train,X_valid,X_flat_valid,y_train,y_valid)

        # Wiener Filter Decoder
        if m == 0:
            from decoders import WienerFilterDecoder
            if j==fi-1:
                _,X_flat_train0f,_,X_flat_testf,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,_,X_flat_shuff,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,_,X_flat_nullf,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=WienerFilterDecoder() #Define model
                model.fit(X_flat_train0f,y_train0f) #Fit model
                max_params = 0

                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_flat_shuff)
                y_null_predicted = model.predict(X_flat_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))

        # Wiener Cascade Decoder
        if m == 1:
            from decoders import WienerCascadeDecoder
            BO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.99)}, verbose=0, allow_duplicate_points=True)
            BO.maximize(init_points=3, n_iter=5) #Set number of initial runs and subsequent tests, and do the optimization
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([(round(((BO.res[key]['params']['degree'])*2))/2) for key in range(len(BO.res))]))).T)
            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','degree'])
                df_mn = df.groupby(['degree']).agg(['count','mean'])
                deg = df_mn['R2']['mean'].idxmax()
                max_params = deg

                _,X_flat_train0f,_,X_flat_testf,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,_,X_flat_shuff,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,_,X_flat_nullf,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=WienerCascadeDecoder(deg) #Define model
                model.fit(X_flat_train0f,y_train0f) #Fit model

                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_flat_shuff)
                y_null_predicted = model.predict(X_flat_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))
        
        # XGBoost Decoder
        if m == 2:
            from decoders import XGBoostDecoder

            BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=0)
            BO.maximize(init_points=2, n_iter=3)
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
                
                max_params = [max_depth,num_round,eta]

                _,X_flat_train0f,_,X_flat_testf,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,_,X_flat_shuff,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,_,X_flat_nullf,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta))
                model.fit(X_flat_train0f,y_train0f) #Fit model

                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_flat_shuff)
                y_null_predicted = model.predict(X_flat_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))
        # SVR Decoder
        if m == 3:
            from decoders import SVRDecoder

            BO = BayesianOptimization(svr_evaluate, {'C': (0.5, 10)}, verbose=0, allow_duplicate_points=True)
            BO.maximize(init_points=10, n_iter=10)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['C'],1) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','C'])
                df_mn = df.groupby(['C']).agg(['count','mean'])
                C = df_mn['R2']['mean'].idxmax()
                max_params = C
               
                _,X_flat_train0f,_,X_flat_testf,_,_,y_zscore_train0,y_zscore_test=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,_,X_flat_shuff,_,_,_,y_shuff=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,_,X_flat_nullf,_,_,_,y_nullf=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=SVRDecoder(C=C, max_iter=2000)
                model.fit(X_flat_train0f,y_zscore_train0) #Fit model

                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_flat_shuff)
                y_null_predicted = model.predict(X_flat_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))
                
        # DNN
        if m == 4:
            from decoders import DenseNNDecoder

            BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},verbose=0)
            BO.maximize(init_points=10, n_iter=10)
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
                max_params = [num_units,n_epochs,frac_dropout]

                _,X_flat_train0f,_,X_flat_testf,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,_,X_flat_shuff,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,_,X_flat_nullf,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_flat_train0f,y_train0f) #Fit model

                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_flat_shuff)
                y_null_predicted = model.predict(X_flat_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))
        # RNN
        if m == 5:
            from decoders import SimpleRNNDecoder

            BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},verbose=0)
            BO.maximize(init_points=5, n_iter=5)
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
                max_params = [num_units,n_epochs,frac_dropout]

                X_train0f,_,X_testf,_,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,X_shuff,_,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,X_nullf,_,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=SimpleRNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_test_predicted = model.predict(X_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_shuff)
                y_null_predicted = model.predict(X_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))

        # GRU Decoder
        if m == 6:
            from decoders import GRUDecoder

            BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},verbose=0)
            BO.maximize(init_points=5, n_iter=5)
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
                max_params = [num_units,n_epochs,frac_dropout]

                X_train0f,_,X_testf,_,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,X_shuff,_,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,X_nullf,_,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=GRUDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_test_predicted = model.predict(X_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_shuff)
                y_null_predicted = model.predict(X_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))

        # LSTM Decoder
        if m == 7:
            from decoders import LSTMDecoder

            BO = BayesianOptimization(lstm_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)},verbose=0)
            BO.maximize(init_points=5, n_iter=5)
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
                max_params = [num_units,n_epochs,frac_dropout]

                X_train0f,_,X_testf,_,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)
                _,_,X_shuff,_,_,y_shuff,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_shuf,X_flat_shuf,y_train0,y_shuf)
                _,_,X_nullf,_,_,y_nullf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_null,X_flat_null,y_train0,y_null)
                
                model=LSTMDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_test_predicted = model.predict(X_testf) #Validation set predictions
                y_shuf_predicted = model.predict(X_shuff)
                y_null_predicted = model.predict(X_nullf)

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                mean_R2_shuf = np.mean(get_R2(y_shuff,y_shuf_predicted))
                mean_rho_shuf = np.mean(get_rho(y_shuff,y_shuf_predicted))
                mean_R2_null = np.mean(get_R2(y_nullf,y_null_predicted))
                mean_rho_null = np.mean(get_rho(y_nullf,y_null_predicted))
                
                print("R2 = {}".format(mean_R2))
                print("R2 (shuffled) = {}".format(mean_R2_shuf))
                print("R2 (null) = {}".format(mean_R2_null))

    time_elapsed = time.time()-t1
    result = [s,repeat,outer_fold,nm,nf,m,mean_R2,mean_rho,mean_R2_shuf,mean_R2_null,mean_rho_null]     
    results.append([s,repeat,outer_fold,nm,nf,m,mean_R2,mean_rho,mean_R2_shuf,mean_R2_null,mean_rho_null])
    params_all.append(max_params)
    #neurons_all.append(neuron_inds)
    times_all.append(time_elapsed)

    pfile = helpers.make_directory('all_decoders/'+(jobname[:-6]))
    #with open(cwd+pfile+'/fold{:0>2d}'.format(outer_fold)+'.pickle','wb') as p:
    #    pickle.dump([results,params_all,times_all,trainTest_index],p)
    with open(cwd+pfile+'/fold{:0>2d}-m{:0>1d}'.format(outer_fold,m)+'.pickle','wb') as p:
        pickle.dump([result,max_params,time_elapsed,trainTest_index],p)

