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
from itertools import product
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', 'Solver terminated early.*')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir
params = 'params.txt'

from metrics import get_R2
from metrics import get_rho
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
from psutil import cpu_count
import glob

import helpers
import model_evals

line = np.loadtxt(params)[int(sys.argv[1])]
print(line)
s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)
foldneuron_pairs = helpers.get_foldneuronPairs(int(sys.argv[1]))

contrasts,speeds,directions = helpers.get_sessConditions(s)

contrast_pairs = list(product(range(len(contrasts)), range(len(contrasts))))
speed_pairs = list(product(range(len(speeds)), range(len(speeds))))
#dir_pairs = list(product(range(len(directions)), range(len(directions))))

cond_pairs = contrast_pairs + speed_pairs
conditionPair = cond_pairs[int(sys.argv[2])] # if local
if int(sys.argv[2])<len(contrast_pairs):
    condType = 'contrast'
    n1 = contrasts[conditionPair[0]]
    n2 = contrasts[conditionPair[1]]
    name = '/c{:0>3d}-c{:0>3d}'.format(n1,n2)
elif (int(sys.argv[2])>=len(contrast_pairs) and int(sys.argv[2])<(len(contrast_pairs)+len(speed_pairs))):
    condType = 'speed'
    n1 = speeds[conditionPair[0]]
    n2 = speeds[conditionPair[1]]
    name = '/sp{:0>2d}-sp{:0>2d}'.format(n1,n2)
else:
    condType = 'direction'

#conditionPair = cond_pairs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

############ training ################
results,params_all,neurons_all,times_all = [],[],[],[]

new_line = line
for i in foldneuron_pairs:
    print(i)
    X_train0,X_flat_train0,y_train0,X_test,X_flat_test,y_test,X_shuf,X_flat_shuf,y_shuf,X_null,X_flat_null,y_null,neuron_inds = helpers.get_dataX(new_line,i[1],i[0],conditionPair,condType)

    inner_cv = KFold(n_splits=fi, random_state=None, shuffle=False)

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

            BO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.99)}, verbose=1,allow_duplicate_points=True)
            BO.maximize(init_points=10, n_iter=10) #Set number of initial runs and subsequent tests, and do the optimization
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([(round(((BO.res[key]['params']['degree'])*2))/2) for key in range(len(BO.res))]))).T)
            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','degree'])
                df_mn = df.groupby(['degree']).agg(['count','mean'])
                deg = df_mn['R2']['mean'].idxmax()
                max_params = deg

                _,X_flat_train0f,_,X_flat_testf,y_train0f,y_testf,_,_=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)

                # Run model w/ above hyperparameters
                model=WienerCascadeDecoder(deg) #Define model
                model.fit(X_flat_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                print(mean_R2)

        # XGBoost Decoder
        if m == 2:
            from decoders import XGBoostDecoder

            BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=1)
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

                model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta))
                model.fit(X_flat_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                print(mean_R2)

        # SVR Decoder
        if m == 3:
            from decoders import SVRDecoder

            BO = BayesianOptimization(svr_evaluate, {'C': (0.5, 10)}, verbose=1, allow_duplicate_points=True)
            BO.maximize(init_points=10, n_iter=10)
            params = max(BO.res, key=lambda x:x['target'])
            hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['C'],1) for key in range(len(BO.res))]))).T)

            if j==fi-1:
                df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','C'])
                df_mn = df.groupby(['C']).agg(['count','mean'])
                C = df_mn['R2']['mean'].idxmax()
                max_params = C
               
                _,X_flat_train0f,_,X_flat_testf,_,_,y_zscore_train0,y_zscore_test=helpers.normalize_trainTest(X_train0,X_flat_train0,X_test,X_flat_test,y_train0,y_test)

                model=SVRDecoder(C=C, max_iter=2000)
                model.fit(X_flat_train0f,y_zscore_train0) #Fit model
                y_train_predicted = model.predict(X_flat_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_flat_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_zscore_test,y_test_predicted))
                mean_rho = np.mean(get_rho(y_zscore_test,y_test_predicted))
                print(mean_R2)
                
        # DNN
        if m == 4:
            from decoders import DenseNNDecoder

            BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
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

                model=GRUDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                print(mean_R2)

        # LSTM Decoder
        if m == 7:
            from decoders import LSTMDecoder

            BO = BayesianOptimization(lstm_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)})
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

                model=LSTMDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),num_epochs=int(n_epochs))
                model.fit(X_train0f,y_train0f) #Fit model
                y_train_predicted = model.predict(X_train0f) #Validation set predictions
                y_test_predicted = model.predict(X_testf) #Validation set predictions

                mean_R2 = np.mean(get_R2(y_testf,y_test_predicted))
                mean_rho = np.mean(get_rho(y_testf,y_test_predicted))
                print(mean_R2)

    
    time_elapsed = time.time()-t1

    results.append([s,i[1],i[0],n1,n2,mean_R2,mean_R2_shuf,mean_R2_null,mean_rho])
    params_all.append(max_params)
    neurons_all.append(neuron_inds)
    times_all.append(time_elapsed)

pfile = helpers.make_directory('cross-conditions/'+jobname)
with open(cwd+pfile+name+'.pickle','wb') as p:
    pickle.dump([results,params_all,neurons_all,times_all],p)
#    pickle.dump([y_train0,y_test,y_train_predicted,y_test_predicted,mean_R2,mean_rho,time_elapsed,max_params,neuron_inds],p)

