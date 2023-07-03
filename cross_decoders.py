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
import helpers

line = np.loadtxt(params)[int(sys.argv[1])]
print(line)
s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)
#foldneuron_pairs = helpers.get_foldneuronPairs(int(sys.argv[1]))

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    outer_fold = int(sys.argv[3])
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    outer_fold = int(os.environ["SLURM_ARRAY_TASK_ID"])

trainTest_pairs = list(product(['contrast','speed'],range(3), range(3)))
tuples_to_remove = [('contrast',0,2), ('contrast',1,2), ('contrast',2,0), ('contrast',2,1), ('speed',0,2), ('speed',1,2), ('speed',2,0), ('speed',2,1)]
for item in tuples_to_remove:
    if item in trainTest_pairs:
        trainTest_pairs.remove(item)

results,times = [],[]
for repeat in range(num_repeats):
    ############ training ################
    #X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,neuron_inds = helpers.get_data_Xbrainarea(line,repeat,outer_fold,0,'FEF','MT')
    #X_trainN,X_testN,X_validN,X_flat_trainN,X_flat_testN,X_flat_validN,y_trainN,y_testN,y_validN,y_zscore_trainN,y_zscore_testN,y_zscore_validN,_ = helpers.get_data_Xbrainarea(line,repeat,outer_fold,1,'FEF','MT')

    for tt in trainTest_pairs:
        X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,neuron_inds = helpers.get_data_Xconditions(line,repeat,outer_fold,0,tt[0],tt[1],tt[2])

        init_points = 10
        n_iter = 10

        t1=time.time()
        
        ##################### Wiener Filter Decoder ############################
        if m == 0:
            from decoders import WienerFilterDecoder
            model=WienerFilterDecoder()
            model.fit(X_flat_train,y_train)

            y_train_predicted=model.predict(X_flat_train)   
            mean_r2_train = np.mean(get_R2(y_train,y_train_predicted))
            mean_rho_train = np.mean(get_rho(y_train,y_train_predicted))
            
            y_test_predicted=model.predict(X_flat_test)   
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))


      #      modelN=WienerFilterDecoder()
      #      modelN.fit(X_flat_trainN,y_trainN)
      #      y_test_predictedN=model.predict(X_flat_testN)   
      #      mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
      #      mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 train = {}".format(mean_r2_train))
            print("R2 test = {}".format(mean_r2))


     #       print("R2 (null) = {}".format(mean_r2N))

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
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))
            
            print("R2 = {}".format(mean_r2))

            # null hypothesis
            def wc_evaluateN(degree):
                model_wc=WienerCascadeDecoder(degree) 
                model_wc.fit(X_flat_trainN,y_trainN) 
                y_valid_predicted_wc=model_wc.predict(X_flat_validN) 
                return np.mean(get_R2(y_validN,y_valid_predicted_wc))
            BO = BayesianOptimization(wc_evaluateN, {'degree': (1, 5.01)}, verbose=0, allow_duplicate_points=True)    
            BO.maximize(init_points=10, n_iter=10) 
            params = max(BO.res, key=lambda x:x['target'])
            degree = params['params']['degree']
            
            modelN=WienerCascadeDecoder(degree) #Declare model
            modelN.fit(X_flat_trainN,y_trainN)
            y_test_predictedN=model.predict(X_flat_testN)   
            mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 (null) = {}".format(mean_r2N))
        
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
            model.fit(X_flat_train,y_train) 
            y_test_predicted=model.predict(X_flat_test) 
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))
            
            print("R2 = {}".format(mean_r2))

            # null hypothesis
            def xgb_evaluateN(max_depth,num_round,eta):
                max_depth=int(max_depth) 
                num_round=int(num_round) 
                eta=float(eta) 
                model_xgb=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta) 
                model_xgb.fit(X_flat_trainN,y_trainN) 
                y_valid_predicted_xgb=model_xgb.predict(X_flat_validN) 
                return np.mean(get_R2(y_validN,y_valid_predicted_xgb)) 
            BO = BayesianOptimization(xgb_evaluateN, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=0, allow_duplicate_points=True) 
            BO.maximize(init_points=5, n_iter=5)  
            params = max(BO.res, key=lambda x:x['target'])
            num_round = int(params['params']['num_round'])
            max_depth = int(params['params']['max_depth'])
            eta = params['params']['eta']
            
            modelN=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta) 
            modelN.fit(X_flat_trainN,y_trainN)
            y_test_predictedN=model.predict(X_flat_testN)   
            mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 (null) = {}".format(mean_r2N))

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
            model.fit(X_flat_train,y_zscore_train) 
            y_test_predicted=model.predict(X_flat_test) 
            mean_r2 = np.mean(get_R2(y_zscore_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_zscore_test,y_test_predicted))
            
            print("R2 = {}".format(mean_r2))

            # null hypothesis
            def svr_evaluateN(C):
                model_svr=SVRDecoder(C=C, max_iter=max_iter)
                model_svr.fit(X_flat_trainN,y_zscore_trainN) 
                y_valid_predicted_svr=model_svr.predict(X_flat_validN)
                return np.mean(get_R2(y_zscore_validN,y_valid_predicted_svr))
            BO = BayesianOptimization(svr_evaluateN, {'C': (.5, 10)}, verbose=1, allow_duplicate_points=True)    
            BO.maximize(init_points=5, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            C = params['params']['C']
        
            modelN=SVRDecoder(C=C, max_iter=max_iter)
            modelN.fit(X_flat_trainN,y_zscore_trainN) 
            y_test_predictedN=modelN.predict(X_flat_testN) 
            mean_r2N = np.mean(get_R2(y_zscore_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_zscore_testN,y_test_predictedN))
            
            print("R2 (null) = {}".format(mean_r2N))

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
            model.fit(X_flat_train,y_train) 
            y_test_predicted=model.predict(X_flat_test) 
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

            print("R2 = {}".format(mean_r2))
            '''
            # null hypothesis
            def dnn_evaluateN(num_units,frac_dropout,n_epochs):
                num_units=int(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_dnn=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
                model_dnn.fit(X_flat_trainN,y_trainN)
                y_valid_predicted_dnn=model_dnn.predict(X_flat_validN)
                return np.mean(get_R2(y_validN,y_valid_predicted_dnn))
            BO = BayesianOptimization(dnn_evaluateN, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
            BO.maximize(init_points=5, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            frac_dropout=float(params['params']['frac_dropout'])
            n_epochs=int(params['params']['n_epochs'])
            num_units=int(params['params']['num_units'])

            modelN=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
            modelN.fit(X_flat_trainN,y_trainN) 
            y_test_predictedN=modelN.predict(X_flat_testN) 
            mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 (null) = {}".format(mean_r2N))
            '''
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
            BO.maximize(init_points=5, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            frac_dropout=float(params['params']['frac_dropout'])
            n_epochs=int(params['params']['n_epochs'])
            num_units=int(params['params']['num_units'])

            model=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
            model.fit(X_train,y_train)
            y_test_predicted=model.predict(X_test)
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))
            
            print("R2 = {}".format(mean_r2))

            # null hypothesis
            def rnn_evaluateN(num_units,frac_dropout,n_epochs):
                num_units=int(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_rnn=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
                model_rnn.fit(X_trainN,y_trainN)
                y_valid_predicted_rnn=model_rnn.predict(X_validN)
                return np.mean(get_R2(y_validN,y_valid_predicted_rnn))
            BO = BayesianOptimization(rnn_evaluateN, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
            BO.maximize(init_points=5, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            frac_dropout=float(params['params']['frac_dropout'])
            n_epochs=int(params['params']['n_epochs'])
            num_units=int(params['params']['num_units'])

            modelN=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
            modelN.fit(X_trainN,y_trainN)
            y_test_predictedN=modelN.predict(X_testN)
            mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 (null) = {}".format(mean_r2N))


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
            model.fit(X_train,y_train)
            y_test_predicted=model.predict(X_test)
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))
            
            print("R2 = {}".format(mean_r2))
            def gru_evaluateN(num_units,frac_dropout,n_epochs):
                num_units=int(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_gru=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
                model_gru.fit(X_trainN,y_trainN)
                y_valid_predicted_gru=model_gru.predict(X_validN)
                return np.mean(get_R2(y_validN,y_valid_predicted_gru))
            BO = BayesianOptimization(gru_evaluateN, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
            BO.maximize(init_points=3, n_iter=3)
            params = max(BO.res, key=lambda x:x['target'])
            frac_dropout=float(params['params']['frac_dropout'])
            n_epochs=int(params['params']['n_epochs'])
            num_units=int(params['params']['num_units'])
            
            modelN=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
            modelN.fit(X_trainN,y_trainN)
            y_test_predictedN=modelN.predict(X_testN)
            mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 (null) = {}".format(mean_r2N))
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
            BO.maximize(init_points=10, n_iter=10)
            params = max(BO.res, key=lambda x:x['target'])
            frac_dropout=float(params['params']['frac_dropout'])
            n_epochs=int(params['params']['n_epochs'])
            num_units=int(params['params']['num_units'])

            model=LSTMDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
            model.fit(X_train,y_train)
            y_test_predicted=model.predict(X_test)
            mean_r2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))
            
            print("R2 = {}".format(mean_r2))

            def lstm_evaluateN(num_units,frac_dropout,n_epochs):
                num_units=int(num_units)
                frac_dropout=float(frac_dropout)
                n_epochs=int(n_epochs)
                model_lstm=LSTMDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
                model_lstm.fit(X_trainN,y_trainN)
                y_valid_predicted_lstm=model_lstm.predict(X_validN)
                return np.mean(get_R2(y_validN,y_valid_predicted_lstm))
            BO = BayesianOptimization(lstm_evaluateN, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'n_epochs': (2,21)}, allow_duplicate_points=True)
            BO.maximize(init_points=5, n_iter=5)
            params = max(BO.res, key=lambda x:x['target'])
            frac_dropout=float(params['params']['frac_dropout'])
            n_epochs=int(params['params']['n_epochs'])
            num_units=int(params['params']['num_units'])

            modelN=LSTMDecoder(units=num_units,dropout=frac_dropout,batch_size=128,num_epochs=n_epochs,workers=workers)
            modelN.fit(X_trainN,y_trainN)
            y_test_predictedN=modelN.predict(X_testN)
            mean_r2N = np.mean(get_R2(y_testN,y_test_predictedN))
            mean_rhoN = np.mean(get_rho(y_testN,y_test_predictedN))

            print("R2 (null) = {}".format(mean_r2N))

        time_elapsed = time.time()-t1

        if tt[1]==0:
            trainC = 'lo'
        elif tt[1]==1:
            trainC = 'hi'
        else:
            trainC = 'mix'
        if tt[2]==0:
            testC = 'lo'
        elif tt[2]==1:
            testC = 'hi'
        else:
            testC = 'mix'

        
        result = [s,outer_fold,repeat,nm,nf,m,tt[0],trainC,testC,mean_r2_train,mean_rho_train,mean_r2,mean_rho,neuron_inds,time_elapsed]
        results.append(result)

    #######################################################################################################################################
#    print("time elapsed = {} mins".format(time_elapsed/60))
#    result = [s,repeat,outer_fold,nm,nf,m,mean_r2,mean_rho,mean_r2N,mean_rhoN]     
#df = pd.DataFrame(results,columns=['sess','outer_fold','repeat','nMT','nFEF','model','condition','train','test','mean_R2','mean_rho','neuron_inds','time_elapsed'])
#print(df)

pfile = helpers.make_directory('cross_conditions/'+(jobname),0)
with open(cwd+pfile+'/fold{:0>2d}'.format(outer_fold)+'.pickle','wb') as p:
    pickle.dump(results,p)
     
    #with open(cwd+pfile+'/fold{:0>2d}-m{:0>1d}-eyetrace'.format(outer_fold,m)+'.pickle','wb') as p:
    #    pickle.dump([y_test,y_test_predicted],p)

