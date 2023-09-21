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

from metrics import get_R2
from metrics import get_rho
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
from psutil import cpu_count
import helpers
import neuronsSample

def run_model(m,o,verb,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid):

##################### WF ############################
    if m == 0:
        from decoders import WienerFilterDecoder
        model=WienerFilterDecoder()
        
        t1=time.time()
        coeffs,intercept = model.fit(X_flat_train,y_train)
        train_time = time.time()-t1
        
        y_train_predicted=model.predict(X_flat_train)   
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)
        
        t2=time.time()
        y_test_predicted=model.predict(X_flat_test)   
        test_time = (time.time()-t1) / y_test.shape[0]
        
        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)

        coef_dict = {'coef': coeffs, 'intercept': intercept}
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        prms = {'nan': 0}

        print("R2 = {}".format(r2mn_test))

##################### C-WF ###########################
    if m == 1:
        from decoders import WienerCascadeDecoder
        t1=time.time()
        def wc_evaluate(degree):
            model_wc=WienerCascadeDecoder(degree) 
            model_wc.fit(X_flat_train,y_train) 
            y_valid_predicted_wc=model_wc.predict(X_flat_valid) 
            return np.mean(get_R2(y_valid,y_valid_predicted_wc))
        
        BO = BayesianOptimization(wc_evaluate, {'degree': (1, 10.01)}, verbose=verb, allow_duplicate_points=True)    
        BO.maximize(init_points=20, n_iter=20)#, n_jobs=workers)
        params = max(BO.res, key=lambda x:x['target'])
        degree = params['params']['degree']
        prms = {'degree': degree}
        
        model=WienerCascadeDecoder(degree) #Declare model
        model.fit(X_flat_train,y_train) #Fit model on training data
        train_time = time.time()-t1
        
        t2=time.time()
        y_test_predicted=model.predict(X_flat_test)   
        test_time = (time.time()-t1) / y_test.shape[0]
       
        y_train_predicted=model.predict(X_flat_train)
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
       
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        coeffs, intercept = model.get_coefficients_intercepts(0) 
        coef_dict = {'coef': coeffs, 'intercept': intercept}

        print("R2 = {}".format(r2mn_test))

######################### Kalman Filter ############################
    if m == 2:
        from decoders import KalmanFilterDecoder
        t1=time.time()
       
        [bins_before,bins_current,bins_after] = helpers.get_bins(bn)

        valid_lags=np.arange(-1*bins_before,bins_after)
        num_valid_lags=valid_lags.shape[0] 
        lag_results=np.empty(num_valid_lags) #Array to store validation R2 results for each lag
        C_results=np.empty(num_valid_lags) #Array to store the best hyperparameter for each lag

        max_out = [0,1,2,3,4,5]

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
                y_valid_predicted,_=model.predict(X_valid,y_valid) #Get validation set predictions
                
                return np.mean(get_R2(y_valid,y_valid_predicted)[max_out]) #Velocity is components 2 and 3
            
            #Do Bayesian optimization!
            BO = BayesianOptimization(kf_evaluate, {'C': (0.5, 20)}, verbose=verb, allow_duplicate_points=True) #Define Bayesian optimization, and set limits of hyperparameters
            BO.maximize(init_points=10, n_iter=10)#, n_jobs=workers)
            params = max(BO.res, key=lambda x:x['target'])
            C=float(params['params']['C'])

            model=KalmanFilterDecoder(C=C) #Define model
            model.fit(X_train,y_train) #Fit model
            y_valid_predicted,_=model.predict(X_valid,y_valid) #Get validation set predictions
            
            return [np.mean(get_R2(y_valid,y_valid_predicted)[max_out]), C] #Velocity is components 2 and 3

        for j in range(num_valid_lags):
            valid_lag=valid_lags[j] #Set what lag you're using
            #Run the wrapper function, and put the R2 value and corresponding C (hyperparameter) in arrays
            [lag_results[j],C_results[j]]=kf_evaluate_lag(valid_lag,X_train,y_train,X_valid,y_valid)

            print(lag_results[j])

        lag=valid_lags[np.argmax(lag_results)] #The lag
        C=C_results[np.argmax(lag_results)] #The hyperparameter C 

        prms = {'lag': lag, 'process_noise_scale': C}
        train_time = time.time()-t1

        #Re-align data to take lag into account
        t2=time.time()
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
        coeffs = model.fit(X_train,y_train) #Fit model, intercept = kalman gains
        y_test_predicted,gains = model.predict(X_test,y_test) #Get test set predictions

        #coef_dict = {'transition_matrix_A': coeffs[0], 'cov_transition_matrix_W': coeffs[1], 'measurement_matrix_H': coeffs[2], 'cov_measurement_matrix_Q': coeffs[3], 'kalman_gains': gains}
        coef_dict = {'test': np.nan}

        r2 = get_R2(y_test,y_test_predicted)
        rho = get_rho(y_test,y_test_predicted)
        test_time = time.time()-t1

        print("R2 = {}".format(r2))

##################### XGBoost Decoder #########################
    if m == 3:
        from decoders import XGBoostDecoder
        t1=time.time()
        def xgb_evaluate(max_depth,num_round,eta):
            max_depth=int(max_depth) 
            num_round=int(num_round) 
            eta=float(eta) 
            model_xgb=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta, workers=workers) 
            model_xgb.fit(X_flat_train,y_train) 
            y_valid_predicted_xgb=model_xgb.predict(X_flat_valid) 
            return np.mean(get_R2(y_valid,y_valid_predicted_xgb)) 

        BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}, verbose=verb, allow_duplicate_points=True) 
        BO.maximize(init_points=3, n_iter=5)#, n_jobs=workers)

        params = max(BO.res, key=lambda x:x['target'])
        num_round = int(params['params']['num_round'])
        max_depth = int(params['params']['max_depth'])
        eta = params['params']['eta']
       
        prms = {'num_round': num_round, 'max_depth': max_depth, 'eta': eta}

        model=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta, workers=workers) 
        model.fit(X_flat_train,y_train) 
        train_time = time.time()-t1

        t2=time.time()
        y_test_predicted=model.predict(X_flat_test) 
        test_time = (time.time()-t1) / y_test.shape[0]
        
        y_train_predicted=model.predict(X_flat_train) 
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
       
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        weights = model.get_feature_importance(importance_type='weight')
        coef_dict = {'weights': weights}

        print("R2 = {}".format(r2mn_test))

######################## SVR Decoder #########################
    if m == 4:
        from decoders import SVRDecoder
        t1=time.time()
        max_iter=2000
        def svr_evaluate(C):
            model_svr=SVRDecoder(C=C, max_iter=max_iter)
            model_svr.fit(X_flat_train,y_zscore_train) 
            y_valid_predicted_svr=model_svr.predict(X_flat_valid)
            return np.mean(get_R2(y_zscore_valid,y_valid_predicted_svr))

        BO = BayesianOptimization(svr_evaluate, {'C': (.5, 10)}, verbose=verb, allow_duplicate_points=True)    
        BO.maximize(init_points=5, n_iter=5)#, n_jobs=workers)

        params = max(BO.res, key=lambda x:x['target'])
        C = params['params']['C']
        prms = {'C': C}

        model=SVRDecoder(C=C, max_iter=max_iter)
        support_vects, coeffs = model.fit(X_flat_train,y_zscore_train) 
        train_time = time.time()-t1
        
        t2=time.time()
        y_test_predicted=model.predict(X_flat_test) 
        test_time = (time.time()-t1) / y_test.shape[0]

        y_train_predicted=model.predict(X_flat_train) 
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
       
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        margin_widths = model.get_margin_width
        coef_dict = {'support_vectors': support_vects, 'coefficients': coeffs, 'margin_widths': margin_widths}

        print("R2 = {}".format(r2mn_test))

####################### DNN #######################
    if m == 5:
        from decoders import DenseNNDecoder
        t1=time.time()
        def dnn_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            num_units=int(num_units)
            frac_dropout=float(frac_dropout)
            batch_size=int(batch_size)
            n_epochs=int(n_epochs)
            model_dnn=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
            model_dnn.fit(X_flat_train,y_train)
            y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)
            return np.mean(get_R2(y_valid,y_valid_predicted_dnn))

        BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'batch_size': (32,256), 'n_epochs': (2,21)}, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10)#, n_jobs=workers)

        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
        weights = model.fit(X_flat_train,y_train) 
        train_time = time.time()-t1

        t2=time.time()
        y_test_predicted=model.predict(X_flat_test) 
        test_time = (time.time()-t1) / y_test.shape[0]

        y_train_predicted=model.predict(X_flat_train) 
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
        
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        coef_dict = {'weights': weights}  

        print("R2 = {}".format(r2mn_test))
        
########################## RNN ##############################3
    if m == 6:
        from decoders import SimpleRNNDecoder
        t1=time.time()
        def rnn_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            num_units=int(num_units)
            frac_dropout=float(frac_dropout)
            batch_size=int(batch_size)
            n_epochs=int(n_epochs)
            model_rnn=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
            model_rnn.fit(X_train,y_train)
            y_valid_predicted_rnn=model_rnn.predict(X_valid)
            return np.mean(get_R2(y_valid,y_valid_predicted_rnn))

        BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'batch_size': (32,256), 'n_epochs': (2,21)}, allow_duplicate_points=True)
        BO.maximize(init_points=3, n_iter=3)#, n_jobs=workers)
        
        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])

        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
        weights = model.fit(X_train,y_train)
        train_time = time.time()-t1
        
        t2=time.time()
        y_test_predicted=model.predict(X_test)
        test_time = (time.time()-t1) / y_test.shape[0]

        y_train_predicted=model.predict(X_train) 
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
        
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        coef_dict = {'weights': weights}  

        print("R2 = {}".format(r2mn_test))

######################### GRU Decoder ################################
    if m == 7:
        from decoders import GRUDecoder
        t1=time.time()
        def gru_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            num_units=int(num_units)
            frac_dropout=float(frac_dropout)
            batch_size = int(batch_size)
            n_epochs=int(n_epochs)
            model_gru=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers,verbose=0)
            model_gru.fit(X_train,y_train)
            y_valid_predicted_gru=model_gru.predict(X_valid)
            return np.mean(get_R2(y_valid,y_valid_predicted_gru))

        BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'batch_size': (32, 256),'n_epochs': (2,21)}, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10)#, n_jobs=workers)
        
        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
       
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}
        train_time = time.time()-t1

        model=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers,verbose=0)
        weights = model.fit(X_train,y_train)
        
        t2=time.time()
        y_test_predicted=model.predict(X_test)
        test_time = (time.time()-t1) / y_test.shape[0]

        y_train_predicted=model.predict(X_train) 
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
        
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        coef_dict = {'weights': weights}  

        print("R2 = {}".format(r2mn_test))
        
######################### LSTM Decoder ############################
    if m == 8:
        t1=time.time()
        from decoders import LSTMDecoder

# Define the evaluation function
        def lstm_evaluate(units, dropout, batch_size, num_epochs):
            units = int(units)
            dropout = float(dropout)
            batch_size = int(batch_size)
            num_epochs = int(num_epochs)

            model = LSTMDecoder(units=units, dropout=dropout, batch_size=batch_size, num_epochs=num_epochs, verbose=0)
            model.fit(X_train, y_train)
            y_valid_predicted = model.predict(X_valid)
            return np.mean(get_R2(y_valid, y_valid_predicted))

        pbounds = {
            'units': (50, 600),
            'dropout': (0, 0.5),
            'batch_size': (32, 256),
            'num_epochs': (2, 21)
        }

        BO = BayesianOptimization(lstm_evaluate, pbounds, verbose=verb, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10)#, n_jobs=workers)
        
        best_params = BO.max['params']
        units = int(best_params['units'])
        dropout = float(best_params['dropout'])
        batch_size = int(best_params['batch_size'])
        num_epochs = int(best_params['num_epochs'])
        
        prms = {'num_units': units, 'frac_dropout': dropout, 'batch_size': batch_size, 'n_epochs': num_epochs}
        train_time = time.time()-t1

        model = LSTMDecoder(units=units, dropout=dropout, batch_size=batch_size, num_epochs=num_epochs, verbose=1)
        weights = model.fit(X_train, y_train)
        
        t2=time.time()
        y_test_predicted = model.predict(X_test)
        test_time = (time.time()-t1) / y_test.shape[0]

        y_train_predicted=model.predict(X_train) 
        r2mn_train,r2_train = get_R2(y_train,y_train_predicted)
        rhomn_train,rho_train = get_rho(y_train,y_train_predicted)

        r2mn_test,r2_test = get_R2(y_test,y_test_predicted)
        rhomn_test,rho_test = get_rho(y_test,y_test_predicted)
        
        eval_full = {'r2_train': r2_train, 'rho_train': rho_train, 'r2_test': r2_test, 'rho_test': rho_test}
        coef_dict = {'weights': weights}  

        print("R2 = {}".format(r2mn_test))
    
    return r2mn_train,rhomn_train,r2mn_test,rhomn_test,eval_full,coef_dict,prms,y_test,y_test_predicted,train_time,test_time


######################################################## OTHER MODELS #######################################################3
'''
##################### LR ############################
    if m == 0:
        from decoders import LinearDecoder
        model=LinearDecoder()
        coeffs,intercept = model.fit(X_flat_train,y_train)
        y_test_predicted=model.predict(X_flat_test)   
        r2 = get_R2(y_test,y_test_predicted)
        rho = get_rho(y_test,y_test_predicted)

        prms = [np.nan]

        print("R2 = {}".format(r2))

##################### C-LR ###########################
    if m == 1:
        from decoders import LinearCascadeDecoder
        def wc_evaluate(degree):
            model_wc=LinearCascadeDecoder(degree) 
            model_wc.fit(X_flat_train,y_train) 
            y_valid_predicted_wc=model_wc.predict(X_flat_valid) 
            return np.mean(get_R2(y_valid,y_valid_predicted_wc))
        BO = BayesianOptimization(wc_evaluate, {'degree': (1, 5.01)}, verbose=verb, allow_duplicate_points=True)    
        BO.maximize(init_points=10, n_iter=10) 
        params = max(BO.res, key=lambda x:x['target'])
        degree = params['params']['degree']

        prms = [degree]
        
        model=LinearCascadeDecoder(degree) #Declare model
        model.fit(X_flat_train,y_train) #Fit model on training data
        y_test_predicted=model.predict(X_flat_test)   
        r2 = get_R2(y_test,y_test_predicted)
        rho = get_rho(y_test,y_test_predicted)
       
        coeffs, intercept = model.get_coefficients_intercepts(0) 

        print("R2 = {}".format(r2))


######################### Extended Kalman Filter (E-KF) ############################
    if m == 5:
        from decoders import ExtendedKalmanFilterDecoder  # Import the ExtendedKalmanFilterRegression class

        [bins_before, bins_current, bins_after] = helpers.get_bins(bn)

        valid_lags = np.arange(-1 * bins_before, bins_after)
        num_valid_lags = valid_lags.shape[0]
        lag_results = np.empty(num_valid_lags)  # Array to store validation R2 results for each lag
        C_results = np.empty(num_valid_lags)  # Array to store the best hyperparameter for each lag

        if o == 0:
            max_out = [0, 1]
        elif o == 1:
            max_out = [2, 3]
        else:
            max_out = [4, 5]

        def ekf_evaluate_lag(lag, X_train, y_train, X_valid, y_valid):
            if lag < 0:
                y_train = y_train[-lag:, :]
                X_train = X_train[:lag, :]
                y_valid = y_valid[-lag:, :]
                X_valid = X_valid[:lag, :]
            if lag > 0:
                y_train = y_train[0:-lag, :]
                X_train = X_train[lag:, :]
                y_valid = y_valid[0:-lag, :]
                X_valid = X_valid[lag:, :]

            def ekf_evaluate(C):
                model = ExtendedKalmanFilterDecoder(C=C)  # Define EKF model
                model.fit(X_train, y_train)  # Fit model
                y_valid_predicted = model.predict(X_valid, y_valid)  # Get validation set predictions

                return np.mean(get_R2(y_valid, y_valid_predicted)[max_out])  # Velocity is components 2 and 3

            # Do Bayesian optimization
            BO = BayesianOptimization(ekf_evaluate, {'C': (.01, 100)}, verbose=verb, allow_duplicate_points=True) #Define Bayesian optimization, and set limits of hyperparameters
            BO.maximize(init_points=10, n_iter=10)  # Set number of initial runs and subsequent tests, and perform optimization
            params = max(BO.res, key=lambda x: x['target'])
            C = float(params['params']['C'])

            model = ExtendedKalmanFilterDecoder(C=C)  # Define EKF model
            model.fit(X_train, y_train)  # Fit model
            y_valid_predicted = model.predict(X_valid, y_valid)  # Get validation set predictions

            return [np.mean(get_R2(y_valid, y_valid_predicted)[max_out]), C]  # Velocity is components 2 and 3

        for j in range(num_valid_lags):
            valid_lag = valid_lags[j]  # Set what lag you're using
            # Run the wrapper function, and put the R2 value and corresponding C (hyperparameter) in arrays
            [lag_results[j], C_results[j]] = ekf_evaluate_lag(valid_lag, X_train, y_train, X_valid, y_valid)

            print(lag_results[j])

        lag = valid_lags[np.argmax(lag_results)]  # The lag
        C = C_results[np.argmax(lag_results)]  # The hyperparameter C

        prms = [lag, C]

# Re-align data to take lag into account
        if lag < 0:
            y_train = y_train[-lag:, :]
            X_train = X_train[:lag, :]
            y_test = y_test[-lag:, :]
            X_test = X_test[:lag, :]
            y_valid = y_valid[-lag:, :]
            X_valid = X_valid[:lag, :]
        if lag > 0:
            y_train = y_train[0:-lag, :]
            X_train = X_train[lag:, :]
            y_test = y_test[0:-lag, :]
            X_test = X_test[lag:, :]
            y_valid = y_valid[0:-lag, :]
            X_valid = X_valid[lag:, :]

        model = ExtendedKalmanFilterDecoder(C=C)  # Define EKF model
        coeffs = model.fit(X_train, y_train)  # Fit model
        intercept = np.nan
        y_test_predicted = model.predict(X_test, y_test)  # Get test set predictions

        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        print("R2 = {}".format(r2))

######################### Unscented Kalman Filter (U-KF) ############################
    if m == 6: 
        from decoders import UnscentedKalmanFilterDecoder

# Define bins_before, bins_current, bins_after
        [bins_before, bins_current, bins_after] = helpers.get_bins(bn)

        valid_lags = np.arange(-1 * bins_before, bins_after)
        num_valid_lags = valid_lags.shape[0]
        lag_results = np.empty(num_valid_lags)  # Array to store validation R2 results for each lag
        C_results = np.empty(num_valid_lags)  # Array to store the best hyperparameter for each lag

        if o == 0:
            max_out = [0, 1]
        elif o == 1:
            max_out = [2, 3]
        else:
            max_out = [4, 5]

        def ukf_evaluate_lag(lag, X_train, y_train, X_valid, y_valid):
            if lag < 0:
                y_train = y_train[-lag:, :]
                X_train = X_train[:lag, :]
                y_valid = y_valid[-lag:, :]
                X_valid = X_valid[:lag, :]
            if lag > 0:
                y_train = y_train[0:-lag, :]
                X_train = X_train[lag:, :]
                y_valid = y_valid[0:-lag, :]
                X_valid = X_valid[lag:, :]

            def ukf_evaluate(C):
                model = UnscentedKalmanFilterDecoder(C=C)  # Define UKF model
                model.fit(X_train, y_train)  # Fit UKF model
                y_valid_predicted = model.predict(X_valid, y_valid)  # Get validation set predictions

                return np.mean(get_R2(y_valid, y_valid_predicted)[max_out])  # Velocity is components 2 and 3

            # Perform Bayesian optimization
            BO = BayesianOptimization(ukf_evaluate, {'C': (.01, 100)}, verbose=verb, allow_duplicate_points=True)  # Define Bayesian optimization, and set limits of hyperparameters
            BO.maximize(init_points=30, n_iter=50)  # Set number of initial runs and subsequent tests, and perform optimization
            params = max(BO.res, key=lambda x: x['target'])
            C = float(params['params']['C'])

            model = UnscentedKalmanFilterDecoder(C=C)  # Define UKF model
            model.fit(X_train, y_train)  # Fit UKF model
            y_valid_predicted = model.predict(X_valid, y_valid)  # Get validation set predictions

            return [np.mean(get_R2(y_valid, y_valid_predicted)[max_out]), C]  # Velocity is components 2 and 3

        for j in range(num_valid_lags):
            valid_lag = valid_lags[j]  # Set the lag you're using
            # Run the wrapper function, and store the R2 value and corresponding C (hyperparameter) in arrays
            [lag_results[j], C_results[j]] = ukf_evaluate_lag(valid_lag, X_train, y_train, X_valid, y_valid)

            print(lag_results[j])

        lag = valid_lags[np.argmax(lag_results)]  # The lag
        C = C_results[np.argmax(lag_results)]  # The hyperparameter C

        prms = [lag, C]

# Re-align data to take lag into account
        if lag < 0:
            y_train = y_train[-lag:, :]
            X_train = X_train[:lag, :]
            y_test = y_test[-lag:, :]
            X_test = X_test[:lag, :]
            y_valid = y_valid[-lag:, :]
            X_valid = X_valid[:lag, :]
        if lag > 0:
            y_train = y_train[0:-lag, :]
            X_train = X_train[lag:, :]
            y_test = y_test[0:-lag, :]
            X_test = X_test[lag:, :]
            y_valid = y_valid[0:-lag, :]
            X_valid = X_valid[lag:, :]

        model = UnscentedKalmanFilterDecoder(C=C)  # Define UKF model
        coeffs = model.fit(X_train, y_train)  # Fit UKF model
        intercept = np.nan
        y_test_predicted = model.predict(X_test, y_test)  # Get test set predictions

        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        print("R2 = {}".format(r2))

##################### WMP ############################
    if m==6:
        from decoders import WeightedMovingAverage
        def wmp_evaluate(window_size):
            window_size=int(window_size)
            model = WeightedMovingAverage(window_size=window_size,n_outputs=y_train.shape[1])
            y_valid_predicted = model.predict(X_flat_valid)
            return np.mean(get_R2(y_valid, y_valid_predicted))

        BO = BayesianOptimization(wmp_evaluate, {'window_size': (1, 20.01)}, verbose=verb, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=20)
        params = max(BO.res, key=lambda x: x['target'])
        window_size = int(params['params']['window_size'])

        prms = [window_size]

        model = WeightedMovingAverage(window_size=window_size,n_outputs=y_train.shape[1])
        y_test_predicted = model.predict(X_flat_test)
        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        coeffs = np.nan
        intercept = np.nan

        print("R2 = {}".format(r2))

######################### Weighted Recursive Least Squares (WRLS) ############################
    if m == 3:
        from decoders import WeightedRecursiveLeastSquares

        def wrls_evaluate(alpha):
            model = WeightedRecursiveLeastSquares(alpha=alpha)
            model.fit(X_flat_train, y_train)
            y_valid_predicted = model.predict(X_flat_valid)
            return np.mean(get_R2(y_valid, y_valid_predicted))

        BO = BayesianOptimization(wrls_evaluate, {'alpha': (0.01, 0.99)}, verbose=verb, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10)
        params = max(BO.res, key=lambda x: x['target'])
        alpha = params['params']['alpha']

        prms = [alpha]

        model = WeightedRecursiveLeastSquares(alpha=alpha)
        model.fit(X_flat_train, y_train)
        y_test_predicted = model.predict(X_flat_test)
        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        print("R2 = {}".format(r2))
#################### Orthogonal Matching Pursuit (OMP) Decoder  ########################
    if m==7:
        from decoders import OrthogonalMatchingPursuitDecoder

        def omp_evaluate(n_nonzero_coefs):
            n_nonzero_coefs = int(n_nonzero_coefs)
            model = OrthogonalMatchingPursuitDecoder(n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])
            model.fit(X_flat_train, y_train)
            y_valid_predicted = model.predict(X_flat_valid)
            return np.mean(get_R2(y_valid, y_valid_predicted))

        BO = BayesianOptimization(omp_evaluate, {'n_nonzero_coefs': (1, 50.01)}, verbose=verb, allow_duplicate_points=True)
        BO.maximize(init_points=5, n_iter=5)
        params = max(BO.res, key=lambda x: x['target'])
        n_nonzero_coefs = int(params['params']['n_nonzero_coefs'])

        n_nonzero_coefs = 50
        prms = [n_nonzero_coefs]

        model = OrthogonalMatchingPursuitDecoder(n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])
        coeffs,intercept = model.fit(X_flat_train, y_train)
        y_test_predicted = model.predict(X_flat_test)
        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        print("R2 = {}".format(r2))
        print("rho = {}".format(rho))

#################### Cascade Orthogonal Matching Pursuit (C-OMP) Decoder  ########################
    if m==8:
        from decoders import CascadeOrthogonalMatchingPursuitDecoder
        
        n_nonzero_coefs = 50
        def cascade_omp_evaluate(n_stages):
            n_stages = int(n_stages)
            model = CascadeOrthogonalMatchingPursuitDecoder(n_stages=n_stages, n_nonzero_coefs=int(n_nonzero_coefs), n_outputs=y_train.shape[1])  # You can adjust n_nonzero_coefs
            model.fit(X_flat_train, y_train)
            y_valid_predicted = model.predict(X_flat_valid)
            return np.mean(get_R2(y_valid, y_valid_predicted))

        BO = BayesianOptimization(cascade_omp_evaluate, {'n_stages': (5, 100.01)}, verbose=verb, allow_duplicate_points=True)
        BO.maximize(init_points=5, n_iter=10)  # You can adjust the number of initial points and iterations
        params = max(BO.res, key=lambda x: x['target'])
        n_stages = int(params['params']['n_stages'])
        #n_nonzero_coefs = int(params['params']['n_nonzero_coefs'])

        prms = [n_stages,n_nonzero_coefs]

        model = CascadeOrthogonalMatchingPursuitDecoder(n_stages=n_stages, n_nonzero_coefs=n_nonzero_coefs, n_outputs=y_train.shape[1])  # You can adjust n_nonzero_coefs
        coeffs,intercept = model.fit(X_flat_train, y_train)
        y_test_predicted = model.predict(X_flat_test)
        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        print("R2 = {}".format(r2))
        print("rho = {}".format(rho))

#################### Gaussian Process Regression (GPR) Decoder  ########################
    if m==9:
        from decoders import GaussianProcessRegressionDecoder

        def gpr_evaluate(kernel_length_scale):
            model = GaussianProcessRegressionDecoder(kernel_length_scale=kernel_length_scale)
            model.fit(X_flat_train, y_train)
            y_valid_predicted = model.predict(X_flat_valid)
            return np.mean(get_R2(y_valid, y_valid_predicted))

        gp_params = {"kernel_length_scale": (0.1, 10.0)}

        BO = BayesianOptimization(gpr_evaluate, gp_params, verbose=verb, random_state=42)
        BO.maximize(init_points=1, n_iter=3)
        params = max(BO.res, key=lambda x: x['target'])
        kernel_length_scale = params['params']['kernel_length_scale']
    
        prms = [kernel_length_scale]

        kernel_length_scale = 1.0

        model = GaussianProcessRegressionDecoder(kernel_length_scale=kernel_length_scale)
        coeffs = model.fit(X_flat_train, y_train)
        intercept = np.nan

        y_test_predicted = model.predict(X_flat_test)

        r2 = get_R2(y_test, y_test_predicted)
        rho = get_rho(y_test, y_test_predicted)

        print("R2 = {}".format(r2))
'''
