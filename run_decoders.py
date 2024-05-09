import numpy as np, sys, pickle, time, pandas as pd, os.path, os, random
from scipy import io, stats
import warnings
from metrics import get_R2, get_rho, get_RMSE
from bayes_opt import BayesianOptimization, UtilityFunction

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', 'Solver terminated early.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fitModel(model,Xtr,ytr,t1,Xte,yte):
    # model training
    model.fit(Xtr,ytr) 
    train_time = time.time()-t1
    y_train_predicted=model.predict(Xtr) # train accuracy 
    
    # model testing
    t2=time.time()
    y_test_predicted=model.predict(Xte)   
    test_time = (time.time()-t2) / yte.shape[0]
   
    return y_train_predicted, y_test_predicted, train_time, test_time

def get_metric(yTrue,yPred,em=0):
    if em==0:
        out = get_R2(yTrue,yPred)
    elif em==1:
        out = get_rho(yTrue,yPred)
    elif em==2:
        out = -1*get_RMSE(yTrue,yPred) # making this negative, since bayes_opt wants to maximize during hyperparameterization

    return out

####################################################################################################################################################################################
def run_model(m,o,verb,workers,comp,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid):
    t1=time.time()
    if comp==0:
        init_points = 1
        n_iter     = 1
    else:
        init_points = 10
        n_iter     = 10

##################### WF ############################
    if m == 0:
        from decoders import WienerFilterDecoder
        model=WienerFilterDecoder()

        Xtr, Xte, ytr, yte = X_flat_train, X_flat_test, y_train, y_test
        prms = {'nan': 0}

        result = fitModel(model, Xtr, ytr, t1, Xte, yte)

##################### C-WF ###########################
    elif m == 1:
        from decoders import WienerCascadeDecoder
        
        Xtr, Xva, Xte, ytr, yva, yte = X_flat_train, X_flat_valid, X_flat_test, y_train, y_valid, y_test

        def wc_evaluate(degree):
            model_wc=WienerCascadeDecoder(degree) 
            model_wc.fit(Xtr,ytr) 
            y_valid_predicted_wc=model_wc.predict(Xva) 
            return np.mean(get_metric(yva,y_valid_predicted_wc))
        
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(wc_evaluate, {'degree': (1, 10.01)}, verbose=verb, allow_duplicate_points=True,random_state=m)    
        BO.maximize(init_points=init_points*2, n_iter=n_iter*2,acquisition_function=acquisition_function)#, n_jobs=workers)
        params = max(BO.res, key=lambda x:x['target'])
        degree = params['params']['degree']
        prms = {'degree': degree}
        
        model=WienerCascadeDecoder(degree) #Declare model
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)

##################### XGBoost Decoder #########################
    elif m == 2:
        from decoders import XGBoostDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_flat_train, X_flat_valid, X_flat_test, y_train, y_valid, y_test
        
        def xgb_evaluate(max_depth,num_round,eta,subsample):
            model_xgb=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta), subsample=float(subsample), workers=workers) 
            model_xgb.fit(Xtr,ytr) 
            y_valid_predicted_xgb=model_xgb.predict(Xva) 
            return np.mean(get_metric(yva,y_valid_predicted_xgb))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 8.01), 'num_round': (200,1000), 'eta': (0.01, 0.3), 'subsample': (0.25,0.75)}, verbose=verb, allow_duplicate_points=True,random_state=m) 
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)#, n_jobs=workers) 5,5

        params = max(BO.res, key=lambda x:x['target'])
        num_round = int(params['params']['num_round'])
        max_depth = int(params['params']['max_depth'])
        eta = float(params['params']['eta'])
        subsample = float(params['params']['subsample'])
        prms = {'num_round': num_round, 'max_depth': max_depth, 'eta': eta, 'subsample': subsample}

        model=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta, subsample=subsample, workers=workers) 
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)
        
######################## SVR Decoder #########################
    elif m == 3:
        from decoders import SVRDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_flat_train, X_flat_valid, X_flat_test, y_zscore_train, y_zscore_valid, y_zscore_test
        
        def svr_evaluate(C,kernel):
            kernel_mapping = {0: 'linear', 1: 'poly', 2: 'rbf'}
            kernel_str = kernel_mapping[int(kernel)]
            
            model_svr=SVRDecoder(C=C, kernel=kernel_str, max_iter=2000)
            model_svr.fit(Xtr,ytr) 
            y_valid_predicted_svr=model_svr.predict(Xva)
            return np.mean(get_metric(yva,y_valid_predicted_svr))
            
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(svr_evaluate, {'C': (0.5, 10), 'kernel': (0, 2.5)}, verbose=verb, allow_duplicate_points=True,random_state=m)    
        BO.maximize(init_points=int(init_points/2), n_iter=int(n_iter/2), acquisition_function=acquisition_function)#, n_jobs=workers) 5,5

        params = max(BO.res, key=lambda x:x['target'])
        C = params['params']['C']
        kernel = int(params['params']['kernel'])
        prms = {'C': C, 'kernel': kernel}
        kernel_mapping = {0: 'linear', 1: 'poly', 2: 'rbf'}
        kernel_str = kernel_mapping[int(kernel)]

        model=SVRDecoder(C=C, kernel=kernel_str, max_iter=2000)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)
        
####################### DNN #######################
    elif m == 4:
        from decoders import DenseNNDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_flat_train, X_flat_valid, X_flat_test, y_train, y_valid, y_test
        
        def dnn_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            model_dnn=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_dnn.fit(Xtr,ytr)
            y_valid_predicted_dnn=model_dnn.predict(Xva)
            return np.mean(get_metric(yva,y_valid_predicted_dnn))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0.1,.5), 'batch_size': (32,128), 'n_epochs': (2,21)}, allow_duplicate_points=True,random_state=m)
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)#, n_jobs=workers) 5,5

        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)
        
########################## RNN ##############################3
    elif m == 5:
        from decoders import SimpleRNNDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_train, X_valid, X_test, y_train, y_valid, y_test
        
        def rnn_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            model_rnn=SimpleRNNDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_rnn.fit(Xtr,ytr)
            y_valid_predicted_rnn=model_rnn.predict(Xva)
            return np.mean(get_metric(yva,y_valid_predicted_rnn))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 300), 'frac_dropout': (0.1,.5), 'batch_size': (32,128), 'n_epochs': (2,15)}, allow_duplicate_points=True,random_state=m)
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)#, n_jobs=workers) 5,5
        
        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)

######################### GRU Decoder ################################
    elif m == 6:
        from decoders import GRUDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_train, X_valid, X_test, y_train, y_valid, y_test
        
        def gru_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            model_gru=GRUDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_gru.fit(Xtr,ytr)
            y_valid_predicted_gru=model_gru.predict(Xva)
            return np.mean(get_metric(yva,y_valid_predicted_gru))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 300), 'frac_dropout': (0.1,.5), 'batch_size': (32, 128),'n_epochs': (2,15)}, allow_duplicate_points=True,random_state=m)
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)#, n_jobs=workers) 5,5
        
        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers,verbose=0)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)
        
######################### LSTM Decoder ############################
    elif m == 7:
        from decoders import LSTMDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_train, X_valid, X_test, y_train, y_valid, y_test

        def lstm_evaluate(num_units, frac_dropout, batch_size, n_epochs):
            model_lstm=LSTMDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_lstm.fit(Xtr, ytr)
            y_valid_predicted_lstm = model_lstm.predict(Xva)
            return np.mean(get_metric(yva,y_valid_predicted_lstm))

        pbounds = {
            'num_units': (50, 300),
            'frac_dropout': (0.1, 0.5),
            'batch_size': (32, 128),
            'n_epochs': (2, 15)
        }
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(lstm_evaluate, pbounds, verbose=verb, allow_duplicate_points=True,random_state=m)
        BO.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acquisition_function)#, n_jobs=workers) 5,5
        
        best_params = BO.max['params']
        num_units = int(best_params['num_units'])
        frac_dropout = float(best_params['frac_dropout'])
        batch_size = int(best_params['batch_size'])
        n_epochs = int(best_params['n_epochs'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model = LSTMDecoder(units=num_units, dropout=frac_dropout, batch_size=batch_size, num_epochs=n_epochs, workers=workers, verbose=1)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte)
    
    return result, prms 
