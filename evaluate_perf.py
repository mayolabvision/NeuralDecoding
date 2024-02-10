import numpy as np, sys, pickle, time, pandas as pd, os.path, os, random
from scipy import io, stats
import warnings
from metrics import get_R2, get_rho
from bayes_opt import BayesianOptimization, UtilityFunction

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', 'Solver terminated early.*')

def evalMetrics(y_train, y_train_predicted, y_test, *predictions):
    metrics = []
    metrics.extend([get_R2(y_train,y_train_predicted),get_rho(y_train,y_train_predicted)])
    for y_pred in predictions:
        r2 = get_R2(y_test, y_pred)
        rho = get_rho(y_test, y_pred)
        metrics.extend([r2, rho])
    return tuple(metrics)

def baselineMetrics(y_test):
    print(y_test.shape)

def fitModel(model,Xtr,ytr,t1,Xte,yte,y_base_predicted):
    # model training
    model.fit(Xtr,ytr) 
    train_time = time.time()-t1
    y_train_predicted=model.predict(Xtr) # train accuracy 
    
    # model testing
    t2=time.time()
    y_test_predicted=model.predict(Xte)   
    test_time = (time.time()-t2) / yte.shape[0]
   
    # baseline comparisons
#    result = baselineMetrics(yte)

#print(result)
#    print(blah)

    Xte_shuf = Xte
    np.random.shuffle(Xte_shuf)
    y_shuf_predicted = model.predict(Xte_shuf) # shuffled accuracy
    y_mean_predicted = np.full_like(yte, fill_value=np.mean(ytr)) # mean coordinates

    return (y_train_predicted, y_test_predicted, y_shuf_predicted, y_mean_predicted, y_base_predicted) + evalMetrics(ytr, y_train_predicted, yte, y_test_predicted, y_shuf_predicted, y_mean_predicted, y_base_predicted) + (train_time, test_time)

####################################################################################################################################################################################
def run_model(m,o,verb,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,y_base,y_base_zscore):
    t1=time.time()
##################### WF ############################
    if m == 0:
        from decoders import WienerFilterDecoder
        model=WienerFilterDecoder()

        Xtr, Xte, ytr, yte, y_base_predicted = X_flat_train, X_flat_test, y_train, y_test, y_base
        prms = {'nan': 0}

        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)

##################### C-WF ###########################
    elif m == 1:
        from decoders import WienerCascadeDecoder
        
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_flat_train, X_flat_valid, X_flat_test, y_train, y_valid, y_test, y_base

        def wc_evaluate(degree):
            model_wc=WienerCascadeDecoder(degree) 
            model_wc.fit(Xtr,ytr) 
            y_valid_predicted_wc=model_wc.predict(Xva) 
            return np.mean(get_R2(yva,y_valid_predicted_wc))
        
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.01)}, verbose=verb, allow_duplicate_points=True)    
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers)
        params = max(BO.res, key=lambda x:x['target'])
        degree = params['params']['degree']
        prms = {'degree': degree}
        
        model=WienerCascadeDecoder(degree) #Declare model
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)

##################### XGBoost Decoder #########################
    elif m == 3:
        from decoders import XGBoostDecoder
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_flat_train, X_flat_valid, X_flat_test, y_train, y_valid, y_test, y_base
        
        def xgb_evaluate(max_depth,num_round,eta,subsample):
            model_xgb=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta), subsample=float(subsample), workers=workers) 
            model_xgb.fit(Xtr,ytr,Xva,yva) 
            y_valid_predicted_xgb=model_xgb.predict(Xva) 
            return np.mean(get_R2(yva,y_valid_predicted_xgb)) 

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,500), 'eta': (0.01, 0.3), 'subsample': (0.5,1.0)}, verbose=verb, allow_duplicate_points=True) 
        BO.maximize(init_points=10, n_iter=10, acquisition_function=acquisition_function)#, n_jobs=workers) 5,5

        params = max(BO.res, key=lambda x:x['target'])
        num_round = int(params['params']['num_round'])
        max_depth = int(params['params']['max_depth'])
        eta = float(params['params']['eta'])
        subsample = float(params['params']['subsample'])
        prms = {'num_round': num_round, 'max_depth': max_depth, 'eta': eta, 'subsample': subsample}

        model=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta, subsample=subsample, workers=workers) 
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)
        
######################## SVR Decoder #########################
    elif m == 4:
        from decoders import SVRDecoder
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_flat_train, X_flat_valid, X_flat_test, y_zscore_train, y_zscore_valid, y_zscore_test, y_base_zscore
        
        def svr_evaluate(C,kernel):
            kernel_mapping = {0: 'linear', 1: 'poly', 2: 'rbf'}
            kernel_str = kernel_mapping[int(kernel)]
            
            model_svr=SVRDecoder(C=C, kernel=kernel_str, max_iter=2000)
            model_svr.fit(Xtr,ytr) 
            y_valid_predicted_svr=model_svr.predict(Xva)
            return np.mean(get_R2(yva,y_valid_predicted_svr))
        
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(svr_evaluate, {'C': (0, 10), 'kernel': (0, 2.5)}, verbose=verb, allow_duplicate_points=True)    
        BO.maximize(init_points=1, n_iter=2,acquisition_function=acquisition_function)#, n_jobs=workers), 10,10

        params = max(BO.res, key=lambda x:x['target'])
        C = params['params']['C']
        kernel = int(params['params']['kernel'])
        prms = {'C': C, 'kernel': kernel}
        kernel_mapping = {0: 'linear', 1: 'poly', 2: 'rbf'}
        kernel_str = kernel_mapping[int(kernel)]

        model=SVRDecoder(C=C, kernel=kernel_str, max_iter=2000)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)
        
####################### DNN #######################
    elif m == 5:
        from decoders import DenseNNDecoder
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_flat_train, X_flat_valid, X_flat_test, y_train, y_valid, y_test, y_base
        
        def dnn_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            model_dnn=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_dnn.fit(Xtr,ytr)
            y_valid_predicted_dnn=model_dnn.predict(Xva)
            return np.mean(get_R2(yva,y_valid_predicted_dnn))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'batch_size': (32,256), 'n_epochs': (2,21)}, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10

        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)
        
########################## RNN ##############################3
    elif m == 6:
        from decoders import SimpleRNNDecoder
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_train, X_valid, X_test, y_train, y_valid, y_test, y_base
        
        def rnn_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            model_rnn=SimpleRNNDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_rnn.fit(Xtr,ytr)
            y_valid_predicted_rnn=model_rnn.predict(Xva)
            return np.mean(get_R2(yva,y_valid_predicted_rnn))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'batch_size': (32,256), 'n_epochs': (2,21)}, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10
        
        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)

######################### GRU Decoder ################################
    elif m == 7:
        from decoders import GRUDecoder
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_train, X_valid, X_test, y_train, y_valid, y_test, y_base
        
        def gru_evaluate(num_units,frac_dropout,batch_size,n_epochs):
            model_gru=GRUDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_gru.fit(Xtr,ytr)
            y_valid_predicted_gru=model_gru.predict(Xva)
            return np.mean(get_R2(yva,y_valid_predicted_gru))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(gru_evaluate, {'num_units': (50, 600), 'frac_dropout': (0,.5), 'batch_size': (32, 256),'n_epochs': (2,21)}, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10
        
        params = max(BO.res, key=lambda x:x['target'])
        frac_dropout=float(params['params']['frac_dropout'])
        batch_size=int(params['params']['batch_size'])
        n_epochs=int(params['params']['n_epochs'])
        num_units=int(params['params']['num_units'])
        prms = {'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model=GRUDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers,verbose=0)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)
        
######################### LSTM Decoder ############################
    elif m == 8:
        from decoders import LSTMDecoder
        Xtr, Xva, Xte, ytr, yva, yte, y_base_predicted = X_train, X_valid, X_test, y_train, y_valid, y_test, y_base

        def lstm_evaluate(units, dropout, batch_size, num_epochs):
            model_lstm=LSTMDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_lstm.fit(Xtr, ytr)
            y_valid_predicted_lstm = model_lstm.predict(Xva)
            return np.mean(get_R2(yva, y_valid_predicted_lstm))

        pbounds = {
            'units': (50, 600),
            'dropout': (0, 0.5),
            'batch_size': (32, 256),
            'num_epochs': (2, 21)
        }
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(lstm_evaluate, pbounds, verbose=verb, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10
        
        best_params = BO.max['params']
        units = int(best_params['units'])
        dropout = float(best_params['dropout'])
        batch_size = int(best_params['batch_size'])
        num_epochs = int(best_params['num_epochs'])
        prms = {'num_units': units, 'frac_dropout': dropout, 'batch_size': batch_size, 'n_epochs': num_epochs}

        model = LSTMDecoder(units=units, dropout=dropout, batch_size=batch_size, num_epochs=num_epochs, workers=workers, verbose=1)
        result = fitModel(model, Xtr, ytr, t1, Xte, yte, y_base_predicted)
    
    return result, prms 