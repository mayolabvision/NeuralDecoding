##################### BASIC EVALUATION OF ALL THE MODEL TYPES #########################
import numpy as np
from metrics import get_R2
from metrics import get_rho

######################## MODEL 0 - Wiener Filter Decoder ##########################
def wienerFilter(X_flat_train,X_flat_valid,X_flat_test,y_train,y_valid,y_test):
    from decoders import WienerFilterDecoder
    #Note - the Wiener Filter has no hyperparameters to fit, unlike all other methods

    model=WienerFilterDecoder()
    model.fit(X_flat_train,y_train)
    y_train_predicted=model.predict(X_flat_train)
    y_valid_predicted=model.predict(X_flat_valid)
    y_test_predicted=model.predict(X_flat_test)

    R2s = [np.mean(get_R2(y_train,y_train_predicted)), np.mean(get_R2(y_valid,y_valid_predicted)), np.mean(get_R2(y_test,y_test_predicted))] 
    rhos = [np.mean(get_rho(y_train,y_train_predicted)), np.mean(get_rho(y_valid,y_valid_predicted)), np.mean(get_rho(y_test,y_test_predicted))] 
    
    return y_train_predicted,y_valid_predicted,y_test_predicted,R2s,rhos,[]
    
###################### MODEL 1 - Wiener Cascade Decoder ###########################
def wienerCascade(X_flat_train,X_flat_valid,X_flat_test,y_train,y_valid,y_test):
    from decoders import WienerCascadeDecoder
    from bayes_opt import BayesianOptimization
    ### Get hyperparameters using Bayesian optimization based on validation set R2 values###

    #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
    #as a function of the hyperparameter we are fitting (here, degree)
    def wc_evaluate(degree):
        model=WienerCascadeDecoder(degree) #Define model
        model.fit(X_flat_train,y_train) #Fit model
        y_valid_predicted=model.predict(X_flat_valid) #Validation set predictions
        return np.mean(get_R2(y_valid,y_valid_predicted)) #R2 value of validation set (mean over x and y position/velocity)

    #Do bayesian optimization
    BO = BayesianOptimization(wc_evaluate, {'degree': (1, 5.01)}, verbose=0,allow_duplicate_points=True)
    BO.maximize(init_points=5, n_iter=5) #Set number of initial runs and subsequent tests, and do the optimization
    
    params = max(BO.res, key=lambda x:x['target'])
    degree = int(params['params']['degree'])

    ### Run model w/ above hyperparameters
    model=WienerCascadeDecoder(degree) #Declare model
    model.fit(X_flat_train,y_train) #Fit model on training data
    y_train_predicted=model.predict(X_flat_train)
    y_valid_predicted=model.predict(X_flat_valid)
    y_test_predicted=model.predict(X_flat_test)

    R2s = [np.mean(get_R2(y_train,y_train_predicted)), np.mean(get_R2(y_valid,y_valid_predicted)), np.mean(get_R2(y_test,y_test_predicted))] 
    rhos = [np.mean(get_rho(y_train,y_train_predicted)), np.mean(get_rho(y_valid,y_valid_predicted)), np.mean(get_rho(y_test,y_test_predicted))] 
    
    return y_train_predicted,y_valid_predicted,y_test_predicted,R2s,rhos,params['params']

# MODEL 2 - SVR Decoder
def SVR(X_flat_train,X_flat_valid,y_train,C,max_iter):
    y_train_std=np.nanstd(y_train,axis=0)
    y_zscore_train=y_train/y_train_std
    y_zscore_test=y_test/y_train_std
    y_zscore_valid=y_valid/y_train_std
   
    model=SVRDecoder(C=int(C), max_iter=int(max_iter))
    model.fit(X_flat_train,y_zscore_train)
    y_zscore_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_zscore_valid,y_zscore_valid_predicted)

    return y_valid_predicted,R2s


######################## MODEL 3 - XGBoost Decoder #########################
def XGBoost(X_flat_train,X_flat_valid,X_flat_test,y_train,y_valid,y_test):
    from decoders import XGBoostDecoder
    from bayes_opt import BayesianOptimization
    ### Get hyperparameters using Bayesian optimization based on validation set R2 values###
    
    #Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
    #as a function of the hyperparameter we are fitting (max_depth, num_round, eta)
    def xgb_evaluate(max_depth,num_round,eta):
        model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta)) #Define model
        model.fit(X_flat_train,y_train) #Fit model
        y_valid_predicted=model.predict(X_flat_valid) #Get validation set predictions
        return np.mean(get_R2(y_valid,y_valid_predicted)) #Return mean validation set R2

    #Do bayesian optimization
    BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 10.01), 'num_round': (100,700), 'eta': (0, 1)}) 
    BO.maximize(init_points=20, n_iter=20) 
    params = max(BO.res, key=lambda x:x['target'])
    max_depth = int(params['params']['max_depth'])
    num_round = int(params['params']['num_round'])
    eta = float(params['params']['eta'])

    # Run model w/ above hyperparameters
    model=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta) #Declare model w/ fit hyperparameters
    model.fit(X_flat_train,y_train) #Fit model

    y_train_predicted=model.predict(X_flat_train)
    y_valid_predicted=model.predict(X_flat_valid)
    y_test_predicted=model.predict(X_flat_test)

    R2s = [np.mean(get_R2(y_train,y_train_predicted)), np.mean(get_R2(y_valid,y_valid_predicted)), np.mean(get_R2(y_test,y_test_predicted))] 
    rhos = [np.mean(get_rho(y_train,y_train_predicted)), np.mean(get_rho(y_valid,y_valid_predicted)), np.mean(get_rho(y_test,y_test_predicted))] 
    
    return y_train_predicted,y_valid_predicted,y_test_predicted,R2s,rhos,params['params']

# MODEL 4 - Dense NN Decoder
def denseNN(X_flat_train,X_flat_valid,y_train,units,dropout,num_epochs):
    model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_flat_train,y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 5 - Simple RNN
def simpleRNN(X_train,X_valid,y_train,units,dropout,num_epochs):
    model=SimpleRNNDecoder(units=int(units),dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 6 - GRU Decoder
def GRU(X_train,X_valid,y_train,units,dropout,num_epochs):
    model=GRUDecoder(units=int(units),dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 7 - LSTM Decoder
def LSTM(X_train,X_valid,y_train,units,dropout,num_epochs):
    model=LSTMDecoder(units=int(units),dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

