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

#folder = '/jet/home/knoneman/NeuralDecoding/'
#folder = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/'
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
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

import helpers
import decodingSetup

########### model evaluations #############
def wc_evaluate(degree):
    model=WienerCascadeDecoder(degree) #Define model
    model.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted=model.predict(X_flat_valid) #Validation set predictions
    return np.mean(get_R2(y_valid,y_valid_predicted)) #R2 value of validation set (mean over x and y position/velocity)

def xgb_evaluate(max_depth,num_round,eta):
    model=XGBoostDecoder(max_depth=int(max_depth), num_round=int(num_round), eta=float(eta)) #Define model
    model.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted=model.predict(X_flat_valid) #Get validation set predictions
    return np.mean(get_R2(y_valid,y_valid_predicted)) #Return mean validation set R2

def svr_evaluate(C):
    model=SVRDecoder(C=C, max_iter=2000)
    model.fit(X_flat_train,y_zscore_train) #Note for SVR that we use z-scored y values
    y_valid_predicted=model.predict(X_flat_valid)
    return np.mean(get_R2(y_zscore_valid,y_valid_predicted))

X_train0, X_flat_train0, y_train0, X_test, X_flat_test, y_test = decodingSetup.get_dataParams(int(sys.argv[1]))
print(X_train0)

s,t,d,m,o,nm,nf,bn,fo,fi = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi)
pfile = helpers.make_directory(jobname)

outer_fold = int(sys.argv[2])

inner_cv = KFold(n_splits=fi, random_state=None, shuffle=False)

t1=time.time()

hp_tune = []
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
        BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 6.99), 'num_round': (100,600.99), 'eta': (0.01, 8)}, verbose=1,allow_duplicate_points=True)
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
        BO = BayesianOptimization(svr_evaluate, {'C': (2, 6.99)}, verbose=1, allow_duplicate_points=True)
        BO.maximize(init_points=10, n_iter=10)
        params = max(BO.res, key=lambda x:x['target'])
        hp_tune.append(np.vstack((np.array([BO.res[key]['target'] for key in range(len(BO.res))]),np.array([round(BO.res[key]['params']['C'],1) for key in range(len(BO.res))]))).T)

        if j==fi-1:
            df = pd.DataFrame(np.vstack(np.array(hp_tune)), columns = ['R2','C'])
            df_mn = df.groupby(['C']).agg(['count','mean'])
            C = df_mn['R2']['mean'].idxmax()
            
            X_flat_test=(X_flat_test[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            X_flat_train0=(X_flat_train0[outer_fold]-np.nanmean(X_flat_train0[outer_fold],axis=0))/(np.nanstd(X_flat_train0[outer_fold],axis=0))
            y_test=y_test[outer_fold]-np.mean(y_train0[outer_fold],axis=0)
            y_train0=y_train0[outer_fold]-np.mean(y_train0[outer_fold],axis=0) 
            y_zscore_test=y_test/(np.nanstd(y_train0,axis=0))
            y_zscore_train0=y_train0/(np.nanstd(y_train0,axis=0))

            model=SVRDecoder(C=C, max_iter=2000)
            model.fit(X_flat_train0,y_zscore_train0) #Fit model
            y_train_predicted=model.predict(X_flat_train0) #Validation set predictions
            y_test_predicted=model.predict(X_flat_test) #Validation set predictions

            print(np.mean(get_R2(y_zscore_test,y_test_predicted)))
            
            mean_R2 = np.mean(get_R2(y_test,y_test_predicted))
            mean_rho = np.mean(get_rho(y_test,y_test_predicted))

time_elapsed=time.time()-t1 #How much time has passed
print("time elapsed: %.3f seconds" % time_elapsed)



with open(cwd+pfile+'/fold_'+str(outer_fold)+'.pickle','wb') as p:
    pickle.dump([y_train0,y_test,y_train_predicted,y_test_predicted,mean_R2,mean_rho,time_elapsed],p)

'''
        # SVR Decoder
        if m == 2:
            from evaluateModels import wienerCascade
            [y_train_predicted,y_valid_predicted,y_test_predicted,R2s,rhos,BOparams] = wienerCascade(X_flat_train,X_flat_valid,X_flat_test,y_train,y_valid,y_test)

        # XGBoost Decoder
        if m == 3:
            from evaluateModels import XGBoost
            [y_train_predicted,y_valid_predicted,y_test_predicted,R2s,rhos,BOparams] = XGBoost(X_flat_train,X_flat_valid,X_flat_test,y_train,y_valid,y_test)

    y_train_pred_all.append(y_train_predicted)
    y_valid_pred_all.append(y_valid_predicted)
    y_test_pred_all.append(y_test_predicted)
    R2s_all.append(R2s)
    rhos_all.append(rhos)
    
    time_elapsed=time.time()-t1 #How much time has passed
    print ("\n") #Line break after each fold   
    
with open(folder+pfile,'wb') as p:
    pickle.dump([y_train_all,y_valid_all,y_test_all,y_train_pred_all,y_valid_pred_all,y_test_pred_all,np.array(R2s_all),np.array(rhos_all),time_elapsed],p)

print("time elapsed: %.3f seconds" % time_elapsed)
print(np.mean(R2s_all,axis=0))
# ### Wiener Cascade Decoder

# In[ ]:
'''
'''
#Declare model
model_wc=WienerCascadeDecoder(degree=3)

#Fit model
model_wc.fit(X_flat_train,y_train)

#Get predictions
y_valid_predicted_wc=model_wc.predict(X_flat_valid)

#Get metric of fit
R2s_wc=get_R2(y_valid,y_valid_predicted_wc)
print('R2s:', R2s_wc)


# In[ ]:


def wc_evaluate(degree):
    model_wc=WienerCascadeDecoder(degree) #Define model
    model_wc.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted_wc=model_wc.predict(X_flat_valid) #Validation set predictions
    return np.mean(get_R2(y_valid,y_valid_predicted_wc)) #R2 value of validation set (mean over x and y position/velocity)


# In[ ]:


from bayes_opt import BayesianOptimization
#Define Bayesian optimization, and set limits of hyperparameters 
#Here, we set the limit of "degree" to be [1, 6.99], so we test degrees 1,2,3,4,5,6
wcBO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.99)}, verbose=0)
#Set number of initial runs (init_points) and subsequent tests (n_iter), and do the optimization
#kappa is a parameter that sets exploration vs exploitation in the algorithm
#We set kappa=10 (greater than the default) so there is more exploration when there are more hyperparameters
wcBO.maximize(init_points=5, n_iter=5, kappa=10)


# In[ ]:


wcBO.res
wcBO.res[2]


# In[ ]:


#Declare model
model_wc=WienerCascadeDecoder(degree=16)

#Fit model
model_wc.fit(X_flat_train,y_train)

#Get predictions
y_valid_predicted_wc=model_wc.predict(X_flat_valid)

#Get metric of fit
R2s_wc=get_R2(y_valid,y_valid_predicted_wc)
print('R2s:', R2s_wc)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_wc_optimized_pa29.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_valid_predicted_wc': y_valid_predicted_wc,'R2s_wc':R2s_wc, 'wcBO':wcBO})


# ### XGBoost Decoder

# In[ ]:


#Declare model
model_xgb=XGBoostDecoder(max_depth=3,num_round=200,eta=0.3,gpu=-1) 

#Fit model
model_xgb.fit(X_flat_train, y_train)

#Get predictions
y_valid_predicted_xgb=model_xgb.predict(X_flat_valid)

#Get metric of fit
R2s_xgb=get_R2(y_valid,y_valid_predicted_xgb)
print('R2s:', R2s_xgb)


# In[ ]:


def xgb_evaluate(max_depth,num_round,eta):
    #The parameters need to be in the correct format for the decoder, so we do that below
    max_depth=int(max_depth) 
    num_round=int(num_round) 
    eta=float(eta) 
    #Define model
    model_xgb=XGBoostDecoder(max_depth=max_depth, num_round=num_round, eta=eta) 
    model_xgb.fit(X_flat_train,y_train) #Fit model
    y_valid_predicted_xgb=model_xgb.predict(X_flat_valid) #Get validation set predictions
    return np.mean(get_R2(y_valid,y_valid_predicted_xgb)) #Return mean validation set R2


# In[ ]:


#Do bayesian optimization, and set limits of hyperparameters
xgbBO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 6.99), 'num_round': (100,600.99), 'eta': (0.01, 0.8)},verbose=0) #Define Bayesian optimization, and set limits of hyperparameters    
#Set number of initial runs and subsequent tests, and do the optimization. Also, we set kappa=10 (greater than the default) so there is more exploration when there are more hyperparameters
xgbBO.maximize(init_points=10, n_iter=10, kappa=10)


# In[ ]:


xgbBO.res
xgbBO.res[10]


# In[ ]:


#Declare model
model_xgb=XGBoostDecoder(max_depth=7,num_round=600,eta=0.23,gpu=-1) 

#Fit model
model_xgb.fit(X_flat_train, y_train)

#Get predictions
y_valid_predicted_xgb=model_xgb.predict(X_flat_valid)

#Get metric of fit
R2s_xgb=get_R2(y_valid,y_valid_predicted_xgb)
print('R2s:', R2s_xgb)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_xgb_optimized_pa29.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_valid_predicted_xgb': y_valid_predicted_xgb,'R2s_xgb':R2s_xgb,'xgbBO':xgbBO})


# ### SVR Decoder

# In[ ]:


#The SVR works much better when the y values are normalized, so we first z-score the y values
#They have previously been zero-centered, so we will just divide by the stdev (of the training set)
y_train_std=np.nanstd(y_train,axis=0)
y_zscore_train=y_train/y_train_std
y_zscore_test=y_test/y_train_std
y_zscore_valid=y_valid/y_train_std

#Declare model
model_svr=SVRDecoder(C=5, max_iter=4000)

#Fit model
model_svr.fit(X_flat_train,y_zscore_train)

#Get predictions
y_zscore_valid_predicted_svr=model_svr.predict(X_flat_valid)

#Get metric of fit
R2s_svr=get_R2(y_zscore_valid,y_zscore_valid_predicted_svr)
print('R2s:', R2s_svr)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_svr_.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_zscore_valid_predicted_svr': y_zscore_valid_predicted_svr,'R2s_svr':R2s_svr}) #,'svrBO':svrBO})


# In[ ]:


def svr_evaluate(C,max_iter):
    #The parameters need to be in the correct format for the decoder, so we do that below
    C=int(C) 
    max_iter=int(max_iter) 
    #Define model
    model_svr=SVRDecoder(C=C, max_iter=max_iter) 
    model_svr.fit(X_flat_train,y_zscore_train) #Fit model
    y_zscore_valid_predicted_svr=model_svr.predict(X_flat_valid) #Get validation set predictions
    return np.mean(get_R2(y_zscore_valid,y_zscore_valid_predicted_svr)) #Return mean validation set R2


# In[ ]:


#Do bayesian optimization, and set limits of hyperparameters
svrBO = BayesianOptimization(svr_evaluate, {'C': (2, 6.99), 'max_iter': (100,600.99)},verbose=0) #Define Bayesian optimization, and set limits of hyperparameters    
#Set number of initial runs and subsequent tests, and do the optimization. Also, we set kappa=10 (greater than the default) so there is more exploration when there are more hyperparameters
svrBO.maximize(init_points=10, n_iter=10, kappa=10)


# In[ ]:


svrBO.res
svrBO.res[18]


# ### Dense NN Decoder

# In[ ]:


#Declare model
model_dnn=DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)

#Fit model
model_dnn.fit(X_flat_train,y_train)

#Get predictions
y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)

#Get metric of fit
R2s_dnn=get_R2(y_valid,y_valid_predicted_dnn)
print('R2s:', R2s_dnn)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_dnn_optimized_pa29.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_valid_predicted_dnn': y_valid_predicted_dnn,'R2s_dnn':R2s_dnn})#, 'dnnBO':dnnBO})


# In[ ]:


def dnn_evaluate(num_units,frac_dropout,n_epochs):
    #The parameters need to be in the correct format for the decoder, so we do that below
    num_units=int(num_units)
    frac_dropout=float(frac_dropout)
    n_epochs=int(n_epochs)
    #Declare and fit decoder
    model_dnn=DenseNNDecoder(units=[num_units,num_units],dropout=frac_dropout,num_epochs=n_epochs)
    model_dnn.fit(X_flat_train,y_train)
    #Make predictions and get R2 values on validation set
    y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted_dnn))


# In[ ]:


#Do bayesian optimization, and set limits of hyperparameters
dnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 700.99), 'frac_dropout': (0,.5), 'n_epochs': (2,15.99)},verbose=0)

#Set number of initial runs (init_points) and subsequent tests (n_iter), and do the optimization
#kappa is a parameter that sets exploration vs exploitation in the algorithm - 10 seems to work pretty welldnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 500), 'frac_dropout': (0.,.5), 'n_epochs': (2,15)})
dnnBO.maximize(init_points=10, n_iter=10, kappa=10)


# In[ ]:


dnnBO.res
dnnBO.res[6]


# In[ ]:


#Declare model
model_dnn=DenseNNDecoder(units=539,dropout=0.05,num_epochs=9)

#Fit model
model_dnn.fit(X_flat_train,y_train)

#Get predictions
y_valid_predicted_dnn=model_dnn.predict(X_flat_valid)

#Get metric of fit
R2s_dnn=get_R2(y_valid,y_valid_predicted_dnn)
print('R2s:', R2s_dnn)


# ### Simple RNN Decoder

# In[ ]:


#Declare model
model_rnn=SimpleRNNDecoder(units=400,dropout=0,num_epochs=5)

#Fit model
model_rnn.fit(X_train,y_train)

#Get predictions
y_valid_predicted_rnn=model_rnn.predict(X_valid)

#Get metric of fit
R2s_rnn=get_R2(y_valid,y_valid_predicted_rnn)
print('R2s:', R2s_rnn)


# In[ ]:


def rnn_evaluate(num_units,frac_dropout,n_epochs):
    #The parameters need to be in the correct format for the decoder, so we do that below
    num_units=int(num_units)
    frac_dropout=float(frac_dropout)
    n_epochs=int(n_epochs)
    #Declare and fit decoder
    model_rnn=SimpleRNNDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
    model_rnn.fit(X_train,y_train)
    #Make predictions and get R2 values on validation set
    y_valid_predicted_rnn=model_rnn.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted_rnn))


# In[ ]:


#Do bayesian optimization, and set limits of hyperparameters
rnnBO = BayesianOptimization(rnn_evaluate, {'num_units': (50, 800.99), 'frac_dropout': (0,.5), 'n_epochs': (2,15.99)},verbose=0)

#Set number of initial runs (init_points) and subsequent tests (n_iter), and do the optimization
#kappa is a parameter that sets exploration vs exploitation in the algorithm - 10 seems to work pretty welldnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 500), 'frac_dropout': (0.,.5), 'n_epochs': (2,15)})
rnnBO.maximize(init_points=10, n_iter=10, kappa=10)


# In[ ]:


rnnBO.res
rnnBO.res[7]


# In[ ]:


#Declare model
model_rnn=SimpleRNNDecoder(units=352,dropout=0.08,num_epochs=4)

#Fit model
model_rnn.fit(X_train,y_train)

#Get predictions
y_valid_predicted_rnn=model_rnn.predict(X_valid)

#Get metric of fit
R2s_rnn=get_R2(y_valid,y_valid_predicted_rnn)
print('R2s:', R2s_rnn)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_rnn_optimized_pa29.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_valid_predicted_rnn': y_valid_predicted_rnn,'R2s_rnn':R2s_rnn,'rnnBO':rnnBO})


# ### GRU Decoder

# In[ ]:


#Declare model
model_gru=GRUDecoder(units=400,dropout=0,num_epochs=5)

#Fit model
model_gru.fit(X_train,y_train)

#Get predictions
y_valid_predicted_gru=model_gru.predict(X_valid)

#Get metric of fit
R2s_gru=get_R2(y_valid,y_valid_predicted_gru)
print('R2s:', R2s_gru)


# In[ ]:


def gru_evaluate(num_units,frac_dropout,n_epochs):
    #The parameters need to be in the correct format for the decoder, so we do that below
    num_units=int(num_units)
    frac_dropout=float(frac_dropout)
    n_epochs=int(n_epochs)
    #Declare and fit decoder
    model_rnn=GRUDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
    model_rnn.fit(X_train,y_train)
    #Make predictions and get R2 values on validation set
    y_valid_predicted_rnn=model_rnn.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted_rnn))


# In[ ]:


#Do bayesian optimization, and set limits of hyperparameters
gruBO = BayesianOptimization(gru_evaluate, {'num_units': (50, 700.99), 'frac_dropout': (0,.5), 'n_epochs': (2,15.99)},verbose=0)

#Set number of initial runs (init_points) and subsequent tests (n_iter), and do the optimization
#kappa is a parameter that sets exploration vs exploitation in the algorithm - 10 seems to work pretty welldnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 500), 'frac_dropout': (0.,.5), 'n_epochs': (2,15)})
gruBO.maximize(init_points=10, n_iter=10, kappa=10)


# In[ ]:


gruBO.res
gruBO.res[15]


# In[ ]:


#Declare model
model_gru=GRUDecoder(units=507,dropout=0,num_epochs=16)

#Fit model
model_gru.fit(X_train,y_train)

#Get predictions
y_valid_predicted_gru=model_gru.predict(X_valid)

#Get metric of fit
R2s_gru=get_R2(y_valid,y_valid_predicted_gru)
print('R2s:', R2s_gru)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_gru_optimized_pa29.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_valid_predicted_gru': y_valid_predicted_gru,'R2s_gru':R2s_gru,'gruBO':gruBO})


# In[ ]:


from bayes_opt import BayesianOptimization
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK


# ### LSTM Decoder

# In[ ]:


#Declare model
model_lstm=LSTMDecoder(units=400,dropout=0,num_epochs=5)

#Fit model
model_lstm.fit(X_train,y_train)

#Get predictions
y_valid_predicted_lstm=model_lstm.predict(X_valid)

#Get metric of fit
R2s_lstm=get_R2(y_valid,y_valid_predicted_lstm)
print('R2s:', R2s_lstm)


# In[ ]:


def lstm_evaluate(num_units,frac_dropout,n_epochs):
    #The parameters need to be in the correct format for the decoder, so we do that below
    num_units=int(num_units)
    frac_dropout=float(frac_dropout)
    n_epochs=int(n_epochs)
    #Declare and fit decoder
    model_lstm=LSTMDecoder(units=num_units,dropout=frac_dropout,num_epochs=n_epochs)
    model_lstm.fit(X_train,y_train)
    #Make predictions and get R2 values on validation set
    y_valid_predicted_lstm=model_lstm.predict(X_valid)
    return np.mean(get_R2(y_valid,y_valid_predicted_lstm))


# In[ ]:


#Do bayesian optimization, and set limits of hyperparameters
lstmBO = BayesianOptimization(lstm_evaluate, {'num_units': (50, 700.99), 'frac_dropout': (0,.5), 'n_epochs': (2,15.99)},verbose=0)

#Set number of initial runs (init_points) and subsequent tests (n_iter), and do the optimization
#kappa is a parameter that sets exploration vs exploitation in the algorithm - 10 seems to work pretty welldnnBO = BayesianOptimization(dnn_evaluate, {'num_units': (50, 500), 'frac_dropout': (0.,.5), 'n_epochs': (2,15)})
lstmBO.maximize(init_points=10, n_iter=10, kappa=10)


# In[ ]:


lstmBO.res
len(lstmBO.res)
lstmBO.res[19]


# In[ ]:


#Declare model
model_lstm=LSTMDecoder(units=447,dropout=0.269,num_epochs=11)

#Fit model
model_lstm.fit(X_train,y_train)

#Get predictions
y_valid_predicted_lstm=model_lstm.predict(X_valid)

#Get metric of fit
R2s_lstm=get_R2(y_valid,y_valid_predicted_lstm)
print('R2s:', R2s_lstm)


# In[ ]:


import scipy.io as sio
sio.savemat('results/decoding_data_1600ms_lstm_optimized_pa29.mat',{'y_valid': y_valid,'y_train_mean': y_train_mean, 'y_valid_predicted_lstm': y_valid_predicted_lstm, 'R2s_lstm':R2s_lstm,'lstmBO':lstmBO})


# In[ ]:


#As an example, I plot an example 1000 values of the x velocity (column index 0), both true and predicted with the Wiener filter
#Note that I add back in the mean value, so that both true and predicted values are in the original coordinates
fig_x_lstm=plt.figure()
plt.plot(y_valid[1400:2400,0]+y_train_mean[0],'b')
plt.plot(y_valid_predicted_lstm[1400:2400,0]+y_train_mean[0],'r')

#Save figure
# fig_x_wf.savefig('x_velocity_decoding.eps')


# In[ ]:



'''
