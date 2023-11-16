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
params = 'params/params_cosyne.txt'

from metrics import get_R2
from metrics import get_rho
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import multiprocessing
from psutil import cpu_count
import helpers
import neuronsSample
from run_decoders import run_model
from matlab_funcs import mat_to_pickle

bins_predict = 20

line = np.loadtxt(params)[int(sys.argv[1])]
print(line)
s,t,dto,df,o,wi,dti,m,nm,nf,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]),params)
jobs = helpers.get_jobArray(fo,num_repeats)
print('# of jobs: {}'.format(len(jobs)))

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    job = jobs[int(sys.argv[3])]
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    job = jobs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

outer_fold = job[0]
repeat = job[1]

sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)
neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,df)

# Create a matrix with a sliding window view
if o==0:
    X_binned = np.lib.stride_tricks.sliding_window_view(pos_binned[:,0], window_shape=(bins_predict,))
    Y_binned = np.lib.stride_tricks.sliding_window_view(pos_binned[:,1], window_shape=(bins_predict,))
elif o==1:
    X_binned = np.lib.stride_tricks.sliding_window_view(vel_binned[:,0], window_shape=(bins_predict,))
    Y_binned = np.lib.stride_tricks.sliding_window_view(vel_binned[:,1], window_shape=(bins_predict,))
else:
    X_binned = np.lib.stride_tricks.sliding_window_view(acc_binned[:,0], window_shape=(bins_predict,))
    Y_binned = np.lib.stride_tricks.sliding_window_view(acc_binned[:,1], window_shape=(bins_predict,))

neural_data = neural_data[:-(bins_predict-1),:,:]
cond_binned = cond_binned[:-(bins_predict-1),:]

[neurons_perRepeat,nm,nf] = neuronsSample.get_neuronRepeats(s,t,num_repeats,nm,nf)
these_neurons = neurons_perRepeat[repeat]

outputs = ['position','velocity','acceleration']
#######################################################################################################################################
# x 
X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test = helpers.get_data(neural_data[:,:,these_neurons],o,X_binned,vel_binned,acc_binned,cond_binned,fo,fi,outer_fold,wi/dti,m)

r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,coef_dict,prms,y_test,y_test_predicted,train_time,test_time = run_model(m,o,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)

result = [s,t,dto,df,wi,dti,m,bins_predict,outputs[o],nm,nf,repeat,outer_fold,r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,prms,pp_time,train_time,test_time,these_neurons]     

jobname = helpers.make_name(s,t,dto,df,o,wi,dti,m,nm,nf,fo,fi,num_repeats)
pfile = helpers.make_directory((jobname),0)
with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>3d}_bp{:0>2d}_X'.format(outer_fold,repeat,bins_predict)+'.pickle','wb') as p:
    pickle.dump([result,c_test,y_test,y_test_predicted],p)

# y 
X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test = helpers.get_data(neural_data[:,:,these_neurons],o,Y_binned,vel_binned,acc_binned,cond_binned,fo,fi,outer_fold,wi/dti,m)

r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,coef_dict,prms,y_test,y_test_predicted,train_time,test_time = run_model(m,o,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)

result = [s,t,dto,df,wi,dti,m,outputs[o],nm,nf,repeat,outer_fold,r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,prms,pp_time,train_time,test_time,these_neurons]     

with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>3d}_bp{:0>2d}_Y'.format(outer_fold,repeat,bins_predict)+'.pickle','wb') as p:
    pickle.dump([result,c_test,y_test,y_test_predicted],p)


