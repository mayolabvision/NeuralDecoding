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

np.set_printoptions(suppress=True)

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

line = np.loadtxt(params)[int(sys.argv[1])]
stpt = 10 

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

first_column = cond_binned[:, 0]
fifth_column = np.zeros_like(first_column)

unique_numbers, unique_indices = np.unique(first_column, return_inverse=True)
count = np.bincount(unique_indices)
cumulative_count = np.cumsum(count)

for i, num in enumerate(unique_numbers):
    inds = np.array(np.where(first_column==num))
    fifth_column[inds] = np.arange(inds.shape[1])

cond_binned = np.column_stack((cond_binned, fifth_column))

stpts = np.array(np.where((cond_binned[:, 4] <= stpt) & (cond_binned[:, 4] > stpt - 20))).squeeze()
neural_data = neural_data[stpts,:,:]
X_binned = X_binned[stpts,:]
Y_binned = Y_binned[stpts,:]
cond_binned = cond_binned[stpts,:]

print(blah)

[neurons_perRepeat,nm,nf] = neuronsSample.get_neuronRepeats(s,t,num_repeats,nm,nf)
these_neurons = neurons_perRepeat[repeat]

X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,c_test = helpers.get_data_classifier(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,fi,outer_fold,wi/dti,m)

y_train = np.expand_dims(y_train[:,o],axis=1)
y_test = np.expand_dims(y_test[:,o],axis=1)
y_valid = np.expand_dims(y_valid[:,o],axis=1)

print(y_train)
print(blah)

t1=time.time()
#######################################################################################################################################

r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,coef_dict,prms,y_test,y_test_predicted,train_time,test_time = run_model(m,o,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_train,y_test,y_valid)

#######################################################################################################################################
outputs = ['contrast','speed','direction']

result = [s,t,dto,df,wi,dti,m,outputs[o],nm,nf,repeat,outer_fold,r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,prms,pp_time,train_time,test_time,these_neurons]     

jobname = helpers.make_name(s,t,dto,df,o,wi,dti,m,nm,nf,fo,fi,num_repeats)
pfile = helpers.make_directory((jobname),0)
with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>3d}_{}_{}'.format(outer_fold,repeat,conditions[trCo],conditions[teCo])+'.pickle','wb') as p:
    pickle.dump([result],p)

