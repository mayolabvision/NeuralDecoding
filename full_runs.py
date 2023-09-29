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
params = 'params/params_full.txt'

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
#mdls = [0,1,3,4,5,6,7,8]
mdls = [7]

print(line)
s,t,dto,df,o,wi,dti,_,_,_,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
#foldneuron_pairs = helpers.get_foldneuronPairs(int(sys.argv[1]))
foldneuronmodel_pairs = helpers.get_foldneuronmodelPairs(fo,num_repeats,mdls)

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    neuron_fold_model = foldneuronmodel_pairs[int(sys.argv[3])]
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    neuron_fold_model = foldneuronmodel_pairs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

outer_fold = neuron_fold_model[0]
repeat = neuron_fold_model[1]
m = neuron_fold_model[2]
print(m)

sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)
neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,df)

[neurons_perRepeat,nm,nf] = neuronsSample.get_neuronRepeats(s,t,num_repeats)
these_neurons = neurons_perRepeat[repeat]

X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test = helpers.get_data(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,fi,outer_fold,wi/dti,m)

t1=time.time()
#if m==2:
#X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test = helpers.get_data(neural_data,these_neurons,o,pos_binned,vel_binned,acc_binned,cond_binned,outer_fold,bn,m)

#######################################################################################################################################

r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,coef_dict,prms,y_test,y_test_predicted,train_time,test_time = run_model(m,o,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)

#######################################################################################################################################
if o==0:
    output = 'position'
elif o==1:
    output = 'velocity'
elif o==2:
    output = 'acceleration'

result = [s,t,dto,df,wi,dti,m,output,nm,nf,repeat,outer_fold,r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,prms,pp_time,train_time,test_time]     

jobname = helpers.make_name(s,t,dto,df,o,wi,dti,m,nm,nf,fo,fi,num_repeats)
pfile = helpers.make_directory((jobname),0)
if s==29 and repeat==0:
    with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>2d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result,c_test,y_test,y_test_predicted],p)
else:
    with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>2d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result],p)

'''
else:
    for oo in range(3):
        if oo==0:
            output = 'position'
            rr2_train = r2mn_train[:2]
            rrho_train = rhomn_train[:2]
            rr2_test = r2mn_test[:2]
            rrho_test = rhomn_test[:2]
        elif oo==1:
            output = 'velocity'
            rr2_train = r2mn_train[2:4]
            rrho_train = rhomn_train[2:4]
            rr2_test = r2mn_test[2:4]
            rrho_test = rhomn_test[2:4]
        elif oo==2:
            output = 'acceleration'
            rr2_train = r2mn_train[4:]
            rrho_train = rhomn_train[4:]
            rr2_test = r2mn_test[4:]
            rrho_test = rhomn_test[4:]

        result = [s,t,dto,df,wi,dti,m,output,nm,nf,repeat,outer_fold,rr2mn_train,rrhomn_train,rr2mn_test,rrhomn_test,eval_full,prms,pp_time,train_time,test_time]     

        jobname = helpers.make_name(s,t,dto,df,oo,wi,dti,m,nm,nf,fo,fi,num_repeats)
        pfile = helpers.make_directory((jobname),0)
        if s==29 and repeat==0:
            with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>2d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
                pickle.dump([result,c_test,y_test,y_test_predicted],p)
        else:
            with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>2d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
                pickle.dump([result],p)
'''

