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
params = 'params/params_fig3a.txt'

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

line = np.loadtxt(params)[int(sys.argv[1])]
print(line)
s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats = helpers.get_params(int(sys.argv[1]))
foldneuron_pairs = helpers.get_foldneuronPairs(int(sys.argv[1]))

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    neuron_fold = foldneuron_pairs[int(sys.argv[3])]
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    neuron_fold = foldneuron_pairs[int(os.environ["SLURM_ARRAY_TASK_ID"])]

outer_fold = neuron_fold[0]
repeat = neuron_fold[1]

[neurons_perRepeat,nm,nf] = neuronsSample.get_neuronRepeats(s,t,d,num_repeats)
these_neurons = neurons_perRepeat[repeat]

sess,sess_nodt = helpers.get_session(s,t,d)
jobname = helpers.make_name(s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats)

with open(cwd+'/datasets/vars/vars-'+sess+'.pickle','rb') as f:
    neural_data,pos_binned,vel_binned,acc_binned,cond_binned=pickle.load(f,encoding='latin1')

if o==0:
    output = 'position'
elif o==1:
    output = 'velocity'
elif o==2:
    output = 'acceleration'

X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test = helpers.get_data(neural_data,these_neurons,o,pos_binned,vel_binned,acc_binned,cond_binned,outer_fold,bn,m)

t1=time.time()
#######################################################################################################################################

r2,rho,coef_dict,prms,y_test,y_test_predicted = run_model(m,o,bn,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)

#######################################################################################################################################
time_elapsed = time.time()-t1
print("time elapsed = {} mins".format(time_elapsed/60))

#s,t,d,m,o,nm,nf,bn,fo,fi,num_repeats
result = [s,t,d,m,output,nm,nf,bn,repeat,outer_fold,r2,rho,prms,time_elapsed]     

pfile = helpers.make_directory('BinSweep_test/'+(jobname),0)
if s==29 and repeat==0:
    with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>2d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result,c_test,y_test,y_test_predicted],p)
#else:
with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>2d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
    pickle.dump([result],p)


