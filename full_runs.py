import numpy as np, sys, warnings
from scipy import io, stats
import pickle, time, pandas as pd, os.path, os
from joblib import Parallel, delayed
from psutil import cpu_count
import multiprocessing
import helpers
from handy_functions import dataSampling
from run_decoders import run_model
from matlab_funcs import mat_to_pickle
from metrics import get_R2, get_rho, get_RMSE

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# Get job parameters
PARAMS = 'params.txt'
s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,em,num_repeats,j = helpers.get_params(int(sys.argv[1]),PARAMS)

jobs = helpers.get_jobArray(fo,num_repeats)
print('# of jobs: {}'.format(len(jobs)))

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    jobID = int(sys.argv[3])
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    jobID = int(os.environ["SLURM_ARRAY_TASK_ID"])

job = jobs[jobID + (j*1000)]
outer_fold = job[0]
repeat = job[1]

print(f'fo{outer_fold}-re{repeat}')

#######################################################################################################################################
# Do some preprocessing first
sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)
neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,df)
pp_time = pp_time/pos_binned.shape[0]

toss_inds = helpers.remove_overlapBins(cond_binned, wi, dto)  # Remove bins of overlapping trials
neural_data, pos_binned, vel_binned, acc_binned, cond_binned = (
    np.delete(arr, toss_inds, axis=0) for arr in [neural_data, pos_binned, vel_binned, acc_binned, cond_binned])

# Determine which 'regime' we are in
if tp != 1.0:
    tp_repeats = num_repeats
    tp_repeat = repeat
    neuron_repeats = 1
    neuron_repeat = 0
else:
    tp_repeats = 1
    tp_repeat = 0
    neuron_repeats = num_repeats
    neuron_repeat = repeat

# Pull out neurons, either all of them or randomly sampled
neurons_perRepeat, nn, nm, nf = dataSampling.get_neuronRepeats(sess_nodt,nn=nn,nm=nm,nf=nf,num_repeats=neuron_repeats)
these_neurons = neurons_perRepeat[neuron_repeat]

# Split the data into train:valid:test sets and normalize
result = helpers.get_data(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  

# Train on a subset of the observations, based on tp
if tp != 1.0: 
    obs_perRepeat = dataSampling.get_trainSection(c_train,sess_nodt,outer_fold,tp=tp,num_repeats=tp_repeats)
    these_obs = obs_perRepeat[tp_repeat]
    X_train = X_train[these_obs,:,:]
    X_flat_train, y_train, y_zscore_train, c_train = [arr[these_obs, :] for arr in (X_flat_train, y_train, y_zscore_train, c_train)]

# calculate baseline eye traces (averaged within each condition)
#y_test_avg = helpers.avgEye_perCondition(c_train,y_train,c_test,y_test)
#y_test_zscore_avg = helpers.avgEye_perCondition(c_train,y_zscore_train,c_test,y_zscore_test)

#######################################################################################################################################
result,prms = run_model(m,o,em,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)
y_train_predicted, y_test_predicted, train_time, test_time = result
#y_train_predicted, y_test_predicted, y_shuf_predicted, y_mean_predicted, y_base_predicted, r2_train, rho_train, r2_test, rho_test, r2_shuf, rho_shuf, r2_mean, rho_mean, r2_base, rho_base, train_time, test_time = result

if m != 3:
    y_train_data = y_train
    y_test_data = y_test
else:
    y_train_data = y_zscore_train
    y_test_data = y_zscore_test

    y_test_predicted = y_test_predicted*np.std(y_train, axis=0)

R2_train = get_R2(y_train_data, y_train_predicted)
rho_train = get_rho(y_train_data, y_train_predicted)
rmse_train = get_RMSE(y_train_data, y_train_predicted)
R2_test = get_R2(y_test_data, y_test_predicted)
rho_test = get_rho(y_test_data, y_test_predicted)
rmse_test = get_RMSE(y_test_data, y_test_predicted)

print("R2 (test)    =  {}".format(R2_test))
print("rho (test)   =  {}".format(rho_test))
print("RMSE (test)  =  {}".format(rmse_test))
print("R2 (train)   =  {}".format(R2_train))
print("rho (train)  =  {}".format(rho_train))
print("RMSE (train) =  {}".format(rmse_train))

#helpers.plot_first_column_lines(y_test, y_test_predicted)

#######################################################################################################################################
cwd = os.getcwd()
jobname = helpers.make_name(int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,em,num_repeats)
pfile = helpers.make_directory((jobname),0)

output = {0: 'position', 1: 'velocity', 2: 'acceleration'}.get(o)
metric = {0: 'R2', 1: 'rho', 2: 'RMSE'}.get(em)
result = [int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,outer_fold,repeat,tp,y_train.shape[0],output,m,metric,prms,pp_time,train_time,test_time,R2_train,rho_train,rmse_train,R2_test,rho_test,rmse_test]     

truth_file = "actual-s{:02d}-t{:01d}-dto{:03d}-df{:01d}-o{:d}-fold{:0>1d}".format(s, t, dto, df, o, outer_fold)
file_path = os.path.join(cwd, 'runs/actual', truth_file + '.pickle')
if not os.path.isfile(file_path):
    print('saving recorded eye traces')
    with open(file_path, 'wb') as p:
        pickle.dump([y_test, c_test], p)

with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
    pickle.dump([result,y_test_predicted],p)

