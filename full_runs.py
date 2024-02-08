import numpy as np, sys, warnings
from scipy import io, stats
import pickle, time, pandas as pd, os.path, os
from joblib import Parallel, delayed
from psutil import cpu_count
import multiprocessing
import helpers
from handy_functions import neuronsSample
from run_decoders import run_model
from matlab_funcs import mat_to_pickle
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# Get job parameters
PARAMS = 'params/params.txt'
s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,num_repeats = helpers.get_params(int(sys.argv[1]),PARAMS)
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

# Do some preprocessing first
sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)
neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,df)

toss_inds = helpers.remove_overlapBins(cond_binned, wi, dto)  # Remove bins of overlapping trials
neural_data, pos_binned, vel_binned, acc_binned, cond_binned = (
    np.delete(arr, toss_inds, axis=0) for arr in [neural_data, pos_binned, vel_binned, acc_binned, cond_binned])

if nn == 99 and nm == 99 and nf == 99:  # Use all neurons in recording session
    neurons_perRepeat, nm, nf = neuronsSample.get_neuronRepeats(s, t, num_repeats)
    nn = nm + nf
elif nm == 99 and nf == 99:  # Don't specify number of MT or FEF neurons specifically, just total neurons
    neurons_perRepeat, nm, nf = neuronsSample.get_neuronRepeats(s, t, num_repeats, nn)
elif nm != 99 and nf != 99:  # Specify number of MT and FEF neurons, calculate total neurons
    neurons_perRepeat, _, _ = neuronsSample.get_neuronRepeats(s, t, num_repeats, nm, nf)
    nn = nm + nf

these_neurons = neurons_perRepeat[repeat]

result = helpers.get_data(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti,m)
X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_test,y_test_avg,y_test_zscore_avg = result 

#######################################################################################################################################
result,prms = run_model(m,o,1,workers,X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,y_test_avg,y_test_zscore_avg)

y_train_predicted, y_test_predicted, y_shuf_predicted, y_mean_predicted, y_base_predicted, r2_train, rho_train, r2_test, rho_test, r2_shuf, rho_shuf, r2_mean, rho_mean, r2_base, rho_base, train_time, test_time = result

print("R2 = {}".format(r2_test))
print("R2 null = {}".format(r2_shuf))
print("R2 mean = {}".format(r2_mean))
print("R2 base = {}".format(r2_base))

print(y_test.shape)
print(y_test_predicted.shape)
print(y_shuf_predicted.shape)
print(y_mean_predicted.shape)
print(y_base_predicted.shape)

helpers.plot_first_column_lines(y_test, y_test_predicted, y_mean_predicted, y_base_predicted)



print(blahblah)
#######################################################################################################################################
if o==0:
    output = 'position'
elif o==1:
    output = 'velocity'
elif o==2:
    output = 'acceleration'

result = [s,t,dto,df,wi,dti,m,output,nm,nf,repeat,outer_fold,r2mn_train,rhomn_train,r2mn_test,rhomn_test,r2mn_shuf,rhomn_shuf,eval_full,prms,pp_time,train_time,test_time]     

cwd = os.getcwd()
jobname = helpers.make_name(int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,num_repeats)
pfile = helpers.make_directory((jobname),0)
if s==29 and repeat==0:
    with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result,c_test,y_test,y_test_predicted],p)
else:
    with open(cwd+pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result],p)
