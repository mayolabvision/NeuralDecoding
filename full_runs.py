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
s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,num_repeats,_ = helpers.get_params(int(sys.argv[1]),PARAMS)

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    jobID = int(sys.argv[3])
    arraySize = 100
    datapath = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/' 
elif int(sys.argv[2])==1: # hpc cluster, batch job
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    jobID = int(os.environ["SLURM_ARRAY_TASK_ID"])
    arraySize = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    datapath = '/ix1/pmayo/neuraldecoding/' 
else: # hpc cluster, interactive job
    workers = int(2)
    jobID = int(sys.argv[3])
    arraySize = 100
    datapath = '/ix1/pmayo/neuraldecoding/' 

# TO-DO: add a line that looks for whether result for this job exists or not, skips to next if it does
jobs = helpers.get_jobArray(fo,num_repeats)
print('# of jobs: {}'.format(len(jobs)))

if len(jobs) <= arraySize:
    job_arr = [jobID]
else:
    runs_per_job = int(len(jobs)/arraySize) 
    start_job = jobID * runs_per_job
    job_arr = [start_job + i for i in range(runs_per_job)]

#################################################################################################################################
# Loop through each job in this section of the job array 
for j, job in enumerate(job_arr):
    thisjob = jobs[job]
    outer_fold = thisjob[0]
    repeat = thisjob[1]
    print(f'fo{outer_fold}-re{repeat}')

    # Do some preprocessing first
    sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)
    neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,datapath,df)
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
    neurons_perRepeat, nn, nm, nf = dataSampling.get_neuronRepeats(sess_nodt,datapath,nn=nn,nm=nm,nf=nf,num_repeats=neuron_repeats)
    these_neurons = neurons_perRepeat[neuron_repeat]

    ########################
    # Before running decoder, check that this run doesn't already exist.
    pfile = helpers.make_name(int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,num_repeats,datapath)
    decoder_path = pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle'
    if os.path.exists(decoder_path):
        print("ALREADY RAN")
        print('------------')
        continue
    ########################

    # Split the data into train:valid:test sets and normalize
    result = helpers.get_data(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
    X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  

    # Train on a subset of the observations, based on tp
    if tp != 1.0: 
        obs_perRepeat = dataSampling.get_trainSection(c_train,sess_nodt,outer_fold,datapath,dto=dto,tp=tp,num_repeats=tp_repeats)
        these_obs = obs_perRepeat[tp_repeat]
        X_train = X_train[these_obs,:,:]
        X_flat_train, y_train, y_zscore_train, c_train = [arr[these_obs, :] for arr in (X_flat_train, y_train, y_zscore_train, c_train)]

    # calculate baseline eye traces (averaged within each condition)
    #y_test_avg = helpers.avgEye_perCondition(c_train,y_train,c_test,y_test)
    #y_test_zscore_avg = helpers.avgEye_perCondition(c_train,y_zscore_train,c_test,y_zscore_test)

    ##############################
    result,prms = run_model(m,o,1,workers,int(sys.argv[2]),X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)
    y_train_predicted, y_test_predicted, train_time, test_time = result

    if m != 3:
        y_train_data = y_train
        y_test_data = y_test
    else:
        y_train_data = y_zscore_train
        y_test_data = y_zscore_test

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

    if m==3:
        y_test_predicted = y_test_predicted*np.std(y_train, axis=0)

    #######################################################################
    output = {0: 'position', 1: 'velocity', 2: 'acceleration'}.get(o)
    result = [int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,outer_fold,repeat,tp,y_train.shape[0],output,m,prms,pp_time,train_time,test_time,R2_train,rho_train,rmse_train,R2_test,rho_test,rmse_test]     

    truth_file = "actual-s{:02d}-t{:01d}-dto{:03d}-df{:01d}-o{:d}-fold{:0>1d}".format(s, t, dto, df, o, outer_fold)
    file_path = os.path.join(datapath, 'runs/actual', truth_file + '.pickle')
    if not os.path.isfile(file_path):
        print('saving recorded eye traces')
        with open(file_path, 'wb') as p:
            pickle.dump([y_test, c_test], p)
    
    with open(pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result,y_test_predicted],p)
    print('------------')

