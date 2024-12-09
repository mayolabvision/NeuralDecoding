import numpy as np, sys, warnings
from scipy import io, stats
import pickle, time, pandas as pd, os.path, os
from joblib import Parallel, delayed
from psutil import cpu_count
import multiprocessing
import helpers
from handy_functions import dataSampling, crossDecoding
from run_decoders import run_model
from matlab_funcs import mat_to_pickle
from metrics import get_R2, get_rho, get_RMSE
from itertools import combinations

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    datapath = '/ix1/pmayo/decoding/' 
else: # hpc cluster, interactive job
    workers = int(2)
    jobID = int(sys.argv[3])
    arraySize = 100
    datapath = '/ix1/pmayo/decoding/' 

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
    
    sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)

    # Pull out neurons, either all of them or randomly sampled
    neurons_perRepeat, nn, nm, nf = dataSampling.get_neuronRepeats(sess_nodt,datapath,nn=nn,nm=nm,nf=nf,num_repeats=1)
    these_neurons = neurons_perRepeat[0]

    # Do some preprocessing first
    if 'neural_data' not in locals():
        neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,datapath,df)
        pp_time = pp_time/pos_binned.shape[0]
        toss_inds = helpers.remove_overlapBins(cond_binned, wi, dto)  # Remove bins of overlapping trials
        neural_data, pos_binned, vel_binned, acc_binned, cond_binned = (
            np.delete(arr, toss_inds, axis=0) for arr in [neural_data, pos_binned, vel_binned, acc_binned, cond_binned])

    if o==0:
        y = pos_binned
    elif o==1:
        y = vel_binned
    else:
        y = acc_binned

    num_trls = crossDecoding.get_trial_num(cond_binned)
    parameters = ['contrast', 'speed', 'direction', 'AV']
    for param in parameters:
        if param in ['contrast', 'speed', 'AV']:
            num_conds = 3
        elif param in ['direction']:
            num_conds = 6

        for tr in range(num_conds):
            trls_tr, rows_tr, cond_tr = crossDecoding.get_trials(cond_binned,num_trls,num_repeats=num_repeats,condition=parameters[3],cond_ind=tr,repeat=repeat)
            for te in range(num_conds):
                trls_te, rows_te, cond_te = crossDecoding.get_trials(cond_binned,num_trls,num_repeats=num_repeats,condition=param,cond_ind=te,repeat=repeat)

                ########################
                # Before running decoder, check that this run doesn't already exist.
                pfile = helpers.make_name(int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,num_repeats,datapath)
                decoder_path = pfile+'/fold{:0>1d}_repeat{:0>3d}_{}_tr{}_te{}'.format(outer_fold,repeat,param,tr,te)+'.pickle'
                if os.path.exists(decoder_path):
                    print("ALREADY RAN")
                    print('------------')
                    continue
                ########################

                result = helpers.get_data_xd(neural_data[rows_tr,:,:],neural_data[rows_te,:,:],y[rows_tr,:],y[rows_te,:],cond_binned[rows_tr,:],cond_binned[rows_te,:],outer_fold)
                X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  

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

                print("train: {}, test: {}".format(cond_tr, cond_te))
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
                result = [int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,outer_fold,repeat,tp,y_train.shape[0],output,m,prms,param,cond_tr,cond_te,pp_time,train_time,test_time,R2_train,rho_train,rmse_train,R2_test,rho_test,rmse_test]     

                truth_file = "actual-s{:02d}-t{:01d}-dto{:03d}-df{:01d}-o{:d}-fold{:0>1d}".format(s, t, dto, df, o, outer_fold)
                file_path = os.path.join(datapath, 'runs_neurips/actual', truth_file + '.pickle')
                if not os.path.isfile(file_path):
                    print('saving recorded eye traces')
                    with open(file_path, 'wb') as p:
                        pickle.dump([y_test, c_test], p)
                
                with open(pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
                    pickle.dump([result,y_test_predicted],p)
                print('------------')

