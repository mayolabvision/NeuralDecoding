import numpy as np, sys, warnings
from scipy import io, stats
import pickle, time, pandas as pd, os.path, os
from joblib import Parallel, delayed
from psutil import cpu_count
import multiprocessing
import helpers
from handy_functions import dataSampling
from run_decoders import run_model
from etra_decoders import run_siso, run_miso
from matlab_funcs import mat_to_pickle
from metrics import get_R2, get_rho, get_RMSE
import numpy as np, sys, pickle, time, pandas as pd, os.path, os, random
from bayes_opt import BayesianOptimization, UtilityFunction
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get job parameters
PARAMS = 'other_params/params_etra.txt'
s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,style,pcType,num_repeats,_ = helpers.get_params_etra(int(sys.argv[1]),PARAMS)

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

    # Do some preprocessing first
    sess,sess_nodt = helpers.get_session(s,t,dto,df,wi,dti)
    neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,wi,dti,datapath,df)
    pp_time = pp_time/pos_binned.shape[0]

    toss_inds = helpers.remove_overlapBins(cond_binned, wi, dto)  # Remove bins of overlapping trials
    neural_data, pos_binned, vel_binned, acc_binned, cond_binned = (
        np.delete(arr, toss_inds, axis=0) for arr in [neural_data, pos_binned, vel_binned, acc_binned, cond_binned])

    # Pull out neurons, either all of them or randomly sampled
    neurons_perRepeat, nn, nm, nf = dataSampling.get_neuronRepeats(sess_nodt,datapath,nn=nn,nm=nm,nf=nf,num_repeats=num_repeats)
    these_neurons = neurons_perRepeat[repeat]
    
    mt_perRepeat, _, _, _ = dataSampling.get_neuronRepeats(sess_nodt,datapath,nm=99,nf=0)
    mt_neurons = mt_perRepeat[0]
    fef_perRepeat, _, _, _ = dataSampling.get_neuronRepeats(sess_nodt,datapath,nm=0,nf=99)
    fef_neurons = fef_perRepeat[0]

    print(f'nn{nn}-nm{nm}-nf{nf}')

    ########################
    # Before running decoder, check that this run doesn't already exist.
    pfile = helpers.make_name_etra(int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,style,pcType,num_repeats,datapath)
    decoder_path = pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle'
    if os.path.exists(decoder_path):
        print("ALREADY RAN")
        print('------------')
        continue
    ########################

    if style==0: #SISO
        result = helpers.get_data(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
        X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  
        
        if pcType==1:
            Xmt_train_pca, Xmt_valid_pca, Xmt_test_pca, nmt_components, pve_mt = helpers.do_pca(X_train[:,:,mt_neurons],X_valid[:,:,mt_neurons],X_test[:,:,mt_neurons],explain_var=0.9)
            Xfef_train_pca, Xfef_valid_pca, Xfef_test_pca, nfef_components, pve_fef = helpers.do_pca(X_train[:,:,fef_neurons],X_valid[:,:,fef_neurons],X_test[:,:,fef_neurons],explain_var=0.9)

            X_train = np.concatenate((Xmt_train_pca, Xfef_train_pca), axis=2)
            X_valid = np.concatenate((Xmt_valid_pca, Xfef_valid_pca), axis=2)
            X_test = np.concatenate((Xmt_test_pca, Xfef_test_pca), axis=2)

        elif pcType==2:
            X_train, X_valid, X_test, n_components, pve = helpers.do_pca(X_train[:,:,these_neurons],X_valid[:,:,these_neurons],X_test[:,:,these_neurons],explain_var=0.1)

        result,prms = run_siso(m,o,1,workers,int(sys.argv[2]),X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)
        y_train_predicted, y_test_predicted, train_time, test_time = result
                 
        if pcType == 0:
            prms = prms
        elif pcType == 1:
            prms['ncomps_mt'] = nmt_components
            prms['ncomps_fef'] = nfef_components
            prms['pve_mt'] = pve_mt
            prms['pve_fef'] = pve_fef
        else:
            prms['ncomps'] = n_components
            prms['pve'] = pve
    

    elif style==1: #MISO
        result = helpers.get_data(neural_data[:,:,mt_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
        Xmt_train,Xmt_test,Xmt_valid,Xmt_flat_train,Xmt_flat_test,Xmt_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  
        result = helpers.get_data(neural_data[:,:,fef_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
        Xfef_train,Xfef_test,Xfef_valid,Xfef_flat_train,Xfef_flat_test,Xfef_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  
        
        result,prms = run_miso(m,o,1,workers,int(sys.argv[2]),Xmt_train,Xmt_test,Xmt_valid,Xmt_flat_train,Xmt_flat_test,Xmt_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,Xfef_train,Xfef_test,Xfef_valid,Xfef_flat_train,Xfef_flat_test,Xfef_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)
        y_train_predicted, y_test_predicted, train_time, test_time = result


    elif style==2: # attention layer
        result = helpers.get_data(neural_data[:,:,these_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
        X_train,X_test,X_valid,_,_,_,y_train,y_test,y_valid,_,_,_,c_train,c_test = result  
        
        result,prms = run_siso(m,o,1,workers,int(sys.argv[2]),X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid)
        y_train_predicted, y_test_predicted, train_time, test_time = result

    #############################################3
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
    metric = {0: 'siso', 1: 'miso', 2: 'attn'}.get(style)
    pcaFlag = {0: 'none', 1: 'sep', 2: 'tog'}.get(pcType)
    
    result = [int(sys.argv[1]),s,t,dto,df,wi,dti,nn,nm,nf,outer_fold,repeat,tp,y_train.shape[0],output,m,metric,pcaFlag,prms,pp_time,train_time,test_time,R2_train,rho_train,rmse_train,R2_test,rho_test,rmse_test]     
    truth_file = "actual-s{:02d}-t{:01d}-dto{:03d}-df{:01d}-o{:d}-fold{:0>1d}".format(s, t, dto, df, o, outer_fold)
    file_path = os.path.join(datapath, 'runs/actual', truth_file + '.pickle')
    if not os.path.isfile(file_path):
        print('saving recorded eye traces')
        with open(file_path, 'wb') as p:
            pickle.dump([y_test, c_test], p)
    
    with open(pfile+'/fold{:0>1d}_repeat{:0>3d}'.format(outer_fold,repeat)+'.pickle','wb') as p:
        pickle.dump([result,y_test_predicted],p)
    print('------------')

