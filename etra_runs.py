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
import numpy as np, sys, pickle, time, pandas as pd, os.path, os, random
from bayes_opt import BayesianOptimization, UtilityFunction
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# Get job parameters
PARAMS = 'params_etra.txt'
s,t,dto,df,wi,dti,nn,nm,nf,fo,tp,o,m,em,num_repeats,_ = helpers.get_params(int(sys.argv[1]),PARAMS)
print(wi)

jobs = helpers.get_jobArray(fo,num_repeats,2)
print('# of jobs: {}'.format(len(jobs)))

if int(sys.argv[2])==0: # local computer
    workers = multiprocessing.cpu_count() 
    jobID = int(sys.argv[3])
else: # hpc cluster
    workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    jobID = int(os.environ["SLURM_ARRAY_TASK_ID"])

job = jobs[jobID]
outer_fold = job[0]
repeat = job[1]
style = job[2]

print(f'fo{outer_fold}-re{repeat}-st{style}')

#######################################################################################################################################
# Do some preprocessing first
sess,sess_nodt = helpers.get_session(s,t,dto,df,400,dti)
neural_data,pos_binned,vel_binned,acc_binned,cond_binned,pp_time = mat_to_pickle('vars-'+sess_nodt+'.mat',dto,400,dti,df)
pp_time = pp_time/pos_binned.shape[0]

toss_inds = helpers.remove_overlapBins(cond_binned, 400, dto)  # Remove bins of overlapping trials
neural_data, pos_binned, vel_binned, acc_binned, cond_binned = (
    np.delete(arr, toss_inds, axis=0) for arr in [neural_data, pos_binned, vel_binned, acc_binned, cond_binned])

# Pull out neurons, either all of them or randomly sampled
neurons_perRepeat, _, _, _ = dataSampling.get_neuronRepeats(sess_nodt,nm=99,nf=0)
mt_neurons = neurons_perRepeat[repeat]
neurons_perRepeat, _, _, _ = dataSampling.get_neuronRepeats(sess_nodt,nm=0,nf=99)
fef_neurons = neurons_perRepeat[repeat]
neurons_perRepeat, nn, nm, nf = dataSampling.get_neuronRepeats(sess_nodt,nm=99,nf=99)
all_neurons = neurons_perRepeat[repeat]

verb = 1

if style==0: #SISO
    result = helpers.get_data(neural_data[:,:,all_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
    X_train,X_test,X_valid,X_flat_train,X_flat_test,X_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  
    
    t1=time.time()
    pbounds = {
        'bins_before': (1, 8),
        'num_units': (50, 600),
        'frac_dropout': (0.1, 0.75),
        'batch_size': (64, 256),
        'n_epochs': (2, 21)
    }
    
    if m==5:
        from decoders import SimpleRNNDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_train, X_valid, X_test, y_train, y_valid, y_test
        
        def rnn_evaluate(bins_before,num_units,frac_dropout,batch_size,n_epochs):
            model_rnn=SimpleRNNDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_rnn.fit(Xtr[:,int(-bins_before):,:], ytr)
            y_valid_predicted_rnn=model_rnn.predict(Xva[:,int(-bins_before):,:])
            return np.mean(get_R2(yva,y_valid_predicted_rnn))

        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(rnn_evaluate, pbounds, verbose=verb, allow_duplicate_points=True,random_state=m)
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10
        
        best_params = BO.max['params']
        bins_before = int(best_params['bins_before'])
        num_units = int(best_params['num_units'])
        frac_dropout = float(best_params['frac_dropout'])
        batch_size = int(best_params['batch_size'])
        n_epochs = int(best_params['n_epochs'])
        prms = {'bins_before':bins_before,'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model = SimpleRNNDecoder(units=num_units,dropout=frac_dropout,batch_size=batch_size,num_epochs=n_epochs,workers=workers, verbose=1)
        
    if m==7:
        from decoders import LSTMDecoder
        Xtr, Xva, Xte, ytr, yva, yte = X_train, X_valid, X_test, y_train, y_valid, y_test

        def lstm_evaluate(bins_before, num_units, frac_dropout, batch_size, n_epochs):
            model_lstm=LSTMDecoder(units=int(num_units),dropout=float(frac_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
            model_lstm.fit(Xtr[:,int(-bins_before):,:], ytr)
            y_valid_predicted_lstm = model_lstm.predict(Xva[:,int(-bins_before):,:])
            return np.mean(get_R2(yva,y_valid_predicted_lstm))

        pbounds = {
            'bins_before': (1, 8),
            'num_units': (50, 300),
            'frac_dropout': (0.25, 0.75),
            'batch_size': (64, 256),
            'n_epochs': (2, 21)
        }
        acquisition_function = UtilityFunction(kind="ucb", kappa=10)
        BO = BayesianOptimization(lstm_evaluate, pbounds, verbose=verb, allow_duplicate_points=True,random_state=m)
        BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10
        
        best_params = BO.max['params']
        bins_before = int(best_params['bins_before'])
        num_units = int(best_params['num_units'])
        frac_dropout = float(best_params['frac_dropout'])
        batch_size = int(best_params['batch_size'])
        n_epochs = int(best_params['n_epochs'])
        prms = {'bins_before':bins_before,'num_units': num_units, 'frac_dropout': frac_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

        model = LSTMDecoder(units=num_units, dropout=frac_dropout, batch_size=batch_size, num_epochs=n_epochs, workers=workers, verbose=1)
    
    model.fit(Xtr[:,int(-bins_before):,:],ytr,tb=1) 
    train_time = time.time()-t1
    y_train_predicted=model.predict(Xtr[:,int(-bins_before):,:]) # train accuracy 
   
    t2=time.time()
    y_test_predicted=model.predict(Xte[:,int(-bins_before):,:])   
    test_time = (time.time()-t2) / yte.shape[0]

elif style==1: #MISO
    result = helpers.get_data(neural_data[:,:,mt_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
    Xmt_train,Xmt_test,Xmt_valid,Xmt_flat_train,Xmt_flat_test,Xmt_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  
    result = helpers.get_data(neural_data[:,:,fef_neurons],o,pos_binned,vel_binned,acc_binned,cond_binned,fo,outer_fold,wi/dti)
    Xfef_train,Xfef_test,Xfef_valid,Xfef_flat_train,Xfef_flat_test,Xfef_flat_valid,y_train,y_test,y_valid,y_zscore_train,y_zscore_test,y_zscore_valid,c_train,c_test = result  
    
    t1=time.time()

    from special_decoders import LSTMDecoder_miso
    Xtr_mt, Xva_mt, Xte_mt, Xtr_fef, Xva_fef, Xte_fef, ytr, yva, yte = Xmt_train, Xmt_valid, Xmt_test, Xfef_train, Xfef_valid, Xfef_test, y_train, y_valid, y_test

    def lstm_evaluate(mt_units, fef_units, mt_dropout, fef_dropout, batch_size, n_epochs):
        model_lstm=LSTMDecoder_miso(mt_units=int(mt_units),fef_units=int(fef_units),mt_dropout=float(mt_dropout),fef_dropout=float(fef_dropout),batch_size=int(batch_size),num_epochs=int(n_epochs),workers=workers)
        model_lstm.fit(Xtr_mt, Xtr_fef, ytr)
        y_valid_predicted_lstm = model_lstm.predict(Xva_mt, Xva_fef)
        return np.mean(get_R2(yva,y_valid_predicted_lstm))

    pbounds = {
        'mt_units': (50, 300),
        'fef_units': (50, 300),
        'mt_dropout': (0.25, 0.75),
        'fef_dropout': (0.25, 0.75),
        'batch_size': (128, 256),
        'n_epochs': (2, 21)
    }
    acquisition_function = UtilityFunction(kind="ucb", kappa=10)
    BO = BayesianOptimization(lstm_evaluate, pbounds, verbose=verb, allow_duplicate_points=True,random_state=m)
    BO.maximize(init_points=10, n_iter=10,acquisition_function=acquisition_function)#, n_jobs=workers) 10,10
    
    best_params = BO.max['params']
    mt_units = int(best_params['mt_units'])
    fef_units = int(best_params['fef_units'])
    mt_dropout = float(best_params['mt_dropout'])
    fef_dropout = float(best_params['fef_dropout'])
    batch_size = int(best_params['batch_size'])
    n_epochs = int(best_params['n_epochs'])
    prms = {'mt_units': mt_units, 'fef_units': fef_units, 'mt_dropout': mt_dropout, 'fef_dropout': fef_dropout, 'batch_size': batch_size, 'n_epochs': n_epochs}

    model = LSTMDecoder_miso(mt_units=mt_units, fef_units=fef_units, mt_dropout=mt_dropout, fef_dropout=fef_dropout, batch_size=batch_size, num_epochs=n_epochs, workers=workers, verbose=1)
    model.fit(Xtr_mt,Xtr_fef,ytr) 
    train_time = time.time()-t1
    y_train_predicted=model.predict(Xtr_mt,Xtr_fef) # train accuracy 
   
    # model testing
    t2=time.time()
    y_test_predicted=model.predict(Xte_mt,Xte_fef)   
    test_time = (time.time()-t2) / yte.shape[0]

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

print(blah)

if m==3:
    y_test_predicted = y_test_predicted*np.std(y_train, axis=0)

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



