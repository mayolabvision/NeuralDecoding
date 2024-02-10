import pandas as pd
import numpy as np
from scipy import io
import os
import pickle
from scipy.stats import circstd

def get_outputs(data_folder,conditions,column_names):
    results_all,conds_all,yTest_all,yTestPred_all = [],[],[],[]
    direcs = sorted(os.listdir(data_folder))
    current_direcs = [d for d in direcs if all(c in d for c in conditions)]
    for direc in current_direcs:
        files = sorted(os.listdir(os.path.join(data_folder, direc)))
        for file in files:
            file_path = os.path.join(data_folder, direc, file)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                    results_all.append(results[0])
                        
    df = pd.DataFrame(results_all,columns=column_names)
    df['R2_x']   = df['R2'].apply(lambda x: x[0])
    df['R2_y']   = df['R2'].apply(lambda x: x[1])
    df['R2_mn']  = df['R2'].apply(lambda x: x.mean())
    df['rho_x']  = df['rho'].apply(lambda x: x[0])
    df['rho_y']  = df['rho'].apply(lambda x: x[1])
    df['rho_mn']  = df['rho'].apply(lambda x: x.mean())
    df['R2sh_x']   = df['R2_shuf'].apply(lambda x: x[0])
    df['R2sh_y']   = df['R2_shuf'].apply(lambda x: x[1])
    df['R2sh_mn']  = df['R2_shuf'].apply(lambda x: x.mean())
    df['R2tr_x']   = df['R2_train'].apply(lambda x: x[0])
    df['R2tr_y']   = df['R2_train'].apply(lambda x: x[1])
    df['R2tr_mn']  = df['R2_train'].apply(lambda x: x.mean())

    return df

def get_outputs_wTraces(data_folder,conditions,column_names):
    results_all,conds_all,yTest_all,yTestPred_all = [],[],[],[]
    direcs = sorted(os.listdir(data_folder))
    current_direcs = [d for d in direcs if all(c in d for c in conditions)]
    for direc in current_direcs:
        files = sorted(os.listdir(os.path.join(data_folder, direc)))
        for file in files:
            file_path = os.path.join(data_folder, direc, file)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                if file.endswith('.pickle') and ('s29' in file) and ('repeat000.' in file or 'repeat00.' in file):
                    with open(file_path, 'rb') as f:
                        results,conds,y_test,y_test_predicted = pickle.load(f)
                        conds_all.append(conds)
                        results_all.append(results)
                        yTest_all.append(y_test)
                        yTestPred_all.append(y_test_predicted)
                elif file.endswith('.pickle'):
                    with open(file_path, 'rb') as f:
                        results = pickle.load(f)
                        results_all.append(results[0])
                        
    df = pd.DataFrame(results_all,columns=column_names)
    df['R2_x']   = df['R2'].apply(lambda x: x[0])
    df['R2_y']   = df['R2'].apply(lambda x: x[1])
    df['R2_mn']  = df['R2'].apply(lambda x: x.mean())
    df['rho_x']  = df['rho'].apply(lambda x: x[0])
    df['rho_y']  = df['rho'].apply(lambda x: x[1])
    df['rho_mn']  = df['rho'].apply(lambda x: x.mean())
    df['R2sh_x']   = df['R2_shuf'].apply(lambda x: x[0])
    df['R2sh_y']   = df['R2_shuf'].apply(lambda x: x[1])
    df['R2sh_mn']  = df['R2_shuf'].apply(lambda x: x.mean())
    df['R2tr_x']   = df['R2_train'].apply(lambda x: x[0])
    df['R2tr_y']   = df['R2_train'].apply(lambda x: x[1])
    df['R2tr_mn']  = df['R2_train'].apply(lambda x: x.mean())

    #df['num_bins'] =
    #df['inference_time'] = df['preProcess_time'] 

    '''
    df['R2N_x']  = df['R2_null'].apply(lambda x: x[0])
    df['R2N_y']  = df['R2_null'].apply(lambda x: x[1])
    df['R2N_mn']  = df['R2_null'].apply(lambda x: x.mean())
    df['rhoN_x'] = df['rho_null'].apply(lambda x: x[0])
    df['rhoN_y'] = df['rho_null'].apply(lambda x: x[1])
    df['rhoN_mn']  = df['rho_null'].apply(lambda x: x.mean())
    df = df.drop(columns=['R2','rho','R2_null','rho_null'])
    '''
    return df,conds_all,yTest_all,yTestPred_all

def get_outputs_tempTraces(data_folder,conditions,column_names):
    results_all,conds_all,yTest_all,yTestPred_all = [],[],[],[]
    direcs = sorted(os.listdir(data_folder))
    current_direcs = [d for d in direcs if all(c in d for c in conditions)]
    for direc in current_direcs:
        for file in sorted(os.listdir(data_folder+'/'+direc)):
            if file.endswith('.pickle') and 'repeat000' in file:
                with open(data_folder+'/'+direc+'/'+file, 'rb') as f:
                    results,conds,y_test,y_test_predicted = pickle.load(f)
                    conds_all.append(conds)
                    results_all.append(results)
                    yTest_all.append(y_test)
                    yTestPred_all.append(y_test_predicted)
            elif file.endswith('.pickle'):
                with open(data_folder+'/'+direc+'/'+file, 'rb') as f:
                    results = pickle.load(f)
                    results_all.append(results[0])
                        
    df = pd.DataFrame(results_all,columns=column_names)

    #df['num_bins'] =
    #df['inference_time'] = df['preProcess_time'] 

    '''
    df['R2N_x']  = df['R2_null'].apply(lambda x: x[0])
    df['R2N_y']  = df['R2_null'].apply(lambda x: x[1])
    df['R2N_mn']  = df['R2_null'].apply(lambda x: x.mean())
    df['rhoN_x'] = df['rho_null'].apply(lambda x: x[0])
    df['rhoN_y'] = df['rho_null'].apply(lambda x: x[1])
    df['rhoN_mn']  = df['rho_null'].apply(lambda x: x.mean())
    df = df.drop(columns=['R2','rho','R2_null','rho_null'])
    '''
    return df,conds_all,yTest_all,yTestPred_all


def get_outputsSweep(data_folder,conditions,column_names):
    results_all = []
    direcs = sorted(os.listdir(data_folder))
    current_direcs = [d for d in direcs if all(c in d for c in conditions)]
    for direc in current_direcs:
        for file in sorted(os.listdir(data_folder+'/'+direc)):
            if file.endswith('.pickle'):
                with open(data_folder+'/'+direc+'/'+file, 'rb') as f:
                    results = pickle.load(f)
                    for i in range(len(results)):
                        results_all.append(results[i])
                        
    df = pd.DataFrame(results_all,columns=column_names)
    #df2 = df1.loc[(df1['mean_R2'] < 1) & (df1['mean_R2'] > -1)]
    #df = pd.concat([df,df1],ignore_index=False)
    
    return df


def get_outputs_wUnits(result_dir,load_folder,units):
    df = pd.DataFrame() # Creates an empty list
    cnt = 0
    print('yay')
    for file in sorted(os.listdir(data_folder+result_dir)):
        cnt += 1
        if (cnt % 10) == 0:
            print(cnt)
        if file.endswith('.pickle'):
            with open(data_folder+result_dir+file, 'rb') as f:
                results,params_all,neurons_all,times_all = pickle.load(f)
                
                df1 = pd.DataFrame(results,columns=['sess','repeat','outerFold','nMT','nFEF','R2','rho'])
                df1['neuron_inds'] = neurons_all
                props = []
                for index, row in df1.iterrows():
                    n_inds = row['neuron_inds']
                    snr_mn = np.array([units['SNR'].iloc[[i]].values.item() for i in n_inds]).mean()
                    snr_sem = (np.array([units['SNR'].iloc[[i]].values.item() for i in n_inds]).std())/np.sqrt(len(n_inds))
                    pd_sem = np.rad2deg(circstd(np.deg2rad(np.array([units['PrefDirFit'].iloc[[i]].values.item() for i in n_inds]))))/np.sqrt(len(n_inds))
                    mfr_mn = np.array([units['mnFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).mean()
                    mfr_sem = (np.array([units['mnFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).std())/np.sqrt(len(n_inds))
                    vfr_mn = np.array([units['varFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).mean()
                    vfr_sem = (np.array([units['varFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).std())/np.sqrt(len(n_inds))
                    dm_mn = np.array([units['DepthMod'].iloc[[i]].values.item() for i in n_inds]).mean()
                    dm_sem = (np.array([units['DepthMod'].iloc[[i]].values.item() for i in n_inds]).std())/np.sqrt(len(n_inds))
                    di_mn = np.array([units['DI'].iloc[[i]].values.item() for i in n_inds]).mean()
                    di_sem = (np.array([units['DI'].iloc[[i]].values.item() for i in n_inds]).std())/np.sqrt(len(n_inds))
                    si_mn = np.array([units['SI'].iloc[[i]].values.item() for i in n_inds]).mean()
                    si_sem = (np.array([units['SI'].iloc[[i]].values.item() for i in n_inds]).std())/np.sqrt(len(n_inds))
                          
                    props.append([snr_mn,snr_sem,pd_sem,mfr_mn,mfr_sem,vfr_mn,vfr_sem,dm_mn,dm_sem,di_mn,di_sem,si_mn,si_sem])
                    
                df2 = pd.DataFrame(props,columns=['snr_mn','snr_sem','pd_sem','mfr_mn','mfr_sem','vfr_mn','vfr_sem','dm_mn','dm_sem','di_mn','di_sem','si_mn','si_sem'])
                df3 = pd.concat([df1,df2], axis=1, ignore_index=False)
                df = pd.concat([df,df3],ignore_index=False)
    return df

def print_df(DF):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(DF)

def make_DF(sessions):
    DF = pd.DataFrame()
    for s in range(len(sessions)):
        units = pd.read_csv(cwd+'/../datasets/units-pa{:0>2d}dir4A-pre500-post300.csv'.format(sessions[s]))
        data_dir = 's{:0>2d}-t0-d050-m00-o0-bn0-fo10-fi03-r0050/'.format(sessions[s])
        df = get_outputs(data_dir,data_folder,units)
        
        df['R2_norm'] = (df['R2'] - df['R2'].min()) / (df['R2'].max() - df['R2'].min())    
        DF = pd.concat([DF,df],ignore_index=True)
        
        bins = np.linspace(0.5, 6, num=15)
        mt = units[units['BrainArea']=='MT']['SNR'].values
        fef = units[units['BrainArea']=='FEF']['SNR'].values

    conditions = [
        (DF['nFEF'] == 0) & (DF['nMT'] !=0),
        (DF['nMT'] == 0) & (DF['nFEF'] !=0),
        (DF['nMT'] == DF['nFEF']), 
        (DF['nMT'] != 0) & (DF['nFEF'] !=0) & (DF['nMT'] != DF['nFEF'])
        ]
    values = ['mt only', 'fef only', 'mt=fef', 'mt+fef']
    DF['condition'] = np.select(conditions, values)
    DF['num_neurons'] = DF['nMT'] + DF['nFEF']

    DF.to_csv('out.csv',index=False)

