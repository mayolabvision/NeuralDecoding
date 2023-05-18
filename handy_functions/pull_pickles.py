import pandas as pd
import numpy as np
from scipy import io
import os

data_folder = '/Users/kendranoneman/Projects/mayo/data/neural-decoding/outpickles/'

def get_outputs(result_dir,load_folder,units):
    df = pd.DataFrame() # Creates an empty list
    cnt = 0
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
                    snr_var = np.array([units['SNR'].iloc[[i]].values.item() for i in n_inds]).var()
                    pd_var = np.array([units['PrefDirFit'].iloc[[i]].values.item() for i in n_inds]).var()
                    mfr_mn = np.array([units['mnFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).mean()
                    mfr_var = np.array([units['mnFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).var()
                    vfr_mn = np.array([units['varFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).mean()
                    vfr_var = np.array([units['varFR_bestDir'].iloc[[i]].values.item() for i in n_inds]).var()
                    dm_mn = np.array([units['DepthMod'].iloc[[i]].values.item() for i in n_inds]).mean()
                    dm_var = np.array([units['DepthMod'].iloc[[i]].values.item() for i in n_inds]).var()
                    di_mn = np.array([units['DI'].iloc[[i]].values.item() for i in n_inds]).mean()
                    di_var = np.array([units['DI'].iloc[[i]].values.item() for i in n_inds]).var()
                    si_mn = np.array([units['SI'].iloc[[i]].values.item() for i in n_inds]).mean()
                    si_var = np.array([units['SI'].iloc[[i]].values.item() for i in n_inds]).var()
                          
                    props.append([snr_mn,snr_var,pd_var,mfr_mn,mfr_var,vfr_mn,vfr_var,dm_mn,dm_var,di_mn,di_var,si_mn,si_var])
                    
                df2 = pd.DataFrame(props,columns=['snr_mn','snr_var','pd_var','mfr_mn','mfr_var','vfr_mn','vfr_var','dm_mn','dm_var','di_mn','di_var','si_mn','si_var'])
                df3 = pd.concat([df1,df2], axis=1, ignore_index=False)
                df = pd.concat([df,df3],ignore_index=False)
    return df

def print_df(DF):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(DF)
