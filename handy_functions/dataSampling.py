import numpy as np
import sys
import pickle
import pandas as pd
import os.path
import os

cwd = os.getcwd()
sys.path.append(cwd+"/handy_functions") # go to parent dir

from preprocessing_funcs import get_spikes_with_history
from sklearn.model_selection import KFold
import helpers

def get_neuronRepeats(sess_nodt,nn=99,nm=99,nf=99,num_repeats=1):
    units = pd.read_csv(cwd+'/datasets/units/units-'+sess_nodt+'.csv')

    if nn==99 and nm==99:
        nm = (units['BrainArea'] == 'MT').sum()
        nf = (units['BrainArea'] == 'FEF').sum()
        nn = nm + nf
    elif nn==99 and nm!=99:
        nn = nm + nf
    elif nn!=99 and nm==99:
        nm = 99
        nf = 99

    if nm < 99:
        file_path = os.path.join(cwd, 'datasets', 'dsplt', f"dsplt-{sess_nodt}-nm{nm}-nf{nf}-r{num_repeats}.pickle")
        condition = 'nm'
    else:
        file_path = os.path.join(cwd, 'datasets', 'dsplt', f"dsplt-{sess_nodt}-nn{nn}-r{num_repeats}.pickle")
        condition = 'nn'

    if not os.path.isfile(file_path):
        print('sampling neurons')
        neurons_perRepeat = []
        for r in range(num_repeats):
            if condition == 'nm':
                mt_inds = sorted(np.random.choice(units[units['BrainArea'] == 'MT'].index, nm, replace=False))
                fef_inds = sorted(np.random.choice(units[units['BrainArea'] == 'FEF'].index, nf, replace=False))

                if nm == 0:
                    neuron_inds = np.array(fef_inds)
                elif nf == 0:
                    neuron_inds = np.array(mt_inds)
                else:
                    neuron_inds = np.concatenate((mt_inds, fef_inds))
            else:  # condition == 'nn'
                nn_inds = sorted(np.random.choice(units.index, nn, replace=False))
                neuron_inds = np.array(nn_inds)

            neurons_perRepeat.append(neuron_inds)

        with open(file_path, 'wb') as f:
            pickle.dump(neurons_perRepeat, f)
    else:
        with open(file_path, 'rb') as f:
            neurons_perRepeat = pickle.load(f, encoding='latin1')


    return neurons_perRepeat,nn,nm,nf

def get_trainSection(sess_nodt, num_observations, fold, tp=1.0, num_repeats=1):
    file_path = os.path.join(cwd, 'datasets', 'dsplt', f"tsplt-{sess_nodt}-tp{int(tp*100)}-fo{fold}-r{num_repeats}.pickle")
  
    if not os.path.isfile(file_path):
        print('sampling observations')
        inds = np.arange(num_observations)

        observations_perRepeat = []
        for r in range(num_repeats):
            sampled_indices = sorted(np.random.choice(inds, int(num_observations*tp), replace=False))
            observations_perRepeat.append(sampled_indices)

        with open(file_path, 'wb') as f:
            pickle.dump(observations_perRepeat, f)
    else:
        with open(file_path, 'rb') as f:
            observations_perRepeat = pickle.load(f, encoding='latin1')
   
    return observations_perRepeat
