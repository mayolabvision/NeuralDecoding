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

def get_neuronRepeats(line):
    s = int(line[1])
    t = int(line[2])
    d = int(line[3])
    m = int(line[4])
    o = int(line[5])
    nm = int(line[6])
    nf = int(line[7])
    bn = int(line[8])
    fo = int(line[9])
    fi = int(line[10]) 
    num_repeats = int(line[11])
    sess,sess_nodt = helpers.get_session(s,t,d)

    if not os.path.isfile(cwd+'/datasets/dsplt/dsplt-'+sess+'-nm'+str(nm)+'-nf'+str(nf)+'-r'+str(num_repeats)+'.pickle'):
        units = pd.read_csv(cwd+'/datasets/units/units-'+sess_nodt+'.csv')

        neurons_perRepeat = []
        for r in range(num_repeats):
            mt_inds = sorted(np.random.choice(units[units['BrainArea'] == 'MT'].index, nm, replace=False))
            fef_inds = sorted(np.random.choice(units[units['BrainArea'] == 'FEF'].index, nf, replace=False))

            if nm==0:
                neuron_inds = np.array((fef_inds))
            elif nf==0:
                neuron_inds = np.array((mt_inds))
            else:
                neuron_inds = sorted(np.concatenate((np.array((mt_inds)),np.array((fef_inds)))))
           
            neurons_perRepeat.append(neuron_inds)
    
        with open(cwd+'/datasets/dsplt/dsplt-'+sess+'-nm'+str(nm)+'-nf'+str(nf)+'-r'+str(num_repeats)+'.pickle','wb') as f:
            pickle.dump(neurons_perRepeat,f)
    else:
        with open(cwd+'/datasets/dsplt/dsplt-'+sess+'-nm'+str(nm)+'-nf'+str(nf)+'-r'+str(num_repeats)+'.pickle','rb') as f:
            neurons_perRepeat = pickle.load(f,encoding='latin1')

    return neurons_perRepeat

