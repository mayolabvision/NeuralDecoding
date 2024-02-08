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

def get_neuronRepeats(s,t,*args):
    #[neurons_perRepeat,nm,nf] = neuronsSample.get_neuronRepeats(s,t,num_repeats,nn)

    sess,sess_nodt = helpers.get_session(s,t,1,1,1,1)
    units = pd.read_csv(cwd+'/datasets/units/units-'+sess_nodt+'.csv')

    if len(args) == 0:
        num_repeats = 1
        nm = (units['BrainArea'] == 'MT').sum() 
        nf = (units['BrainArea'] == 'FEF').sum()
    elif len(args) == 1:
        num_repeats = args[0]
        nm = (units['BrainArea'] == 'MT').sum()
        nf = (units['BrainArea'] == 'FEF').sum()
    elif len(args) == 2:
        num_repeats = args[0]
        nn = args[1]
        nm = 99
        nf = 99
    elif len(args) == 3:
        num_repeats,nm,nf = args

    if nm < 1000:
        if not os.path.isfile(cwd+'/datasets/dsplt/dsplt-'+sess_nodt+'-nm'+str(nm)+'-nf'+str(nf)+'-r'+str(num_repeats)+'.pickle'):

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
            with open(cwd+'/datasets/dsplt/dsplt-'+sess_nodt+'-nm'+str(nm)+'-nf'+str(nf)+'-r'+str(num_repeats)+'.pickle','wb') as f:
                pickle.dump(neurons_perRepeat,f)
        else:
            with open(cwd+'/datasets/dsplt/dsplt-'+sess_nodt+'-nm'+str(nm)+'-nf'+str(nf)+'-r'+str(num_repeats)+'.pickle','rb') as f:
                neurons_perRepeat = pickle.load(f,encoding='latin1')
    else:
        if not os.path.isfile(cwd+'/datasets/dsplt/dsplt-'+sess_nodt+'-nn'+str(nn)+'-r'+str(num_repeats)+'.pickle'):

            neurons_perRepeat = []
            for r in range(num_repeats):
                nn_inds = sorted(np.random.choice(units.index, nn, replace=False))
                neuron_inds = np.array((nn_inds))
               
                neurons_perRepeat.append(neuron_inds)
            with open(cwd+'/datasets/dsplt/dsplt-'+sess_nodt+'-nn'+str(nn)+'-r'+str(num_repeats)+'.pickle','wb') as f:
                pickle.dump(neurons_perRepeat,f)
        else:
            with open(cwd+'/datasets/dsplt/dsplt-'+sess_nodt+'-nn'+str(nn)+'-r'+str(num_repeats)+'.pickle','rb') as f:
                neurons_perRepeat = pickle.load(f,encoding='latin1')
        

    return neurons_perRepeat,nm,nf

