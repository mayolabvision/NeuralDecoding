# Example of correctly formatting data

# Neural data should be a matrix of size "number of time bins" x "number of neurons", where each entry is the firing rate of a given neuron in a given time bin
#The output you are decoding should be a matrix of size "number of time bins" x "number of features you are decoding"
#In this example, we load Matlab data that contains

#The spike times of all neurons. In Matlab, "spike_times" is a cell of size "number of neurons" x 1. Within spike_times{i} is a vector containing all the spike times of neuron i.
#A continuous stream of the output variables. In this example, we are aiming to decode velocity. In Matlab, "vels" is a matrix of size "number of recorded time points" x 2 (x and y velocities were recorded) that contains the x and y velocity components at all time points. "vel_times" is a vector that states the time at all recorded time points.

# Import all necessary packages and functions
import numpy as np
from scipy import io
import time
import pickle
from scipy import io
from scipy import stats
import sys
import os
import glob

from preprocessing_funcs import bin_spikes
from preprocessing_funcs import bin_output
from preprocessing_funcs import get_spikes_with_history

cwd = os.getcwd()
data_folder = cwd+'/datasets/'

def mat_to_pickle(filename,dto,wi,dti,downsample_factor=1):
    dto = int(dto)
    dti = int(dti)
    df = int(downsample_factor)

    # Load in variables from datafile
    data         =  io.loadmat(data_folder+'vars/'+filename)
    out_times    =  data['vels_times'] # times at which velocities were recorded
    out_times    =  np.squeeze(out_times)
    out_times    =  out_times[wi-dto:]

    # Use preprocessing functions to bin the data
    t_start  =  out_times[0] # time to start extracting data
    t_end    =  out_times[-1] # time to finish extracting data

    if os.path.exists(data_folder+'/pickles/outs-'+filename[5:-4]+'-dto{:03d}-df{}.pickle'.format(dto,df)):
        with open(data_folder+'pickles/outs-'+filename[5:-4]+'-dto{:03d}-df{}.pickle'.format(dto,df),'rb') as f:
            pos_binned,vel_binned,acc_binned,cond_binned,out_edges,t1_elapsed=pickle.load(f,encoding='latin1')
        print('loaded outputs')
        
    else:
        print('pickling outputs')
        t1=time.time()
        pos          =  data['pos'] # x and y eye positions
        vels         =  data['vels'] # x and y eye velocities
        acc          =  data['acc'] # x and y eye accelerations
        conditions   =  data['contConditions']
        
        all_outputs = np.concatenate((pos[wi-dto:,:],vels[wi-dto:,:],acc[wi-dto:,:],conditions[wi-dto:,:]), axis=1)
        outs_binned,out_edges = bin_output(all_outputs,out_times,dto,t_start,t_end,downsample_factor=df,bins_predict=bins_predict)

        t1_elapsed = time.time()-t1

        pos_binned  =  outs_binned[:,0:2]
        vel_binned  =  outs_binned[:,2:4]
        acc_binned  =  outs_binned[:,4:6]
        cond_binned =  outs_binned[:,6:]

        with open(data_folder+'pickles/outs-'+filename[5:-4]+'-dto{:03d}-df{}.pickle'.format(dto,df),'wb') as f:
            pickle.dump([pos_binned,vel_binned,acc_binned,cond_binned,out_edges,t1_elapsed],f)
        print('pickled outputs')

    if os.path.exists(data_folder+'/pickles/ins-'+filename[5:-4]+'-dto{:03d}-wi{:03d}-dti{:03d}.pickle'.format(dto,wi,dti)):
        with open(data_folder+'pickles/ins-'+filename[5:-4]+'-dto{:03d}-wi{:03d}-dti{:03d}.pickle'.format(dto,wi,dti),'rb') as f:
            neural_data,t2_elapsed=pickle.load(f,encoding='latin1')
        print('loaded inputs')
    
    else:
        print('pickling inputs')
        t2=time.time()
        
        spike_times  =  data['spike_times'] # spike times of all neurons
        spike_times  =  np.squeeze(spike_times)
        for i in range(spike_times.shape[0]):
            spike_times[i]  =  np.squeeze(spike_times[i])

        out_edges_slided = out_edges - slide_ms
        neural_data = get_spikes_with_history(spike_times,wi,dti,out_edges_slided)
        
        t2_elapsed = time.time()-t2

        with open(data_folder+'pickles/ins-'+filename[5:-4]+'-dto{:03d}-wi{:03d}-dti{:03d}.pickle'.format(dto,wi,dti,slide_ms),'wb') as f:
            pickle.dump([neural_data,t2_elapsed],f)
        print('pickled inputs')

    time_elapsed = t1_elapsed+t2_elapsed

    return neural_data,pos_binned,vel_binned,acc_binned,cond_binned,time_elapsed


