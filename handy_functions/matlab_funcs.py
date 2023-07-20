# Example of correctly formatting data

# Neural data should be a matrix of size "number of time bins" x "number of neurons", where each entry is the firing rate of a given neuron in a given time bin
#The output you are decoding should be a matrix of size "number of time bins" x "number of features you are decoding"
#In this example, we load Matlab data that contains

#The spike times of all neurons. In Matlab, "spike_times" is a cell of size "number of neurons" x 1. Within spike_times{i} is a vector containing all the spike times of neuron i.
#A continuous stream of the output variables. In this example, we are aiming to decode velocity. In Matlab, "vels" is a matrix of size "number of recorded time points" x 2 (x and y velocities were recorded) that contains the x and y velocity components at all time points. "vel_times" is a vector that states the time at all recorded time points.

# Import all necessary packages and functions
import numpy as np
from scipy import io
import pickle
from scipy import io
from scipy import stats
import sys
import os
import glob

from preprocessing_funcs import bin_spikes
from preprocessing_funcs import bin_output

data_folder     = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/datasets/'

def mat_to_pickle(filename,dt):
    dt = int(dt)

    # Load in variables from datafile
    data         =  io.loadmat(data_folder+'vars/'+filename)

    spike_times  =  data['spike_times'] # spike times of all neurons
    pos          =  data['pos'] # x and y eye positions
    vels         =  data['vels'] # x and y eye velocities
    acc          =  data['acc'] # x and y eye accelerations

    conditions   =  data['contConditions']

    out_times    =  data['vels_times'] # times at which velocities were recorded
    out_times    =  np.squeeze(out_times)

    # Use preprocessing functions to bin the data
    t_start            =  out_times[0] # time to start extracting data
    t_end              =  out_times[-1] # time to finish extracting data
    downsample_factor  =  1 # downsampling of output (to make binning go faster). 1 means no downsampling.

    spike_times  =  np.squeeze(spike_times)
    for i in range(spike_times.shape[0]):
        spike_times[i]  =  np.squeeze(spike_times[i])

    neural_data  =  bin_spikes(spike_times,dt,t_start,t_end)
    pos_binned   =  bin_output(pos,out_times,dt,t_start,t_end,downsample_factor)
    vel_binned   =  bin_output(vels,out_times,dt,t_start,t_end,downsample_factor)
    acc_binned   =  bin_output(acc,out_times,dt,t_start,t_end,downsample_factor)
    
    cond_binned  =  bin_output(conditions,out_times,dt,t_start,t_end,downsample_factor)

    with open(data_folder+'vars/'+filename[:-4]+'-dt'+str(dt)+'.pickle','wb') as f:
        pickle.dump([neural_data,pos_binned,vel_binned,acc_binned,cond_binned],f)

def pickle_allFiles(dt):
    vars_list = glob.glob(data_folder+'vars/vars-*-post300.mat')
    for i in range(len(vars_list)):
        print('{}/{}'.format(i,len(vars_list)))
        if os.path.isfile(vars_list[i][:-4]+'-dt'+str(dt)+'.pickle'):
            print('bitch it already exists')
        else:
            mat_to_pickle(vars_list[i][64:],dt)

