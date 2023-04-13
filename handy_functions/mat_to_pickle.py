# Example of correctly formatting data

# Neural data should be a matrix of size "number of time bins" x "number of neurons", where each entry is the firing rate of a given neuron in a given time bin
#The output you are decoding should be a matrix of size "number of time bins" x "number of features you are decoding"
#In this example, we load Matlab data that contains

#The spike times of all neurons. In Matlab, "spike_times" is a cell of size "number of neurons" x 1. Within spike_times{i} is a vector containing all the spike times of neuron i.
#A continuous stream of the output variables. In this example, we are aiming to decode velocity. In Matlab, "vels" is a matrix of size "number of recorded time points" x 2 (x and y velocities were recorded) that contains the x and y velocity components at all time points. "vel_times" is a vector that states the time at all recorded time points.

#shutup.please() # if you want to supress warnings, type $ pip install shutup into your terminal

# Import all necessary packages and functions
import numpy as np
from scipy import io
import pickle
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import sys

from preprocessing_funcs import bin_spikes
from preprocessing_funcs import bin_output


filename = sys.argv[1]
dt = int(sys.argv[2])

# Load in variables from datafile
folder       = '/Users/kendranoneman/Projects/mayo/NeuralDecoding/datasets/'
data         =  io.loadmat(folder+filename+'.mat')

spike_times  =  data['spike_times'] # spike times of all neurons
outputs         =  data['outputs'] # x and y eye velocities
out_times    =  data['out_times'] # times at which velocities were recorded
out_times    =  np.squeeze(out_times)

# Use preprocessing functions to bin the data
t_start            =  out_times[0] # time to start extracting data
t_end              =  out_times[-1] # time to finish extracting data
downsample_factor  =  1 # downsampling of output (to make binning go faster). 1 means no downsampling.

spike_times  =  np.squeeze(spike_times)
for i in range(spike_times.shape[0]):
    spike_times[i]  =  np.squeeze(spike_times[i])

neural_data  =  bin_spikes(spike_times,dt,t_start,t_end)
out_binned  =  bin_output(outputs,out_times,dt,t_start,t_end,downsample_factor)

# Pickle the file
data_folder='/Users/kendranoneman/Projects/mayo/NeuralDecoding/datasets/' #FOLDER YOU WANT TO SAVE THE DATA TO

with open(data_folder+filename+'-dt'+str(dt)+'.pickle','wb') as f:
    pickle.dump([neural_data,out_binned],f)
