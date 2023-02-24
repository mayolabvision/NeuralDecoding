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

from preprocessing_funcs import bin_spikes
from preprocessing_funcs import bin_output

# Load in variables from datafile
folder       = '/Users/kendranoneman/Projects/mayo/data/neural-decoding/preprocessed/'
data         =  io.loadmat(folder+'MTFEF-pa29dir4A-1600ms.mat')

spike_times  =  data['spike_times'] # spike times of all neurons
vels         =  data['vels'] # x and y eye velocities
vel_times    =  data['vel_times'] # times at which velocities were recorded
vel_times    =  np.squeeze(vel_times)

# Use preprocessing functions to bin the data

dt                 =  50 # size of time bins (in seconds)
t_start            =  vel_times[0] # time to start extracting data
t_end              =  vel_times[-1] # time to finish extracting data
downsample_factor  =  1 # downsampling of output (to make binning go faster). 1 means no downsampling.

spike_times  =  np.squeeze(spike_times)
for i in range(spike_times.shape[0]):
    spike_times[i]  =  np.squeeze(spike_times[i])

neural_data  =  bin_spikes(spike_times,dt,t_start,t_end)
vels_binned  =  bin_output(vels,vel_times,dt,t_start,t_end,downsample_factor)

# Pickle the file
data_folder='/Users/kendranoneman/Projects/mayo/NeuralDecoding/datasets/' #FOLDER YOU WANT TO SAVE THE DATA TO

with open(data_folder+'MTFEF-pa29dir4A-1600ms.pickle','wb') as f:
    pickle.dump([neural_data,vels_binned],f)
