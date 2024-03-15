############### IMPORT PACKAGES ##################

import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter
import statsmodels.api as sm
import math
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn import linear_model 
from sklearn.svm import SVR #For support vector regression (SVR)
from sklearn.svm import SVC #For support vector classification (SVM)
import xgboost as xgb #For xgboost
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import OrthogonalMatchingPursuit
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from keras.regularizers import l2

##################### DECODER FUNCTIONS ##########################

##################### SIMPLE RECURRENT NEURAL NETWORK ##########################

class SimpleRNNRegression_multiInput_singleOutput(object):

    """
    Class for the simple recurrent neural network decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1,patience=5):
         self.units=units
         self.dropout=dropout
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers
         self.patience=patience


    def fit(self,X,y,test_size=0.1):

        """
        Train SimpleRNN Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        model=Sequential() #Declare model
        model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout,activation='relu')) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='min')

        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_data=(X_val, y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                  callbacks=[early_stopping])
        self.model=model

    def predict(self,X_test):

        """
        Predict outcomes using trained SimpleRNN Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test) #Make predictions
        
        return y_test_predicted



#################### LONG SHORT TERM MEMORY (LSTM) DECODER ##########################
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


class LSTMRegression_multiInput_singleOutput(object):
    """
    Class for the LSTM Decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch

    batch_size: integer, optional, default 128
        Batch size for training

    workers: integer, optional, default 1
        Number of workers for data loading during training
    """

    def __init__(self, mt_units=400, fef_units=400, mt_dropout=0, fef_dropout=0, lr=0.001, num_epochs=10, verbose=0, batch_size=128, workers=1, patience=3):
        self.mt_units = mt_units
        self.fef_units = fef_units
        self.mt_dropout = mt_dropout
        self.fef_dropout = fef_dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.workers = workers
        self.patience = patience
        self.model = None

    def fit(self, X_mt, X_fef, y, test_size=0.2):
        """
        Train LSTM Decoder

        Parameters
        ----------
        X_mt: numpy 3d array of shape [n_samples, n_time_bins, n_mt_neurons]
            This is the neural data from MT neurons.

        X_fef: numpy 3d array of shape [n_samples, n_time_bins, n_fef_neurons]
            This is the neural data from FEF neurons.

        y: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        X_mt_train, X_mt_val, X_fef_train, X_fef_val, y_train, y_val = train_test_split(X_mt, X_fef, y, test_size=test_size, shuffle=True)

        # Define the LSTM model
        input_mt = Input(shape=(X_mt_train.shape[1], X_mt_train.shape[2]), name='mt_input')
        input_fef = Input(shape=(X_fef_train.shape[1], X_fef_train.shape[2]), name='fef_input')

        lstm_layer_mt = LSTM(self.mt_units, dropout=self.mt_dropout, name='lstm_mt')(input_mt)  # Remove return_sequences=True
        lstm_layer_fef = LSTM(self.fef_units, dropout=self.fef_dropout, name='lstm_fef')(input_fef)  # Remove return_sequences=True

        concatenated_layers = concatenate([lstm_layer_mt, lstm_layer_fef], name='concatenated_features')  # Concatenate the outputs of both LSTM layers

        output_layer = Dense(y_train.shape[1], name='output')(concatenated_layers)

        model = Model(inputs=[input_mt, input_fef], outputs=output_layer)

        # Compile the model
        model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=['accuracy']) #Set loss function and optimizer

        # Plot model architecture
        plot_model(model, to_file='LSTMDecoder_miso.png', show_shapes=True)

        # Fit the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='min')

        model.fit([X_mt_train, X_fef_train], y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_data=([X_mt_val, X_fef_val], y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                  callbacks=[early_stopping])

        self.model = model

    def predict(self, X_mt_test, X_fef_test):
        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_mt_test: numpy 3d array of shape [n_samples, n_time_bins, n_mt_neurons]
            This is the neural data from MT neurons being used to predict outputs.

        X_fef_test: numpy 3d array of shape [n_samples, n_time_bins, n_fef_neurons]
            This is the neural data from FEF neurons being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict([X_mt_test, X_fef_test])  # Make predictions
        return y_test_predicted


######### ALIASES for Regression ########

#LinearDecoder = LinearRegression
LSTMDecoder_miso = LSTMRegression_multiInput_singleOutput
RNNDecoder_miso = SimpleRNNRegression_multiInput_singleOutput
