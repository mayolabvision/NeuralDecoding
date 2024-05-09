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
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##################### DECODER FUNCTIONS ##########################

##################### WIENER FILTER ##########################
from scipy.signal import wiener
from sklearn import linear_model
import numpy as np


class WienerFilterRegression(object):

    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This class leverages the Wiener filter in combination with scikit-learn linear regression.
    """

    def __init__(self):
        self.model = None

    def fit(self, X_flat_train, y_train, tb=0):
        """
        Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # Apply the Wiener filter to the input data
        X_filtered_train = self.apply_wiener_filter(X_flat_train)

        self.model = linear_model.LinearRegression() # initialize linear regression model
        self.model.fit(X_filtered_train, y_train) # train the model

    def predict(self, X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs
        """

        # Apply the Wiener filter to the input data
        X_filtered_test = self.apply_wiener_filter(X_flat_test)

        # Make predictions using the trained linear regression model
        y_test_predicted = self.model.predict(X_filtered_test)

        return y_test_predicted

    def apply_wiener_filter(self, X):
        """
        Apply the Wiener filter to the input data

        Parameters
        ----------
        X: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.

        Returns
        -------
        X_filtered: numpy 2d array of shape [n_samples, n_features]
            The filtered neural data using the Wiener filter
        """
        # Apply the Wiener filter to each feature independently
        X_filtered = np.apply_along_axis(self.wiener_with_warnings, axis=0, arr=X)

        return X_filtered

    def wiener_with_warnings(self, x):
        """
        Apply the Wiener filter to a single feature with warnings handling

        Parameters
        ----------
        x: numpy 1d array
            A single feature of the neural data.

        Returns
        -------
        x_filtered: numpy 1d array
            The filtered feature using the Wiener filter
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            x_filtered = wiener(x)
        return x_filtered


##################### WIENER CASCADE ##########################

from scipy.signal import wiener
from sklearn import linear_model
import numpy as np

class WienerCascadeRegression(object):

    """
    Class for the Wiener Cascade Decoder

    Parameters
    ----------
    degree: integer, optional, default 3
        The degree of the polynomial used for the static nonlinearity
    """

    def __init__(self, degree=3, patience=5):
         self.degree = degree
         self.patience = patience
         self.models = []

    def fit(self, X_flat, y, test_size=0.1):
        """
        Train Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted

        test_size: float or int, optional, default 0.2
            Represents the proportion of the dataset to include in the test split. 
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. 
            If int, represents the absolute number of test samples.

        patience: int, optional, default 5
            Number of epochs to wait before early stopping if validation loss doesn't improve
        """

        X_flat_train, X_flat_val, y_train, y_val = train_test_split(X_flat, y, test_size=test_size, shuffle=False)

        X_filtered_train = self.apply_wiener_filter(X_flat_train)
        X_filtered_val = self.apply_wiener_filter(X_flat_val)
        
        num_outputs = y_train.shape[1]  # Number of outputs
        for i in range(num_outputs):  # Loop through outputs
            # Fit linear portion of model
            regr = linear_model.LinearRegression()  # Call the linear portion of the model "regr"

            # Perform early stopping based on validation loss
            best_val_loss = float('inf')
            epochs_without_improvement = 0

            while epochs_without_improvement < self.patience:
                regr.fit(X_filtered_train, y_train[:, i])  # Fit linear
                y_train_predicted_linear = regr.predict(X_filtered_train)  # Get outputs of linear portion of model

                # Fit nonlinear portion of model on the filtered linear predictions
                p = np.polyfit(y_train_predicted_linear, y_train[:, i], self.degree)

                # Predict on validation set
                y_val_predicted_linear = regr.predict(X_filtered_val)  # Get predictions on the linear portion of the model
                y_val_predicted = np.polyval(p, y_val_predicted_linear)

                # Calculate validation loss
                val_loss = mean_squared_error(y_val[:, i], y_val_predicted)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            # Add model for this output (both linear and nonlinear parts) to the list "models"
            self.models.append([regr, p])

    def predict(self, X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs
        """

        X_filtered_test = self.apply_wiener_filter(X_flat_test)
        num_outputs = len(self.models)  # Number of outputs being predicted

        y_test_predicted = np.empty([X_flat_test.shape[0], num_outputs])  # Initialize matrix that contains predicted outputs
        for i in range(num_outputs):  # Loop through outputs
            [regr, p] = self.models[i]  # Get the linear (regr) and nonlinear (p) portions of the trained model

            # Predictions on test set
            y_test_predicted_linear = regr.predict(X_filtered_test)  # Get predictions on the linear portion of the model

            # Run the linear predictions through the nonlinearity to get the final predictions
            y_test_predicted[:, i] = np.polyval(p, y_test_predicted_linear)

        return y_test_predicted

    def apply_wiener_filter(self, X):
        """
        Apply the Wiener filter to the input data

        Parameters
        ----------
        X: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.

        Returns
        -------
        X_filtered: numpy 2d array of shape [n_samples, n_features]
            The filtered neural data using the Wiener filter
        """
        # Apply the Wiener filter to each feature independently
        X_filtered = np.apply_along_axis(self.wiener_with_warnings, axis=0, arr=X)

        return X_filtered

    def wiener_with_warnings(self, x):
        """
        Apply the Wiener filter to a single feature with warnings handling

        Parameters
        ----------
        x: numpy 1d array
            A single feature of the neural data.

        Returns
        -------
        x_filtered: numpy 1d array
            The filtered feature using the Wiener filter
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            x_filtered = wiener(x)
        return x_filtered

##################### EXTREME GRADIENT BOOSTING (XGBOOST) ##########################
from sklearn.inspection import permutation_importance

class XGBoostRegression(object):

    """
    Class for the XGBoost Decoder

    Parameters
    ----------
    max_depth: integer, optional, default=3
        the maximum depth of the trees

    num_round: integer, optional, default=300
        the number of trees that are fit

    eta: float, optional, default=0.3
        the learning rate

    gpu: integer, optional, default=-1
        if the gpu version of xgboost is installed, this can be used to select which gpu to use
        for negative values (default), the gpu is not used
    """

    def __init__(self,max_depth=3,num_round=300,eta=0.3,subsample=1,gpu=-1,workers=-1,patience=5):
        self.max_depth=max_depth
        self.num_round=num_round
        self.eta=eta
        self.subsample=subsample
        self.gpu=gpu
        self.nthread=workers
        self.patience=patience

    def fit(self,X_flat,y,test_size=0.1):

        """
        Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted

        X_flat_valid: numpy 2d array of shape [n_samples_valid, n_features]
            Validation set for neural data
        
        y_valid: numpy 2d array of shape [n_samples_valid, n_outputs]
            Validation set for outputs
        """

        X_flat_train, X_flat_val, y_train, y_val = train_test_split(X_flat, y, test_size=test_size, shuffle=False)
        num_outputs=y_train.shape[1] #Number of outputs

        #Set parameters for XGBoost
        param = {'objective': "reg:linear", #for linear output
            'eval_metric': "logloss", #loglikelihood loss
            'max_depth': self.max_depth, #this is the only parameter we have set, it's one of the way or regularizing
            'eta': self.eta,
            'subsample': self.subsample,
            'seed': 2925, #for reproducibility
            'silent': True,
            'verbosity' : 0,
            'nthread':self.nthread}

#if self.gpu<0:
#param['nthread'] = -1 #with -1 it will use all available threads
#else:
#param['gpu_id']=self.gpu
#param['updater']='grow_gpu'

        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            dtrain = xgb.DMatrix(X_flat_train, label=y_train[:,y_idx]) #Put in correct format for XGB
            dval = xgb.DMatrix(X_flat_val, label=y_val[:,y_idx]) #Put in correct format for XGB

            evallist = [(dtrain, 'train'), (dval, 'eval')]
            bst = xgb.train(param, dtrain, self.num_round, evals=evallist, early_stopping_rounds=self.patience, verbose_eval=False) #Train model
            
            models.append(bst) #Add fit model to list of models

        self.model=models

    def predict(self,X_flat_test):

        """
        Predict outcomes using trained XGBoost Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        dtest = xgb.DMatrix(X_flat_test) #Put in XGB format
        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            bst=self.model[y_idx] #Get fit model for this output
            y_test_predicted[:,y_idx] = bst.predict(dtest) #Make prediction
        return y_test_predicted

##################### SUPPORT VECTOR REGRESSION ##########################
from sklearn.preprocessing import StandardScaler

class SVRegression(object):

    """
    Class for the Support Vector Regression (SVR) Decoder
    This simply leverages the scikit-learn SVR

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iterations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self,max_iter=-1,C=3.0,kernel='rbf',patience=5):
        self.max_iter=max_iter
        self.C=C
        self.kernel=kernel
        self.scaler = StandardScaler()
        self.patience = patience
        return


    def fit(self,X_flat,y,test_size=0.1):

        """
        Train SVR Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        
        X_flat_train, X_flat_val, y_train, y_val = train_test_split(X_flat, y, test_size=test_size, shuffle=False)

        num_outputs = y_train.shape[1]  # Number of outputs
        models = []  # Initialize list of models (there will be a separate model for each output)

        X_scaled_train = self.scaler.fit_transform(X_flat_train)
        X_scaled_val = self.scaler.transform(X_flat_val)

        for y_idx in range(num_outputs):  # Loop through outputs
            model = SVR(C=self.C, max_iter=self.max_iter, kernel=self.kernel)  # Initialize SVR model
    
            best_val_loss = float('inf')
            patience_count = 0
            best_model = None

            while patience_count < self.patience:
                model.fit(X_scaled_train, y_train[:, y_idx])  # Train the model
                y_val_pred = model.predict(X_scaled_val)  # Predict on validation set
                val_loss = mean_squared_error(y_val[:, y_idx], y_val_pred)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                    best_model = model
                else:
                    patience_count += 1

            models.append(model)  # Add fit model to list of models

        self.model = models
        
    def predict(self,X_flat_test):

        """
        Predict outcomes using trained SVR Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix of predicted outputs
        for y_idx in range(num_outputs): #Loop through outputs
            model=self.model[y_idx] #Get fit model for that output
            y_test_predicted[:,y_idx]=model.predict(X_flat_test) #Make predictions
        return y_test_predicted

    def get_margin_width(self):
        # Calculate margin width for each output
        margin_widths = []
        for coefficients in self.coefficients:
            # Calculate margin width as 1 / ||w||, where w is the weight vector (coefficients)
            weight_vector_norm = np.linalg.norm(coefficients)
            margin_width = 1.0 / weight_vector_norm
            margin_widths.append(margin_width)
        return margin_widths


##################### DENSE (FULLY-CONNECTED) NEURAL NETWORK ##########################

class DenseNNRegression(object):

    """
    Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------

    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, repsectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self,units=400,lr=0.001,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1,patience=5):
         self.dropout=dropout
         self.lr = lr
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers
         self.patience=patience

         #If "units" is an integer, put it in the form of a vector
         try: #Check if it's a vector
             units[0]
         except: #If it's not a vector, create a vector of the number of units for each layer
             units=[units]
         self.units=units

         #Determine the number of hidden layers (based on "units" that the user entered)
         self.num_layers=len(units)

    def fit(self,X_flat,y,test_size=0.1):

        """
        Train DenseNN Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        X_flat_train, X_flat_val, y_train, y_val = train_test_split(X_flat, y, test_size=test_size, shuffle=False)

        model=Sequential() #Declare model
        #Add first hidden layer
        model.add(Dense(self.units[0],input_dim=X_flat_train.shape[1])) #Add dense layer
        model.add(Activation('relu')) #Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout!=0: model.add(Dropout(self.dropout))  #Dropout some units if proportion of dropout != 0

        #Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers-1): #Loop through additional layers
            model.add(Dense(self.units[layer+1])) #Add dense layer
            model.add(Activation('relu')) #Add nonlinear (tanh) activation
            if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units if proportion of dropout != 0

        #Add dense connections to all outputs
        model.add(Dense(y_train.shape[1])) #Add final dense layer (connected to outputs)

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_flat_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='min')

        model.fit(X_flat_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_data=(X_flat_val, y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                  callbacks=[early_stopping])

        self.model=model

    def predict(self,X_flat_test):

        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted


##################### SIMPLE RECURRENT NEURAL NETWORK ##########################

class SimpleRNNRegression(object):

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

    def __init__(self,units=400,lr=0.001,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1,patience=5):
         self.units=units
         self.lr = lr
         self.dropout=dropout
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers
         self.patience=patience


    def fit(self,X,y,tb=0,test_size=0.1):

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
        model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=['accuracy']) #Set loss function and optimizer
        
        # Plot model architecture
        plot_model(model, to_file='RNNDecoder.png', show_shapes=True)
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='min')
        
        if tb==1:
            tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                      validation_data=(X_val, y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                      callbacks=[early_stopping,tensorboard_callback])
            model.summary()
        else:
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



##################### GATED RECURRENT UNIT (GRU) DECODER ##########################

class GRURegression(object):

    """
    Class for the gated recurrent unit (GRU) decoder

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

    def __init__(self,units=400,lr=0.001,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1,patience=5):
         self.units=units
         self.lr = lr
         self.dropout=dropout
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers
         self.patience=patience


    def fit(self,X,y,test_size=0.1):

        """
        Train GRU Decoder

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
        #Add recurrent layer
        #if keras_v1:
        #    model.add(GRU(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        #else:
        model.add(GRU(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout))
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=['accuracy']) #Set loss function and optimizer
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='min')

        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_data=(X_val, y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                  callbacks=[early_stopping])
        self.model=model

    def predict(self,X_test):

        """
        Predict outcomes using trained GRU Decoder

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
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


class LSTMRegression(object):
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

    def __init__(self, units=400, lr=0.001, dropout=0, num_epochs=10, verbose=0, batch_size=128, workers=1,patience=3):
        self.units = units
        self.lr = lr
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.workers = workers
        self.patience = patience
        self.model = None

    def fit(self, X, y, tb=0, test_size=0.1):
        """
        Train LFADS Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples, n_time_bins, n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)

        # Define the LFADS model
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_layer = LSTM(self.units, dropout=self.dropout)(input_layer)  # Remove return_sequences=True
        output_layer = Dense(y_train.shape[1])(lstm_layer)  # Adjust for y_train shape

        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=['accuracy']) #Set loss function and optimizer

        # Plot model architecture
        plot_model(model, to_file='LSTMDecoder.png', show_shapes=True)
        
        # Fit the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose, mode='min')
        if tb==1:
            tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                      validation_data=(X_val, y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                      callbacks=[early_stopping,tensorboard_callback])
            model.summary()
        else:
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.num_epochs,
                      validation_data=(X_val, y_val), verbose=self.verbose, workers=self.workers, use_multiprocessing=True,
                      callbacks=[early_stopping])
            

        self.model = model
        
    def predict(self, X_test):
        """
        Predict outcomes using trained LFADS Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples, n_time_bins, n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)  # Make predictions
        return y_test_predicted


##################### KALMAN FILTER ##########################

class KalmanFilterRegression(object):

    """
    Class for the Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Kalman filter for neural decoding is based on that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
    with the exception of the addition of the parameter C.
    The original implementation has previously been coded in Matlab by Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
    """

    def __init__(self,C=1):
        self.C=C


    def fit(self,X_kf_train,y_train):

        """
        Train Kalman Filter Decoder

        Parameters
        ----------
        X_kf_train: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples(i.e. timebins), n_outputs]
            This is the outputs that are being predicted
        """

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al, 2003):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_train.T)
        Z=np.matrix(X_kf_train.T)

        #number of time bins
        nt=X.shape[1]

        #Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        #In our case, this is the transition from one kinematic state to the next
        X2 = X[:,1:]
        X1 = X[:,0:nt-1]
        A=X2*X1.T*inv(X1*X1.T) #Transition matrix
        W=(X2-A*X1)*(X2-A*X1).T/(nt-1)/self.C #Covariance of transition matrix. Note we divide by nt-1 since only nt-1 points were used in the computation (that's the length of X1 and X2). We also introduce the extra parameter C here.

        #Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        #In our case, this is the transformation from kinematics to spikes
        H = Z*X.T*(inv(X*X.T)) #Measurement matrix
        Q = ((Z - H*X)*((Z - H*X).T)) / nt #Covariance of measurement matrix
        params=[A,W,H,Q]
        self.model=params

        return params

    def predict(self,X_kf_test,y_test):

        """
        Predict outcomes using trained Kalman Filter Decoder

        Parameters
        ----------
        X_kf_test: numpy 2d array of shape [n_samples(i.e. timebins) , n_neurons]
            This is the neural data in Kalman filter format.

        y_test: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The actual outputs
            This parameter is necesary for the Kalman filter (unlike other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples(i.e. timebins),n_outputs]
            The predicted outputs
        """

        #Extract parameters
        A,W,H,Q=self.model

        #First we'll rename and reformat the variables to be in a more standard kalman filter nomenclature (specifically that from Wu et al):
        #xs are the state (here, the variable we're predicting, i.e. y_train)
        #zs are the observed variable (neural data here, i.e. X_kf_train)
        X=np.matrix(y_test.T)
        Z=np.matrix(X_kf_test.T)

        #Initializations
        num_states=X.shape[0] #Dimensionality of the state
        states=np.empty(X.shape) #Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m=np.matrix(np.zeros([num_states,num_states]))
        P=np.matrix(np.zeros([num_states,num_states]))
        state=X[:,0] #Initial state
        states[:,0]=np.copy(np.squeeze(state))

        #Get predicted state for every time bin
        kalman_gains = []

# Get predicted state for every time bin
        for t in range(X.shape[1]-1):
            # Do first part of state update - based on transition matrix
            P_m = A * P * A.T + W
            state_m = A * state

            # Do second part of state update - based on measurement matrix
            K = P_m * H.T * inv(H * P_m * H.T + Q)  # Calculate Kalman gain

            # Append the Kalman gain to the array
            kalman_gains.append(K)

            P = (np.matrix(np.eye(num_states)) - K * H) * P_m
            state = state_m + K * (Z[:, t+1] - H * state_m)
            states[:, t+1] = np.squeeze(state)  # Record state at the timestep

        y_test_predicted = states.T
 
        return y_test_predicted, kalman_gains

#GLM helper function for the NaiveBayesDecoder
def glm_run(Xr, Yr, X_range):

    X2 = sm.add_constant(Xr)

    poiss_model = sm.GLM(Yr, X2, family=sm.families.Poisson())
    try:
        glm_results = poiss_model.fit()
        Y_range=glm_results.predict(sm.add_constant(X_range))
    except np.linalg.LinAlgError:
        print("\nWARNING: LinAlgError")
        Y_range=np.mean(Yr)*np.ones([X_range.shape[0],1])

    return Y_range


class NaiveBayesRegression(object):

    """
    Class for the Naive Bayes Decoder

    Parameters
    ----------
    encoding_model: string, default='quadratic'
        what encoding model is used

    res:int, default=100
        resolution of predicted values
        This is the number of bins to divide the outputs into (going from minimum to maximum)
        larger values will make decoding slower
    """

    def __init__(self,encoding_model='quadratic',res=100):
        self.encoding_model=encoding_model
        self.res=res
        return

    def fit(self,X_b_train,y_train):

        """
        Train Naive Bayes Decoder

        Parameters
        ----------
        X_b_train: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """

        #### FIT TUNING CURVE ####
        #First, get the output values (x/y position or velocity) that we will be creating tuning curves over
        #Create the range for x and y (position/velocity) values
        input_x_range=np.arange(np.min(y_train[:,0]),np.max(y_train[:,0])+.01,np.round((np.max(y_train[:,0])-np.min(y_train[:,0]))/self.res))
        input_y_range=np.arange(np.min(y_train[:,1]),np.max(y_train[:,1])+.01,np.round((np.max(y_train[:,1])-np.min(y_train[:,1]))/self.res))
        #Get all combinations of x/y values
        input_mat=np.meshgrid(input_x_range,input_y_range)
        #Format so that all combinations of x/y values are in 2 columns (first column x, second column y). This is called "input_xy"
        xs=np.reshape(input_mat[0],[input_x_range.shape[0]*input_y_range.shape[0],1])
        ys=np.reshape(input_mat[1],[input_x_range.shape[0]*input_y_range.shape[0],1])
        input_xy=np.concatenate((xs,ys),axis=1)

        #If quadratic model:
        #   -make covariates have squared components and mixture of x and y
        #   -do same thing for "input_xy", which are the values for creating the tuning curves
        if self.encoding_model=='quadratic':
            input_xy_modified=np.empty([input_xy.shape[0],5])
            input_xy_modified[:,0]=input_xy[:,0]**2
            input_xy_modified[:,1]=input_xy[:,0]
            input_xy_modified[:,2]=input_xy[:,1]**2
            input_xy_modified[:,3]=input_xy[:,1]
            input_xy_modified[:,4]=input_xy[:,0]*input_xy[:,1]
            y_train_modified=np.empty([y_train.shape[0],5])
            y_train_modified[:,0]=y_train[:,0]**2
            y_train_modified[:,1]=y_train[:,0]
            y_train_modified[:,2]=y_train[:,1]**2
            y_train_modified[:,3]=y_train[:,1]
            y_train_modified[:,4]=y_train[:,0]*y_train[:,1]

        #Create tuning curves

        num_nrns=X_b_train.shape[1] #Number of neurons to fit tuning curves for
        tuning_all=np.zeros([num_nrns,input_xy.shape[0]]) #Matrix that stores tuning curves for all neurons

        #Loop through neurons and fit tuning curves
        for j in range(num_nrns): #Neuron number

            if self.encoding_model=='linear':
                tuning=glm_run(y_train,X_b_train[:,j:j+1],input_xy)
            if self.encoding_model=='quadratic':
                tuning=glm_run(y_train_modified,X_b_train[:,j:j+1],input_xy_modified)
            #Enter tuning curves into matrix
            tuning_all[j,:]=np.squeeze(tuning)

        #Save tuning curves to be used in "predict" function
        self.tuning_all=tuning_all
        self.input_xy=input_xy

        #Get information about the probability of being in one state (position/velocity) based on the previous state
        #Here we're calculating the standard deviation of the change in state (velocity/acceleration) in the training set
        n=y_train.shape[0]
        dx=np.zeros([n-1,1])
        for i in range(n-1):
            dx[i]=np.sqrt((y_train[i+1,0]-y_train[i,0])**2+(y_train[i+1,1]-y_train[i,1])**2) #Change in state across time steps
        std=np.sqrt(np.mean(dx**2)) #dx is only positive. this gets approximate stdev of distribution (if it was positive and negative)
        self.std=std #Save for use in "predict" function

        #Get probability of being in each state - we are not using this since it did not help decoding performance
        # n_x=np.empty([input_xy.shape[0]])
        # for i in range(n):
        #     loc_idx=np.argmin(cdist(y_train[0:1,:],input_xy))
        #     n_x[loc_idx]=n_x[loc_idx]+1
        # p_x=n_x/n
        # self.p_x=p_x

    def predict(self,X_b_test,y_test):

        """
        Predict outcomes using trained tuning curves

        Parameters
        ----------
        X_b_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        y_test: numpy 2d array of shape [n_samples,n_outputs]
            The actual outputs
            This parameter is necesary for the NaiveBayesDecoder  (unlike most other decoders)
            because the first value is nececessary for initialization

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        #Get values saved in "fit" function
        tuning_all=self.tuning_all
        input_xy=self.input_xy
        std=self.std

        #Get probability of going from one state to the next
        dists = squareform(pdist(input_xy, 'euclidean')) #Distance between all states in "input_xy"
        #Probability of going from one state to the next, based on the above calculated distances
        #The probability is calculated based on the distances coming from a Gaussian with standard deviation of std
        prob_dists=norm.pdf(dists,0,std)

        #Initializations
        loc_idx= np.argmin(cdist(y_test[0:1,:],input_xy)) #The index of the first location
        num_nrns=tuning_all.shape[0] #Number of neurons
        y_test_predicted=np.empty([X_b_test.shape[0],2]) #Initialize matrix of predicted outputs
        num_ts=X_b_test.shape[0] #Number of time steps we are predicting

        #Loop across time and decode
        for t in range(num_ts):
            rs=X_b_test[t,:] #Number of spikes at this time point (in the interval we've specified including bins_before and bins_after)

            probs_total=np.ones([tuning_all[0,:].shape[0]]) #Vector that stores the probabilities of being in any state based on the neural activity (does not include probabilities of going from one state to the next)
            for j in range(num_nrns): #Loop across neurons
                lam=np.copy(tuning_all[j,:]) #Expected spike counts given the tuning curve
                r=rs[j] #Actual spike count
                probs=np.exp(-lam)*lam**r/math.factorial(r) #Probability of the given neuron's spike count given tuning curve (assuming poisson distribution)
                probs_total=np.copy(probs_total*probs) #Update the probability across neurons (probabilities are multiplied across neurons due to the independence assumption)
            prob_dists_vec=np.copy(prob_dists[loc_idx,:]) #Probability of going to all states from the previous state
            probs_final=probs_total*prob_dists_vec #Get final probability (multiply probabilities based on spike count and previous state)
            # probs_final=probs_total*prob_dists_vec*self.p_x #Get final probability when including p(x), i.e. prior about being in states, which we're not using
            loc_idx=np.argmax(probs_final) #Get the index of the current state (that w/ the highest probability)
            y_test_predicted[t,:]=input_xy[loc_idx,:] #The current predicted output

        return y_test_predicted #Return predictions




######### ALIASES for Regression ########

#LinearDecoder = LinearRegression
WienerFilterDecoder = WienerFilterRegression
WienerCascadeDecoder = WienerCascadeRegression
DenseNNDecoder = DenseNNRegression
SimpleRNNDecoder = SimpleRNNRegression
GRUDecoder = GRURegression
LSTMDecoder = LSTMRegression
XGBoostDecoder = XGBoostRegression
SVRDecoder = SVRegression
KalmanFilterDecoder = KalmanFilterRegression
NaiveBayesDecoder = NaiveBayesRegression
