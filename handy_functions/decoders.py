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

    def fit(self, X_flat_train, y_train):
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

        # Initialize linear regression model
        self.model = linear_model.LinearRegression()

        # Train the model
        self.model.fit(X_filtered_train, y_train)

        return self.model.coef_, self.model.intercept_

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


##################### LINEAR REGRESSION ##########################

class LinearRegression(object):

    """
    Class for the Linear Regression Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self):
        return


    def fit(self,X_flat_train,y_train):

        """
        Train Linear Regression Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        self.model=linear_model.LinearRegression() #Initialize linear regression model
        self.model.fit(X_flat_train, y_train) #Train the model
        
        return self.model.coef_,self.model.intercept_

    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Linear Regression Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted=self.model.predict(X_flat_test) #Make predictions

        return y_test_predicted

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

    def __init__(self, degree=3):
         self.degree = degree
         self.models = []

    def fit(self, X_flat_train, y_train):
        """
        Train Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs = y_train.shape[1]  # Number of outputs
        for i in range(num_outputs):  # Loop through outputs
            # Fit linear portion of model
            regr = linear_model.LinearRegression()  # Call the linear portion of the model "regr"
            regr.fit(X_flat_train, y_train[:, i])  # Fit linear
            y_train_predicted_linear = regr.predict(X_flat_train)  # Get outputs of linear portion of model

            # Apply the Wiener filter to the linear predictions
            y_train_filtered_linear = self.apply_wiener_filter(y_train_predicted_linear)

            # Fit nonlinear portion of model on the filtered linear predictions
            p = np.polyfit(y_train_filtered_linear, y_train[:, i], self.degree)

            # Add model for this output (both linear and nonlinear parts) to the list "models"
            self.models.append([regr, p])

    def get_coefficients_intercepts(self, output_idx):
        if output_idx < len(self.models):
            linear_model = self.models[output_idx][0]
            coefficients = linear_model.coef_
            intercept = linear_model.intercept_
            return coefficients, intercept
        else:
            return None

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

        num_outputs = len(self.models)  # Number of outputs being predicted

        y_test_predicted = np.empty([X_flat_test.shape[0], num_outputs])  # Initialize matrix that contains predicted outputs
        for i in range(num_outputs):  # Loop through outputs
            [regr, p] = self.models[i]  # Get the linear (regr) and nonlinear (p) portions of the trained model

            # Predictions on test set
            y_test_predicted_linear = regr.predict(X_flat_test)  # Get predictions on the linear portion of the model

            # Apply the Wiener filter to the linear predictions
            y_test_filtered_linear = self.apply_wiener_filter(y_test_predicted_linear)

            # Run the linear predictions through the nonlinearity to get the final predictions
            y_test_predicted[:, i] = np.polyval(p, y_test_filtered_linear)

        return y_test_predicted

    def apply_wiener_filter(self, x):
        """
        Apply the Wiener filter to the input data

        Parameters
        ----------
        x: numpy 1d array
            A single feature of the neural data.

        Returns
        -------
        x_filtered: numpy 1d array
            The filtered feature using the Wiener filter
        """
        # Apply the Wiener filter to the input feature
        x_filtered = wiener(x)

        return x_filtered


##################### LINEAR CASCADE ##########################

class LinearCascadeRegression(object):

    """
    Class for the Wiener Cascade Decoder

    Parameters
    ----------
    degree: integer, optional, default 3
        The degree of the polynomial used for the static nonlinearity
    """

    def __init__(self,degree=3):
         self.degree=degree
         self.models = [] 

    def fit(self,X_flat_train,y_train):

        """
        Train Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        num_outputs=y_train.shape[1] #Number of outputs
        for i in range(num_outputs): #Loop through outputs
            #Fit linear portion of model
            regr = linear_model.LinearRegression() #Call the linear portion of the model "regr"
            regr.fit(X_flat_train, y_train[:,i]) #Fit linear
            y_train_predicted_linear=regr.predict(X_flat_train) # Get outputs of linear portion of model
            #Fit nonlinear portion of model
            p=np.polyfit(y_train_predicted_linear,y_train[:,i],self.degree)
            #Add model for this output (both linear and nonlinear parts) to the list "models"
            self.models.append([regr,p])
        self.model=self.models

    def get_coefficients_intercepts(self, output_idx):
        if output_idx < len(self.models):
            linear_model = self.models[output_idx][0]
            coefficients = linear_model.coef_
            intercept = linear_model.intercept_
            return coefficients, intercept
        else:
            return None


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        num_outputs=len(self.model) #Number of outputs being predicted. Recall from the "fit" function that self.model is a list of models
        y_test_predicted=np.empty([X_flat_test.shape[0],num_outputs]) #Initialize matrix that contains predicted outputs
        for i in range(num_outputs): #Loop through outputs
            [regr,p]=self.model[i] #Get the linear (regr) and nonlinear (p) portions of the trained model
            #Predictions on test set
            y_test_predicted_linear=regr.predict(X_flat_test) #Get predictions on the linear portion of the model
            y_test_predicted[:,i]=np.polyval(p,y_test_predicted_linear) #Run the linear predictions through the nonlinearity to get the final predictions
        
        return y_test_predicted

##################### Moving Average Predictor ##########################
class WeightedMovingAverage(object):

    """
    Class for the Weighted Moving Average Decoder

    There are no parameters to set.

    This class implements a simple weighted moving average approach for decoding.
    """
    def __init__(self, window_size=10, n_outputs=None):
        self.window_size = window_size
        self.n_outputs = n_outputs

    def fit(self, X_train, y_train):

        """
        Train Weighted Moving Average Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.

        window_size: integer, optional, default=10
            The size of the moving average window for decoding.

        Returns
        -------
        None
        """

        self.window_size = window_size
        self.n_samples, self.n_features = X_train.shape
        self.n_outputs = y_train.shape[1]  # Number of outputs

    def predict(self, X_test):

        """
        Predict outcomes using the trained Weighted Moving Average Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """

        n_samples_test, n_features_test = X_test.shape
        y_test_predicted = np.zeros((n_samples_test, self.n_outputs))

        for i in range(n_samples_test):
            # Calculate the weighted moving average for each output
            for j in range(self.n_outputs):
                # Use a simple weighted moving average with a window of size self.window_size
                y_test_predicted[i, j] = np.sum(X_test[i, -self.window_size:]) / self.window_size

        return y_test_predicted

##################### Sparse Decoder ##########################
class OrthogonalMatchingPursuitDecoder(object):
    """
    Class for the Orthogonal Matching Pursuit (OMP) Decoder

    Parameters
    ----------
    n_nonzero_coefs: int, optional, default=None
        The maximum number of nonzero coefficients to include in the solution.
        If None, all non-zero coefficients are used.

    n_outputs: int, optional, default=None
        The number of output dimensions to predict.

    Attributes
    ----------
    omp_models: list of OMP models
        List of OMP models, one for each output dimension.

    n_outputs: int
        The number of output dimensions.
    """

    def __init__(self, n_nonzero_coefs=None, n_outputs=None):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_outputs = n_outputs
        self.models = []

    def fit(self, X_train, y_train):
        """
        Train Orthogonal Matching Pursuit (OMP) Decoder

        Parameters
        ----------
        X_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.

        Returns
        -------
        None
        """
        if self.n_outputs is None:
            self.n_outputs = y_train.shape[1]

        for i in range(self.n_outputs):
            model = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
            model.fit(X_train, y_train[:, i])
            self.models.append(model)

        return model.coef_,model.intercept_


    def predict(self, X_test):
        """
        Predict outcomes using the trained Orthogonal Matching Pursuit (OMP) Decoder

        Parameters
        ----------
        X_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """
        n_samples_test, _ = X_test.shape
        y_test_predicted = np.zeros((n_samples_test, self.n_outputs))

        for i in range(self.n_outputs):
            model = self.models[i]
            y_test_predicted[:, i] = model.predict(X_test)

        return y_test_predicted

#################### Cascade OMP ###########################
class CascadeOrthogonalMatchingPursuitDecoder(object):
    """
    Class for the Cascade Orthogonal Matching Pursuit (C-OMP) Decoder

    Parameters
    ----------
    n_stages: int, optional, default=None
        The number of cascade stages.
    
    n_nonzero_coefs: int, optional, default=None
        The maximum number of nonzero coefficients to include in the solution.

    Attributes
    ----------
    omp_models: list of OMP models
        List of OMP models, one for each cascade stage.
    """

    def __init__(self, n_stages=1, n_nonzero_coefs=None, n_outputs=None):
        self.n_stages = n_stages
        self.n_nonzero_coefs = n_nonzero_coefs
        self.n_outputs = n_outputs
        self.models = []

    def fit(self, X_train, y_train):
        """
        Train Cascade Orthogonal Matching Pursuit (C-OMP) Decoder

        Parameters
        ----------
        X_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.

        Returns
        -------
        None
        """
        self.n_samples, self.n_features = X_train.shape
        if self.n_outputs is None:
            self.n_outputs = y_train.shape[1]

        for i in range(self.n_stages):
            model = OrthogonalMatchingPursuit(n_nonzero_coefs=self.n_nonzero_coefs)
            model.fit(X_train, y_train)
            self.models.append(model)

        return model.coef_,model.intercept_

    def predict(self, X_test):
        """
        Predict outcomes using the trained Cascade Orthogonal Matching Pursuit (C-OMP) Decoder

        Parameters
        ----------
        X_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """
        n_samples_test, _ = X_test.shape
        y_test_predicted = np.zeros((n_samples_test, self.n_outputs))

        for i in range(self.n_stages):
            model = self.models[i]
            y_test_predicted += model.predict(X_test)

        return y_test_predicted


#################### WRLS ###########################
from sklearn.linear_model import SGDRegressor

class WeightedRecursiveLeastSquares(object):
    """  
    Class for Weighted Recursive Least Squares (WRLS) Decoder

    Parameters
    ----------
    alpha: float, optional, default 0.9
        Forgetting factor, controlling the weight of past information.

    Attributes
    ----------
    coefficients: numpy 1d array
        Model coefficients for each feature.

    Examples
    --------
    # Create and fit a WRLS decoder
    decoder = WeightedRecursiveLeastSquares()
    decoder.fit(X_train, y_train)
    
    # Make predictions
    y_test_predicted = decoder.predict(X_test)
    """

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.models = None
        self.is_fitted = False  # Custom attribute to track if models are fitted

    def fit(self, X_train, y_train):
        """
        Train the WRLS Decoder.

        Parameters
        ----------
        X_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.
        """
        n_outputs = y_train.shape[1]
        self.models = [SGDRegressor(learning_rate='constant', eta0=0.1, alpha=self.alpha)
                       for _ in range(n_outputs)]

        for t in range(X_train.shape[0]):
            x_t = X_train[t, :].reshape(1, -1)
            y_t = y_train[t, :]

            for i in range(n_outputs):
                # Check if the model is fitted, and if not, fit it
                if not self.is_fitted:
                    self.models[i].partial_fit(x_t, [y_t[i]])
                else:
                    y_hat_t = self.models[i].predict(x_t)
                    error_t = y_t[i] - y_hat_t

                    # Update the model for output i
                    self.models[i].partial_fit(x_t, [y_t[i]], [error_t])

        # After fitting, set the flag to indicate models are fitted
        self.is_fitted = True

    def predict(self, X_test):
        """
        Predict outcomes using the trained WRLS Decoder.

        Parameters
        ----------
        X_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """
        n_samples = X_test.shape[0]
        n_outputs = len(self.models)
        y_test_predicted = np.empty((n_samples, n_outputs))

        for i in range(n_outputs):
            y_test_predicted[:, i] = self.models[i].predict(X_test)

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


########################## Extended Kalman Filter ####################################
import numpy as np
from numpy.linalg import inv

class ExtendedKalmanFilterRegression(object):
    def __init__(self, C=1):
        self.C = C
        self.A = None
        self.Q = None
        self.H = None
        self.R = None
        self.x = None
        self.P = None

    def fit(self, X_kf_train, y_train):
        X = np.matrix(y_train.T)
        Z = np.matrix(X_kf_train.T)

        nt = X.shape[1]

        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]
        self.A = X2 * X1.T * inv(X1 * X1.T)
        self.Q = ((X2 - self.A * X1) * ((X2 - self.A * X1).T)) / (nt - 1) / self.C

        self.H = Z * X.T * (inv(X * X.T))
        self.R = ((Z - self.H * X) * ((Z - self.H * X).T)) / nt

        self.x = np.matrix(np.zeros((X.shape[0], 1)))  # Initialize state estimate
        self.P = np.matrix(np.eye(X.shape[0]))  # Initialize state estimate covariance

    def predict(self, X_kf_test, y_test):
        Z = np.matrix(X_kf_test.T)

        num_states = self.x.shape[0]
        states = np.empty((num_states, Z.shape[1]))
        states[:, 0] = np.squeeze(self.x)

        for t in range(Z.shape[1] - 1):
            # Prediction step
            self.x = self.A * self.x
            self.P = self.A * self.P * self.A.T + self.Q

            # Update step
            K = self.P * self.H.T * inv(self.H * self.P * self.H.T + self.R)
            self.x = self.x + K * (Z[:, t + 1] - self.H * self.x)
            self.P = (np.matrix(np.eye(num_states)) - K * self.H) * self.P

            states[:, t + 1] = np.squeeze(self.x)

        y_test_predicted = states.T

        return y_test_predicted

########################## Unscented Kalman Filter ####################################

from scipy.linalg import sqrtm

class UnscentedKalmanFilterRegression(object):

    """
    Class for the Unscented Kalman Filter Decoder

    Parameters
    -----------
    C - float, optional, default 1
    This parameter scales the noise matrix associated with the transition in kinematic states.
    It effectively allows changing the weight of the new neural evidence in the current update.

    Our implementation of the Unscented Kalman filter for neural decoding is based on the concept of sigma-point Kalman filtering, where sigma points are propagated through nonlinear functions to estimate the state and its covariance.

    """

    def __init__(self, C=1):
        self.C = C

    def fit(self, X_ukf_train, y_train):

        """
        Train Unscented Kalman Filter Decoder

        Parameters
        ----------
        X_ukf_train: numpy 2d array of shape [n_samples (i.e. timebins), n_neurons]
            This is the neural data in Unscented Kalman filter format.
            See example file for an example of how to format the neural data correctly.

        y_train: numpy 2d array of shape [n_samples (i.e. timebins), n_outputs]
            This is the outputs that are being predicted.
        """

        # Rename and reformat the variables for Unscented Kalman filter
        X = np.matrix(y_train.T)
        Z = np.matrix(X_ukf_train.T)

        # Number of time bins
        nt = X.shape[1]

        # Calculate the transition matrix (from x_t to x_t+1) using least-squares, and compute its covariance
        X2 = X[:, 1:]
        X1 = X[:, 0:nt - 1]
        A = X2 * X1.T * np.linalg.inv(X1 * X1.T)  # Transition matrix
        W = ((X2 - A * X1) * ((X2 - A * X1).T)) / (nt - 1) / self.C  # Covariance of transition matrix

        # Calculate the measurement matrix (from x_t to z_t) using least-squares, and compute its covariance
        H = Z * X.T * (np.linalg.inv(X * X.T))  # Measurement matrix
        Q = ((Z - H * X) * ((Z - H * X).T)) / nt  # Covariance of measurement matrix

        params = [A, W, H, Q]
        self.model = params

        return params

    def predict(self, X_ukf_test, y_test):

        """
        Predict outcomes using trained Unscented Kalman Filter Decoder

        Parameters
        ----------
        X_ukf_test: numpy 2d array of shape [n_samples (i.e. timebins), n_neurons]
            This is the neural data in Unscented Kalman filter format.

        y_test: numpy 2d array of shape [n_samples (i.e. timebins), n_outputs]
            The actual outputs
            This parameter is necessary for the Unscented Kalman filter
            because the first value is necessary for initialization.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples (i.e. timebins), n_outputs]
            The predicted outputs
        """

        # Extract parameters
        A, W, H, Q = self.model

        # Rename and reformat the variables for Unscented Kalman filter
        X = np.matrix(y_test.T)
        Z = np.matrix(X_ukf_test.T)

        # Initializations
        num_states = X.shape[0]  # Dimensionality of the state
        states = np.empty(X.shape)  # Keep track of states over time (states is what will be returned as y_test_predicted)
        P_m = np.matrix(np.zeros([num_states, num_states]))
        P = np.matrix(np.zeros([num_states, num_states]))
        state = X[:, 0]  # Initial state
        states[:, 0] = np.copy(np.squeeze(state))

        # Get predicted state for every time bin
        for t in range(X.shape[1] - 1):
            # Do first part of state update - based on transition matrix
            P_m = A * P * A.T + W
            state_m = A * state

            # Do second part of state update - based on measurement matrix
            K = P_m * H.T * np.linalg.inv(H * P_m * H.T + Q)  # Calculate Kalman gain
            P = (np.matrix(np.eye(num_states)) - K * H) * P_m
            state = state_m + K * (Z[:, t + 1] - H * state_m)
            states[:, t + 1] = np.squeeze(state)  # Record state at the timestep
        y_test_predicted = states.T

        return y_test_predicted


##################### GAUSSIAN PROCESS REGRESSION (GPR) ##########################
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Kernel
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GaussianProcessRegressionDecoder(object):
    """
    Class for the Gaussian Process Regression Decoder

    Parameters
    ----------
    kernel_length_scale: float, optional, default 1.0
        The length scale for the RBF kernel function

    Attributes
    ----------
    kernel_length_scale: float
        The length scale for the RBF kernel function
    models: list
        A list to store trained Gaussian Process Regressor models for each output
    """

    def __init__(self, kernel_length_scale=1.0):
        self.kernel_length_scale = kernel_length_scale
        self.models = []

    def fit(self, X_flat_train, y_train):
        """
        Train Gaussian Process Regression Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        num_outputs = y_train.shape[1]

        # Initialize a list to store coefficients for each output dimension
        coefficients = []

        for i in range(num_outputs):
            kernel = 1.0 * RBF(length_scale=self.kernel_length_scale)
            gpr = GaussianProcessRegressor(kernel=kernel)
            gpr.fit(X_flat_train, y_train[:, i])
            self.models.append(gpr)

            # Extract the coefficients (weights) from the GPR kernel
            coefficients_i = gpr.kernel_.get_params()['k1__length_scale']
            coefficients.append(coefficients_i)

        return coefficients

    def predict(self, X_flat_test):
        """
        Predict outcomes using trained Gaussian Process Regression Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs
        """
        num_outputs = len(self.models)
        y_test_predicted = np.empty([X_flat_test.shape[0], num_outputs])

        for i in range(num_outputs):
            gpr_model = self.models[i]
            y_test_predicted[:, i] = gpr_model.predict(X_flat_test)

        return y_test_predicted

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

    def __init__(self,units=400,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1):
         self.dropout=dropout
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers

         #If "units" is an integer, put it in the form of a vector
         try: #Check if it's a vector
             units[0]
         except: #If it's not a vector, create a vector of the number of units for each layer
             units=[units]
         self.units=units

         #Determine the number of hidden layers (based on "units" that the user entered)
         self.num_layers=len(units)

    def fit(self,X_flat_train,y_train):

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
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_flat_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_flat_train,y_train,batch_size=self.batch_size,epochs=self.num_epochs,verbose=self.verbose,workers=self.workers,use_multiprocessing=True) #Fit the model
        self.model=model

        # Retrieve the model's weights and biases
        model_weights = []
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                model_weights.append(layer_weights)

        return model_weights

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

    def __init__(self,units=400,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1):
         self.units=units
         self.dropout=dropout
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers


    def fit(self,X_train,y_train):

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

        model=Sequential() #Declare model
        #Add recurrent layer
        #if keras_v1:
        #    model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout,activation='relu')) #Within recurrent layer, include dropout
        #else:
        model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout,activation='relu')) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        #Fit model (and set fitting parameters)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_train,y_train,batch_size=self.batch_size,epochs=self.num_epochs,verbose=self.verbose,workers=self.workers,use_multiprocessing=True) #Fit the model
        self.model=model

        # Retrieve the model's weights and biases
        model_weights = []
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                model_weights.append(layer_weights)

        return model_weights

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

    def __init__(self,units=400,dropout=0,batch_size=128,num_epochs=10,verbose=0,workers=1):
         self.units=units
         self.dropout=dropout
         self.batch_size=batch_size
         self.num_epochs=num_epochs
         self.verbose=verbose
         self.workers=workers


    def fit(self,X_train,y_train):

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
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_train,y_train,batch_size=self.batch_size,epochs=self.num_epochs,verbose=self.verbose,workers=self.workers,use_multiprocessing=True) #Fit the model
        self.model=model

        # Retrieve the model's weights and biases
        model_weights = []
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                model_weights.append(layer_weights)

        return model_weights

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

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0, batch_size=128, workers=1):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.workers = workers
        self.model = None

    def fit(self, X_train, y_train):
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

        # Define the LFADS model
        input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
        lstm_layer = LSTM(self.units, dropout=self.dropout)(input_layer)  # Remove return_sequences=True
        output_layer = Dense(y_train.shape[1])(lstm_layer)  # Adjust for y_train shape

        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size, verbose=self.verbose, workers=self.workers)

        self.model = model
        
        # Retrieve the model's weights and biases
        model_weights = []
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                model_weights.append(layer_weights)

        return model_weights

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


##################### EXTREME GRADIENT BOOSTING (XGBOOST) ##########################

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

    def __init__(self,max_depth=3,num_round=300,eta=0.3,gpu=-1):
        self.max_depth=max_depth
        self.num_round=num_round
        self.eta=eta
        self.gpu=gpu

    def fit(self,X_flat_train,y_train):

        """
        Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """


        num_outputs=y_train.shape[1] #Number of outputs

        #Set parameters for XGBoost
        param = {'objective': "reg:linear", #for linear output
            'eval_metric': "logloss", #loglikelihood loss
            'max_depth': self.max_depth, #this is the only parameter we have set, it's one of the way or regularizing
            'eta': self.eta,
            'seed': 2925, #for reproducibility
            'silent': True,
            'verbosity' : 0}
        if self.gpu<0:
            param['nthread'] = -1 #with -1 it will use all available threads
        else:
            param['gpu_id']=self.gpu
            param['updater']='grow_gpu'

        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            dtrain = xgb.DMatrix(X_flat_train, label=y_train[:,y_idx]) #Put in correct format for XGB
            bst = xgb.train(param, dtrain, self.num_round) #Train model
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

class SVRegression(object):

    """
    Class for the Support Vector Regression (SVR) Decoder
    This simply leverages the scikit-learn SVR

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self,max_iter=-1,C=3.0):
        self.max_iter=max_iter
        self.C=C
        return


    def fit(self,X_flat_train,y_train):

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

        num_outputs=y_train.shape[1] #Number of outputs
        models=[] #Initialize list of models (there will be a separate model for each output)
        for y_idx in range(num_outputs): #Loop through outputs
            model=SVR(C=self.C, max_iter=self.max_iter) #Initialize SVR model
            model.fit(X_flat_train, y_train[:,y_idx]) #Train the model
            models.append(model) #Add fit model to list of models
        self.model=models

        return self.model.coef_,self.model.intercept_

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

LinearDecoder = LinearRegression
LinearCascadeDecoder = LinearCascadeRegression
WienerFilterDecoder = WienerFilterRegression
WienerCascadeDecoder = WienerCascadeRegression
KalmanFilterDecoder = KalmanFilterRegression
ExtendedKalmanFilterDecoder = ExtendedKalmanFilterRegression
UnscentedKalmanFilterDecoder = UnscentedKalmanFilterRegression
DenseNNDecoder = DenseNNRegression
SimpleRNNDecoder = SimpleRNNRegression
GRUDecoder = GRURegression
LSTMDecoder = LSTMRegression
XGBoostDecoder = XGBoostRegression
SVRDecoder = SVRegression
NaiveBayesDecoder = NaiveBayesRegression




####################################### CLASSIFICATION ####################################################




class WienerFilterClassification(object):

    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn logistic regression.
    """

    def __init__(self,C=1):
        self.C=C
        return


    def fit(self,X_flat_train,y_train):

        """
        Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        # if self.C>0:
        self.model=linear_model.LogisticRegression(C=self.C,multi_class='auto') #Initialize linear regression model
        # else:
            # self.model=linear_model.LogisticRegression(penalty='none',solver='newton-cg') #Initialize linear regression model
        self.model.fit(X_flat_train, y_train) #Train the model


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted=self.model.predict(X_flat_test) #Make predictions
        return y_test_predicted




##################### SUPPORT VECTOR REGRESSION ##########################

class SVClassification(object):

    """
    Class for the Support Vector Classification Decoder
    This simply leverages the scikit-learn SVM

    Parameters
    ----------
    C: float, default=3.0
        Penalty parameter of the error term

    max_iter: integer, default=-1
        the maximum number of iteraations to run (to save time)
        max_iter=-1 means no limit
        Typically in the 1000s takes a short amount of time on a laptop
    """

    def __init__(self,max_iter=-1,C=3.0):
        self.max_iter=max_iter
        self.C=C
        return


    def fit(self,X_flat_train,y_train):

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

        model=SVC(C=self.C, max_iter=self.max_iter) #Initialize model
        model.fit(X_flat_train, y_train) #Train the model
        self.model=model


    def predict(self,X_flat_test):

        """
        Predict outcomes using trained SV Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        model=self.model #Get fit model for that output
        y_test_predicted=model.predict(X_flat_test) #Make predictions
        return y_test_predicted


##################### DENSE (FULLY-CONNECTED) NEURAL NETWORK ##########################

class DenseNNClassification(object):

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

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose

         #If "units" is an integer, put it in the form of a vector
         try: #Check if it's a vector
             units[0]
         except: #If it's not a vector, create a vector of the number of units for each layer
             units=[units]
         self.units=units

         #Determine the number of hidden layers (based on "units" that the user entered)
         self.num_layers=len(units)

    def fit(self,X_flat_train,y_train):

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

        #Use one-hot coding for y
        if y_train.ndim==1:
            y_train=np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1]==1:
            y_train=np_utils.to_categorical(y_train.astype(int))

        model=Sequential() #Declare model
        #Add first hidden layer
        model.add(Dense(self.units[0],input_dim=X_flat_train.shape[1])) #Add dense layer
        model.add(Activation('relu')) #Add nonlinear (tanh) activation
        # if self.dropout!=0:
        if self.dropout!=0: model.add(Dropout(self.dropout))  #Dropout some units if proportion of dropout != 0

        #Add any additional hidden layers (beyond the 1st)
        for layer in range(self.num_layers-1): #Loop through additional layers
            model.add(Dense(self.units[layer+1])) #Add dense layer
            model.add(Activation('tanh')) #Add nonlinear (tanh) activation - can also make relu
            if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units if proportion of dropout != 0

        #Add dense connections to all outputs
        model.add(Dense(y_train.shape[1])) #Add final dense layer (connected to outputs)
        model.add(Activation('softplus'))

        #Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_flat_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_flat_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
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

        y_test_predicted_raw = self.model.predict(X_flat_test) #Make predictions

        y_test_predicted=np.argmax(y_test_predicted_raw,axis=1)

        return y_test_predicted



##################### SIMPLE RNN DECODER ##########################

class SimpleRNNClassification(object):

    """
    Class for the RNN decoder

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

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

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


        #Use one-hot coding for y
        if y_train.ndim==1:
            y_train=np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1]==1:
            y_train=np_utils.to_categorical(y_train.astype(int))

        model=Sequential() #Declare model
        #Add recurrent layer

        #### MAKE RELU ACTIVATION BELOW LIKE IN REGRESSION????? ####
        #if keras_v1:
        #    model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        #else:
        model.add(SimpleRNN(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softplus'))

        #Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_test) #Make predictions
        y_test_predicted=np.argmax(y_test_predicted_raw,axis=1)

        return y_test_predicted






##################### GATED RECURRENT UNIT (GRU) DECODER ##########################

class GRUClassification(object):

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

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

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


        #Use one-hot coding for y
        if y_train.ndim==1:
            y_train=np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1]==1:
            y_train=np_utils.to_categorical(y_train.astype(int))

        model=Sequential() #Declare model
        #Add recurrent layer
        #if keras_v1:
        #    model.add(GRU(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        #else:
        model.add(GRU(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softplus'))

        #Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_test) #Make predictions
        y_test_predicted=np.argmax(y_test_predicted_raw,axis=1)

        return y_test_predicted





#################### LONG SHORT TERM MEMORY (LSTM) DECODER ##########################

class LSTMClassification(object):

    """
    Class for the LSTM decoder

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

    def __init__(self,units=400,dropout=0,num_epochs=10,verbose=0):
         self.units=units
         self.dropout=dropout
         self.num_epochs=num_epochs
         self.verbose=verbose


    def fit(self,X_train,y_train):

        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """


        #Use one-hot coding for y
        if y_train.ndim==1:
            y_train=np_utils.to_categorical(y_train.astype(int))
        elif y_train.shape[1]==1:
            y_train=np_utils.to_categorical(y_train.astype(int))

        model=Sequential() #Declare model
        #Add recurrent layer
        #if keras_v1:
        #    model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout)) #Within recurrent layer, include dropout
        #else:
        model.add(LSTM(self.units,input_shape=(X_train.shape[1],X_train.shape[2]),dropout=self.dropout,recurrent_dropout=self.dropout)) #Within recurrent layer, include dropout
        if self.dropout!=0: model.add(Dropout(self.dropout)) #Dropout some units (recurrent layer output units)

        #Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))
        model.add(Activation('softplus'))

        #Fit model (and set fitting parameters)
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy']) #Set loss function and optimizer
        #if keras_v1:
        #    model.fit(X_train,y_train,nb_epoch=self.num_epochs,verbose=self.verbose) #Fit the model
        #else:
        model.fit(X_train,y_train,epochs=self.num_epochs,verbose=self.verbose) #Fit the model
        self.model=model


    def predict(self,X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted_raw = self.model.predict(X_test) #Make predictions
        y_test_predicted=np.argmax(y_test_predicted_raw,axis=1)

        return y_test_predicted




##################### EXTREME GRADIENT BOOSTING (XGBOOST) ##########################

class XGBoostClassification(object):
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

    def __init__(self, max_depth=3, num_round=300, eta=0.3, gpu=-1):
        self.max_depth = max_depth
        self.num_round = num_round
        self.eta = eta
        self.gpu = gpu

    def fit(self, X_flat_train, y_train):

        """
        Train XGBoost Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 1d array of shape (n_samples), with integers representing classes
                    or 2d array of shape [n_samples, n_outputs] in 1-hot form
            This is the outputs that are being predicted
        """

        # turn to categorial (not 1-hat)
        if (y_train.ndim == 2):
            if (y_train.shape[1] == 1):
                y_train = np.reshape(y_train, -1)
            else:
                y_train = np.argmax(y_train, axis=1, out=None)

        # Get number of classes
        n_classes = len(np.unique(y_train))

        # Set parameters for XGBoost
        param = {'objective': "multi:softmax",  # or softprob
                 'eval_metric': "mlogloss",  # loglikelihood loss
                 # 'eval_metric': "merror",
                 'max_depth': self.max_depth, # this is the only parameter we have set, it's one of the way or regularizing
                 'eta': self.eta,
                 'num_class': n_classes,  # y_train.shape[1],
                 'seed': 2925,  # for reproducibility
                 'silent': 1}
        if self.gpu < 0:
            param['nthread'] = -1  # with -1 it will use all available threads
        else:
            param['gpu_id'] = self.gpu
            param['updater'] = 'grow_gpu'

        dtrain = xgb.DMatrix(X_flat_train, label=y_train)  # Put in correct format for XGB
        bst = xgb.train(param, dtrain, self.num_round)  # Train model

        self.model = bst

    def predict(self, X_flat_test):

        """
        Predict outcomes using trained XGBoost Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 1d array with integers as classes
            The predicted outputs
        """

        dtest = xgb.DMatrix(X_flat_test)  # Put in XGB format
        bst = self.model  # Get fit model
        y_test_predicted = bst.predict(dtest)  # Make prediction
        return y_test_predicted

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
        bst=self.model #Get fit model
        y_test_predicted = bst.predict(dtest) #Make prediction
        return y_test_predicted
