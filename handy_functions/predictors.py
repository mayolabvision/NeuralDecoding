import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

class WeightedMovingAverage(object):

    """
    Class for the Weighted Moving Average Decoder

    There are no parameters to set.

    This class implements a simple weighted moving average approach for decoding.
    """

    def __init__(self):
        return

    def fit(self, X_flat_train, y_train, window_size=10):

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
        self.n_samples, self.n_features = X_flat_train.shape

    def predict(self, X_flat_test):

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

        n_samples_test, n_features_test = X_flat_test.shape
        y_test_predicted = np.zeros((n_samples_test, self.n_outputs))

        for i in range(n_samples_test):
            # Calculate the weighted moving average for each output
            for j in range(self.n_outputs):
                # Use a simple weighted moving average with a window of size self.window_size
                y_test_predicted[i, j] = np.sum(X_flat_test[i, -self.window_size:]) / self.window_size

        return y_test_predicted

class OMPPredictor(object):

    """
    Class for the Orthogonal Matching Pursuit (OMP) Predictor

    There are no parameters to set.

    This class implements an OMP-based approach for predicting motor outputs from neural activity.
    """

    def __init__(self):
        return

    def fit(self, X_flat_train, y_train, n_selected_features=5):

        """
        Train the OMP Predictor

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples, n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted.

        n_selected_features: integer, optional, default=5
            The number of selected features (neurons) to use in the prediction.

        Returns
        -------
        None
        """

        self.n_selected_features = n_selected_features
        self.omp_models = []

        # Fit separate OMP models for each output
        for i in range(y_train.shape[1]):
            omp_model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_selected_features)
            omp_model.fit(X_flat_train, y_train[:, i])
            self.omp_models.append(omp_model)

    def predict(self, X_flat_test):

        """
        Predict outcomes using the trained OMP Predictor

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples, n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples, n_outputs]
            The predicted outputs.
        """

        n_samples_test, _ = X_flat_test.shape
        n_outputs = len(self.omp_models)
        y_test_predicted = np.zeros((n_samples_test, n_outputs))

        # Predict each output using the corresponding OMP model
        for i in range(n_outputs):
            y_test_predicted[:, i] = self.omp_models[i].predict(X_flat_test)

        return y_test_predicted

