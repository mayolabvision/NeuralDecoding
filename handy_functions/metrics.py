import numpy as np

########## R-squared (R2) ##########
def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """
    y_mean = np.mean(y_test, axis=0)
    numer = np.sum((y_test_pred - y_test) ** 2, axis=0)
    denom = np.sum((y_test - y_mean) ** 2, axis=0)
    
    R2 = 1 - numer / denom #Append R2 of this output to the list

    return R2  #Return an array of R2s

def adjust_R2(R2,num_observations,num_predictors):

    adjusted_R2 = 1 - (1 - R2) * (num_observations - 1) / (num_observations - num_predictors - 1)

########## Pearson's correlation (rho) ##########
def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """
    
    rho_array = np.empty([1,y_test.shape[1]])
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        rho_array[0,i]=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]

    return rho_array #Return an array of R2s

########## Root Mean Squared Error (RMSE) ##########
def get_RMSE(y_test, y_test_pred):
    """ 
    Function to get RMSE

    Parameters
    ----------
    y_test : numpy array
        The true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred : numpy array
        The predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    RMSE_array : numpy array
        An array of RMSEs for each output
    """
    num_outputs = y_test.shape[1]

    RMSE_array = np.sqrt(np.mean((y_test_pred - y_test) ** 2, axis=0))

    return RMSE_array

