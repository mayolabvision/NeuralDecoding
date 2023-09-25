import numpy as np

########## R-squared (R2) ##########

def get_R2_wShuf(y_test,y_test_pred):

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
    num_repeats = 100
    num_parts = 10

    a = np.arange(y_test.shape[0])

    R2_array = np.empty([num_repeats*num_parts,y_test.shape[1]]) #Initialize a list that will contain the R2s for all the outputs
    counter = -1
    for _ in range(num_repeats):
        np.random.shuffle(a)
        sub_inds = np.array_split(a, num_parts)
        for i,arr in enumerate(sub_inds):
            sub_ytest = y_test[arr,:]
            sub_ytestP = y_test_pred[arr,:]

            y_mean = np.mean(y_test, axis=0)
            numer = np.sum((sub_ytestP - sub_ytest) ** 2, axis=0)
            denom = np.sum((sub_ytest - y_mean) ** 2, axis=0)
            
            counter += 1
            R2_array[counter,:] = 1 - numer / denom #Append R2 of this output to the list

    return R2_array.mean(axis=0),R2_array #Return an array of R2s

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

########## Pearson's correlation (rho) ##########

def get_rho_wShuf(y_test,y_test_pred):

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
    
    num_repeats = 100
    num_parts = 10

    a = np.arange(y_test.shape[0])

    rho_array = np.empty([num_repeats*num_parts,y_test.shape[1]]) #Initialize a list that will contain the R2s for all the outputs
    counter = -1
    for _ in range(num_repeats):
        np.random.shuffle(a)
        sub_inds = np.array_split(a, num_parts)
        for i,arr in enumerate(sub_inds):
            sub_ytest = y_test[arr,:]
            sub_ytestP = y_test_pred[arr,:]

            counter += 1
            for i in range(sub_ytest.shape[1]): #Loop through outputs
                #Compute rho for each output
                rho_array[counter,i]=np.corrcoef(sub_ytest[:,i].T,sub_ytestP[:,i].T)[0,1]

    return rho_array.mean(axis=0),rho_array #Return an array of R2s

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
    
    rho_array = np.empty([1,y_test.shape[1])
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        rho_array[0,i]=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]

    return rho_array #Return an array of R2s


