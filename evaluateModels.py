##################### BASIC EVALUATION OF ALL THE MODEL TYPES #########################

# MODEL 0 - Wiener Filter Decoder
def wienerFilter(X_flat_train,X_flat_valid,y_train):
    model=WienerFilterDecoder()
    model.fit(X_flat_train,y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s


# MODEL 1 - Wiener Cascade Decoder
def wienerCascade(X_flat_train,X_flat_valid,y_train,degree):
    model=WienerCascadeDecoder(degree=degree)
    model.fit(X_flat_train,y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 2 - XGBoost Decoder
def XGBoost(X_flat_train,X_flat_valid,y_train,max_depth,num_round,eta,gpu):
    model=XGBoostDecoder(max_depth=int(max_depth),num_round=int(num_round),eta=float(eta),gpu=gpu)
    model.fit(X_flat_train, y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 3 - SVR Decoder
def SVR(X_flat_train,X_flat_valid,y_train,C,max_iter):
    y_train_std=np.nanstd(y_train,axis=0)
    y_zscore_train=y_train/y_train_std
    y_zscore_test=y_test/y_train_std
    y_zscore_valid=y_valid/y_train_std
   
    model=SVRDecoder(C=int(C), max_iter=int(max_iter))
    model.fit(X_flat_train,y_zscore_train)
    y_zscore_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_zscore_valid,y_zscore_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 4 - Dense NN Decoder
def denseNN(X_flat_train,X_flat_valid,y_train,units,dropout,num_epochs):
    model=DenseNNDecoder(units=[int(num_units),int(num_units)],dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_flat_train,y_train)
    y_valid_predicted=model.predict(X_flat_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 5 - Simple RNN
def simpleRNN(X_train,X_valid,y_train,units,dropout,num_epochs):
    model=SimpleRNNDecoder(units=int(units),dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 6 - GRU Decoder
def GRU(X_train,X_valid,y_train,units,dropout,num_epochs):
    model=GRUDecoder(units=int(units),dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

# MODEL 7 - LSTM Decoder
def LSTM(X_train,X_valid,y_train,units,dropout,num_epochs):
    model=LSTMDecoder(units=int(units),dropout=float(dropout),num_epochs=int(num_epochs))
    model.fit(X_train,y_train)
    y_valid_predicted=model.predict(X_valid)
    R2s=get_R2(y_valid,y_valid_predicted)

    return y_valid_predicted,R2s

