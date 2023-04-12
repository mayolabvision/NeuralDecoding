from bayes_opt import BayesianOptimization

def wcFit(wc_evaluate,init_points,n_iter,kappa):
    BO = BayesianOptimization(wc_evaluate, {'degree': (1, 20.99)}, verbose=0)
    BO.maximize(init_points=int(init_points), n_iter=int_n_iter), kappa=int(kappa))
    #BO.res
    #BO.res[2]
    return BO

def xgbFit(xgb_evaluate,init_points,n_iter,kappa):
    BO = BayesianOptimization(xgb_evaluate, {'max_depth': (2, 6.99), 'num_round': (100,600.99), 'eta': (0.01, 0.8)},verbose=0) 
    BO.maximize(init_points=int(init_points), n_iter=int_n_iter), kappa=int(kappa))
    #BO.res
    #BO.res[10]
    return BO

def svrFit(svr_evaluate,init_points,n_iter,kappa):
    BO = BayesianOptimization(svr_evaluate, {'C': (2, 6.99), 'max_iter': (100,600.99)},verbose=0) 
    BO.maximize(init_points=int(init_points), n_iter=int_n_iter), kappa=int(kappa))
    #BO.res
    #BO.res[18]
    return BO

def nnFit(nn_evaluate,init_points,n_iter,kappa):
    BO = BayesianOptimization(nn_evaluate, {'num_units': (50, 800.99), 'frac_dropout': (0,.5), 'n_epochs': (2,15.99)},verbose=0)
    BO.maximize(init_points=int(init_points), n_iter=int_n_iter), kappa=int(kappa))
    #BO.res
    #BO.res[6]  #dnn
    #BO.res[7]  #rnn
    #BO.res[15] #gru
    #BO.res[19] #lstm
    return BO


