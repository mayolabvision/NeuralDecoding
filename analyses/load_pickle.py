import pickle
import numpy as np

def get_outputs(result_dir,load_folder):
    num_folds = int(result_dir[35:37])
    num_repeats = int(result_dir[-4:])
    print(num_folds)
    print(num_repeats)
    print(blah)
    y_train, y_test, y_train_pred, y_test_pred, max_params, neuron_inds, R2s, rhos, time_elapsed = [],[],[],[],[],[],[],[],[]
    for fold in range(num_folds):
        with open(load_folder+result_dir+'fold_'+str(fold)+'.pickle','rb') as f:
            y_train0,y_test0,y_train_pred0,y_test_pred0,r2s,rhs,te,params,ninds=pickle.load(f,encoding='latin1')
            y_train.append(y_train0)
            y_test.append(y_test0)
            y_train_pred.append(y_train_pred0)
            y_test_pred.append(y_test_pred0)
            R2s.append(r2s)
            rhos.append(rhs)
            time_elapsed.append(te)
            max_params.append(params)
            neuron_inds.append(ninds)
    return y_train, y_test, y_train_pred, y_test_pred, R2s, rhos, time_elapsed, max_params, neuron_inds

