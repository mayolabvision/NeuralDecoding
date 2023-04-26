import pickle
import numpy as np

def get_outputs(result_dir,load_folder):
    y_train, y_test, y_train_pred, y_test_pred = [],[],[],[]
    R2s = np.empty([3,1])
    rhos = np.empty([3,1])
    time_elapsed = np.empty([3,1])
    for fold in range(3):
        with open(load_folder+result_dir+'fold_'+str(fold)+'.pickle','rb') as f:
            y_train0,y_test0,y_train_pred0,y_test_pred0,r2s,rhs,te=pickle.load(f,encoding='latin1')
            y_train.append(y_train0)
            y_test.append(y_test0)
            y_train_pred.append(y_train_pred0)
            y_test_pred.append(y_test_pred0)
            print(len(r2s))
            R2s[fold] = r2s[0][1]
            rhos[fold] = rhs[0][1]
            time_elapsed[fold] = te
    return y_train, y_test, y_train_pred, y_test_pred, R2s, rhos, time_elapsed

