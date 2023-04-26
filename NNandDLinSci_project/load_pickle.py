import pickle

def get_outputs(result_dir,load_folder):
    y_train, y_test, y_train_pred, y_test_pred, R2s, rhos, time_elapsed = [],[],[],[]
    for fold in range(3):
        with open(load_folder+result_dir+'fold_'+str(fold)+'.pickle','rb') as f:
            y_train,y_test,y_train_pred,y_test_pred,r2s,rhs,te=pickle.load(f,encoding='latin1')
            y_train.append(y_train)
            y_test.append(y_test)
            y_train_pred.append(y_train_pred)
            y_test_pred.append(y_test_pred)
            R2s.append(r2s)
            rhos.append(rhs)
            time_elapsed.append(te)
    return y_train, y_test, y_train_pred, y_test_pred, R2s, rhos, time_elapsed

