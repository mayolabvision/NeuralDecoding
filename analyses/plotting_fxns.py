from sklearn import svm
from mlxtend.plotting import category_scatter
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.svm import LinearSVC
import matplotlib as mpl
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 
import numpy as np

def svm_cloud(features,label,axes,colors,title):
    test_size = int(np.round(label.shape[0] * 0.4, 0))
    x_train = features[:-test_size].values
    y_train = label[:-test_size].values
    x_test = features[-test_size:].values
    y_test = label[-test_size:].values

    #fig = category_scatter(x='snr_mn', y='R2', label_col='condition', 
                           #data=df[:-test_size], legend_loc='upper left', colors=(colors[1],colors[0],colors[2]))
    
    #model = OneVsRestClassifier(LinearSVC(random_state=0))
    #model = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
    #model.fit(x_train, y_train.ravel())

    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x_train, y_train.ravel())
    poly = svm.SVC(kernel='poly', degree=3, C=1).fit(x_train, y_train.ravel())

    poly_pred = poly.predict(x_test)
    rbf_pred = rbf.predict(x_test)

    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    
    plot_decision_regions(x_test, y_test.ravel(), ax=axes,clf=poly, legend=2, colors='{},{},{}'.format(colors[0],colors[1],colors[2]))
    axes.set_xlim(np.min(np.array(x_test)-0.1, axis=0)[0],np.max(np.array(x_test), axis=0)[0]+0.1)
    axes.set_ylim(np.min(np.array(x_test), axis=0)[1]-0.01,np.max(np.array(x_test), axis=0)[1]+0.01)
    axes.set_title('s{} (accuracy: {:.2f}%)'.format(title,poly_accuracy*100))

    return poly_accuracy
