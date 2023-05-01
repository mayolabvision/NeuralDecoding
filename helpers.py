import os
import numpy as np
import decodingSetup
#import pickle
#from sklearn.model_selection import KFold

def make_name(s,t,d,m,o,nm,nf,bn,fo,fi,r):
    #SET,session,timesPrePost,binwidth,model,output,numMT,numFEF,binsPrePost,outerFolds, innerFolds, numRepeats
    #0      1          1         50      1     0     20    20        1          5            5          100
    pname = "s{:02d}-t{}-d{:03d}-m{:02d}-o{}-nm{:02d}-nf{:02d}-bn{}-fo{:02d}-fi{:02d}-r{:04d}".format(s,t,d,m,o,nm,nf,bn,fo,fi,r)
    dname = "s{:02d}-t{}-d{:03d}-o{}-nm{:02d}-nf{:02d}-bn{}-fo{:02d}-r{:04d}".format(s,t,d,m,o,nm,nf,bn,fo,fi,r)
    return pname, dname 

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def get_params(i):
    line = np.loadtxt('params_mlproject.txt')[i]
    print(line)
    s = int(line[1])
    t = int(line[2])
    d = int(line[3])
    m = int(line[4])
    o = int(line[5])
    nm = int(line[6])
    nf = int(line[7])
    bn = int(line[8])
    fo = int(line[9])
    fi = int(line[10]) 
    r = int(line[11])
    return s,t,d,m,o,nm,nf,bn,fo,fi,r

def make_directory(jobname):
    cwd = os.getcwd()
    f="/runs/"+jobname
    if not os.path.isdir(cwd+f):
       os.makedirs(cwd+f)
    return f

def get_session(j,t,d):
    sessions = ['pa29dir4A']
    times = [[500,300]]
    return sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])+'-dt'+str(d),sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])

def get_bins(bn):
    bins = [[6,1,6],[6,1,0]]
    return bins[bn][0],bins[bn][1],bins[bn][2]

def get_outerfold(i,thisFold):
    X_train0, X_flat_train0, y_train0, X_test, X_flat_test, y_test, neurons_perRepeat = decodingSetup.get_dataParams(i)
    return X_train0[thisFold],X_flat_train0[thisFold], y_train0[thisFold], X_test[thisFold], X_flat_test[thisFold], y_test[thisFold], neurons_perRepeat
