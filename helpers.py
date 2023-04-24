import os
import numpy
import pickle
from sklearn.model_selection import KFold

def make_name(s,t,d,m,o,nm,nf,bn,fo,fi):
    return "s{}-t{}-d{}-m{}-o{}-nm{}-nf{}-bn{}-fo{}-fi{}".format(s,t,d,m,o,nm,nf,bn,fo,fi)

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def get_params(i):
    line = numpy.loadtxt('params.txt')[i]
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
    return s,t,d,m,o,nm,nf,bn,fo,fi

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

