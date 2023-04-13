import os
import numpy

def make_name(s,t,d,m,o,nm,nf,r,bn,cf):
    return "s{}-t{}-d{}-m{}-o{}-nm{}-nf{}-r{}-bn{}-cf{}".format(s,t,d,m,o,nm,nf,r,bn,cf)

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
    r = int(line[8])
    bn = int(line[9])
    cf = int(line[10])
    return s,t,d,m,o,nm,nf,r,bn,cf

def make_filenames(jobname):
    f="runs/"+jobname+".pickle"
    return f

def get_session(j,t,d):
    sessions = ['pa29dir4A']
    times = [[500,300],[100,100]]
    return sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])+'-dt'+str(d),sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])


def get_bins(bn):
    bins = [[0,1,0],[6,1,6],[7,1,7]]
    return bins[bn][0],bins[bn][1],bins[bn][2]

def define_outputs(o):
    outputs = [[0,1],[2,3]]
    return outputs[o]
