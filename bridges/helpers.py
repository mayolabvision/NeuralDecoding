import os
import numpy

def make_name(s,t,m,o,nm,nf,r,bn,ttv):
    return "s{}-t{}-m{}-o{}-nm{}-nf{}-r{}-bn{}-ttv{}".format(s,t,m,o,nm,nf,r,bn,ttv)

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def get_params(i):
    line = numpy.loadtxt('params.txt')[i]
    print(line)
    s = int(line[1])
    t = int(line[2])
    m = int(line[3])
    o = int(line[4])
    nm = int(line[5])
    nf = int(line[6])
    r = int(line[7])
    bn = int(line[8])
    ttv = int(line[9])
    return s,t,m,o,nm,nf,r,bn,ttv

def make_filenames(jobname):
    f="runs/"+jobname+".pickle"
    checkdir("runs/"+jobname)
    return f

def get_session(j,t):
    sessions = ['pa29dir4A']
    times = [[600,600],[700,700]]
    return sessions[j]+'-pre'+str(times[t][0])+'-post'+str(times[t][1])

def get_bins(bn):
    bins = [[0,1,0],[6,1,6],[7,1,7]]
    return bins[bn][0],bins[bn][1],bins[bn][2]

def get_range(ttv):
    ranges = [[0.7,0.15],[0.8,0.1]]
    training_range = [0,ranges[ttv][0]]
    testing_range = [ranges[ttv][0],ranges[ttv][0]+ranges[ttv][1]]
    valid_range = [ranges[ttv][0]+ranges[ttv][1],1]
    return training_range,testing_range,valid_range

def define_outputs(o):
    outputs = [[0,1],[1,2]]
    return outputs[o]
