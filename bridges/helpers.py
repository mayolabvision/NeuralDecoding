import os
import numpy

def make_name(s,m,ba,n,r,bn,ttv):
    return "s{}-m{}-ba{}-n{}-r{}-bn{}-ttv{}".format(s,m,ba,n,r,bn,ttv)

def checkdir(name):
    if not os.path.exists(name):
        os.makedirs(name)
    return

def get_params(i):
    line = numpy.loadtxt('params.txt')[i]
    print(line)
    s = int(line[1])
    m = int(line[2])
    ba = int(line[3])
    n = int(line[4])
    r = int(line[5])
    bn = int(line[6])
    ttv = int(line[7])
    return s,m,ba,n,r,bn,ttv

def make_filenames(jobname):
    f="runs/"+jobname+".pickle"
    checkdir("runs/"+jobname)
    return f

