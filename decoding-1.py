import sys
import bridges/helpers

s,m,ba,n,r,bn,ttv = helpers.get_params(int(sys.argv[1]))
jobname = helpers.make_name(s,m,ba,n,r,bn,ttv)
pfile = helpers.make_filenames(jobname)




