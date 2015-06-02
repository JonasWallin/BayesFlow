from mpi4py import MPI
import numpy as np
import sys
sys.path.append('../src')
from utils import dat_util

comm = MPI.COMM_WORLD
rank = comm.Get_rank()




kw = {'ext': '.dat', 'loadfilef': lambda filename: np.loadtxt(filename), 'startrow': 0, 'startcol': 0,
      'datadir': '/Users/johnsson/Forskning/Data/healthyFlowData'}

sampnames = dat_util.sampnames_mpi(kw['datadir'],kw['ext'])
print "sampnames = {}".format(sampnames)

if 0:
	print "dat_util.total_number_events(sampnames,**kw) at rank {} = {}".format(rank,dat_util.total_number_events(sampnames,**kw))

prc,scaleKey = dat_util.pooled_percentile_mpi(1,sampnames=sampnames,**kw)
print "dat_util.pooled_percentile_mpi(1,sampnames,**kw)[0] = {}".format(prc)

print "dat_util.pooled_percentile_mpi(99,sampnames,**kw) = {}".format(dat_util.pooled_percentile_mpi(99,sampnames=sampnames,**kw))

samp = dat_util.load_fcsample(sampnames[0],**kw)
print "sample with shape {} loaded".format(samp.shape)

for d in range(samp.shape[1]):
	print "np.percentile(samp[:,d],1) = {}".format(np.percentile(samp[:,d],1))
	print "np.percentile(samp[:,d],99) = {}".format(np.percentile(samp[:,d],99))




kw = {'ext': '.dat', 'loadfilef': lambda filename: np.loadtxt(filename), 'startrow': 0, 'startcol': 0,
      'datadir': '/Users/johnsson/Forskning/Data/healthyFlowData','Nevent':1000}

sampnames = dat_util.sampnames_mpi(kw['datadir'],kw['ext'])
print "sampnames = {}".format(sampnames)

if 0:
	print "dat_util.total_number_events(sampnames,**kw) at rank {} = {}".format(rank,dat_util.total_number_events(sampnames,**kw))

print "dat_util.pooled_percentile_mpi(1,sampnames,**kw) = {}".format(dat_util.pooled_percentile_mpi(1,sampnames=sampnames,**kw))

print "dat_util.pooled_percentile_mpi(99,sampnames,**kw) = {}".format(dat_util.pooled_percentile_mpi(99,sampnames=sampnames,**kw))


