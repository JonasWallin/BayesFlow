import unittest
import numpy as np
from BayesFlow import hierarical_mixture_mpi
try:
    from BayesFlow.utils.dat_util import sampnames_mpi
except ImportError:
    raise unittest.SkipTest("sampnames_mpi not available")



kw = {'ext': '.dat', 'loadfilef': lambda filename: np.loadtxt(filename), 'startrow': 0, 'startcol': 0,
      'datadir': '/Users/johnsson/Forskning/Data/healthyFlowData'}

sampnames = sampnames_mpi(kw['datadir'],kw['ext'])

hGMM = hierarical_mixture_mpi(K=10)
hGMM.load_data(sampnames,**kw)
