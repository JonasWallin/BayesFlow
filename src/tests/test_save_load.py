import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from .test_pipeline import Pipeline, SynSample, BadQualityError
from .test_balanced import SaveDict

cachedir = 'src/tests/cache'
if not os.path.exists(cachedir):
    os.mkdir(cachedir)


pipeline = Pipeline(J=5, K=8, N=1000, d=2, C=4,
                    data_class=SynSample, ver='A',
                    par_file='src/tests/param/0.py')
pipeline.run(save_hGMM=True)


