import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from .pipeline import SynSample
from ..utils.initialization.EM import (EM_pooled, EMD_to_generated_from_model,
                                       data_log_likelihood)
from ..utils.initialization.distributed_data import DataMPI
from ..plot import component_plot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

J = 5
K = 8
N = 1000
d = 2
C = 4

L = 5  # Number of initializations to test

if rank == 0:
    js = np.array_split(np.arange(J), comm.Get_size())
else:
    js = None
js = comm.scatter(js)

sample = [SynSample(j, N, d, C, ver='A') for j in js]
data = [samp.data for samp in sample]
res = []

for i in range(L):
    res.append(EM_pooled(comm, data, K, n_iter=50, n_init=1, WIS=False))

dim = [0, 1]
EMD = np.empty(len(res))
log_lik = np.empty(len(res))

## Evaluation
for i, mix in enumerate(res):
    data_mpi = DataMPI(comm, data)
    EMD[i] = EMD_to_generated_from_model(data_mpi, mix[0], mix[1], mix[2], N_synsamp=N/10, gamma=1)
    log_lik[i] = data_log_likelihood(data_mpi, mix[0], mix[1], mix[2])
    _, ax = plt.subplots()
    data_mpi.scatter(dim, ax)
    if rank == 0:
        component_plot(mix[0], mix[1], dim, ax=ax)


if rank == 0:
    plt.show()
    print "Order initializations:"
    visual_order = raw_input().split()
    print "visual_ranking = {}".format(visual_order)

    for ind in visual_order:
        ind = int(ind)-1
        print "{:<{width}.1f}{:<{width}.1e}".format(EMD[ind], log_lik[ind], width=8)
