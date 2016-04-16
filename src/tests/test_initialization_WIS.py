import numpy as np
import time
import matplotlib.pyplot as plt
from mpi4py import MPI

from .pipeline import SynSample2
from ..utils.initialization.EM import EM_pooled
from ..utils.initialization.distributed_data import DataMPI
from ..plot import component_plot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_obs = 5000
d = 2
J = 5
K = 15

n_iter = 50
n_init = 10

if rank == 0:
    js = np.array_split(np.arange(J), comm.Get_size())
else:
    js = None
js = comm.scatter(js)
print "js at rank {} = {}".format(rank, js)

samples = [SynSample2(j, n_obs, d=d, ver='C') for j in js]
data = [sample.data for sample in samples]

t0 = time.time()
res_em = EM_pooled(comm, data, K=K, n_iter=n_iter, n_init=n_init)
t1 = time.time()
res_wis = EM_pooled(comm, data, K=K, WIS=True, n_iter=n_iter, n_init=n_init, rho=4,
                    N=n_obs/10)
t2 = time.time()

res_em_maxd = EM_pooled(comm, data, K=K, n_iter=n_iter, n_init=n_init,
                        selection='sum_min_dist')
res_wis_maxd = EM_pooled(comm, data, K=K, WIS=True, n_iter=n_iter, n_init=n_init, rho=4,
                         N=n_obs/10, selection='sum_min_dist')

if rank == 0:
    print "Time EM: {}".format(t1-t0)
    print "Time EMWIS: {}".format(t2-t1)

plotdim = [(i, j) for i in range(d) for j in range(i+1, d)]
if rank == 0:
    fig, axs = plt.subplots(4, len(plotdim), squeeze=False)
else:
    axs = None
axs = comm.bcast(axs)

for i, dim in enumerate(plotdim):
    for j, res in enumerate((res_em, res_wis, res_em_maxd, res_wis_maxd)):
        DataMPI(comm, data).scatter(dim, ax=axs[j, i], marker='+')
        if rank == 0:
            component_plot(res[0], res[1], dim=dim, ax=axs[j, i], colors=[(1, 0, 0)]*len(res[0]))
plt.show()
