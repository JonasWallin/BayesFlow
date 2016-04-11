import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from .pipeline import Pipeline, SynSample, BadQualityError
from .test_balanced import SaveDict


def main():
    C = 4

    fname = 'src/tests/cache/nbr_comp_test_C{}_onoff.pkl'.format(C)
    if not os.path.exists('src/tests/cache'):
        os.mkdir('src/tests/cache')

    res = SaveDict(fname)

    Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for K in Ks:

        if not K in res.keys():
            pipeline = Pipeline(J=5, K=K, N=1000, d=2, C=C,
                                data_class=SynSample, ver='A',
                                par_file='src/tests/param/0.py')
            pipeline.run()
            if rank == 0:
                try:
                    emd, log_lik = pipeline.quality_check()
                except BadQualityError as e:
                    print e
                else:
                    K_active = pipeline.res.K_active
                    print "K_active = {}".format(K_active)
                    res.update(K_active, (emd, log_lik))

    if rank == 0:
        fig, axs = plt.subplots(2)
        emd_data = []
        log_lik_data = []
        for K in res.keys():
            emd_dat, log_lik_dat = res.get(K)
            emd_data.append(emd_dat)
            log_lik_data.append(log_lik_dat)
        print "log_lik_data[2] = {}".format(log_lik_data[2])
        axs[0].boxplot(emd_data)
        axs[1].boxplot(log_lik_data)
        for ax in axs.reshape(-1):
            ax.xaxis.set_ticks(np.arange(len(Ks))+1)
            ax.set_xticklabels(Ks)
        fig.suptitle('Varying K')
        plt.show()

if __name__ == '__main__':
    main()
