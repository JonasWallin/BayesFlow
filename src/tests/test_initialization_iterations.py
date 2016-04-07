import os
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

from .pipeline import Pipeline, SynSample2, BadQualityError
from .test_modified_label_sw import TrackDict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    fname = 'src/tests/cache/initialization.pkl'
    if not os.path.exists('src/tests/cache'):
        os.mkdir('src/tests/cache')
    quality_res = TrackDict(fname)

    fname = 'src/tests/cache/initialization_runs.pkl'
    run_res = TrackDict(fname)

n_iters = [50, 100, 200, 500]

if 0:
    for i in range(10):
        for n_iter in n_iters:
            pipeline = Pipeline(J=5, K=15, N=1000, d=3,
                                data_class=SynSample2, ver='C', copy_data=True,
                                par_file='src/tests/param/0.py', comm=comm)
            pipeline.run(init_method='EM_pooled', WIS=False, rho=2, init_n_iter=n_iter,
                         n_init=10, init_plotting=False, init_selection='likelihood', gamma=2)
            if rank == 0:
                run_res.append_to(n_iter, (pipeline.run_nbr, pipeline.logdata['t_init']))
                run_res.print_table()

if rank == 0:
    emd_all = []
    log_lik_all = []
    for n_iter in n_iters:

        if 1:
            runinfos = run_res.get(n_iter)
            for (run_nbr, time) in runinfos:
                quality_res.append_to('Initialization time, iterations: {}, '.format(n_iter),
                                      time)
                pipeline = Pipeline(J=5, K=15, N=1000, d=3, data_class=SynSample2, ver='B',
                                    par_file='src/tests/param/0.py', run=run_nbr, comm=comm,
                                    copy_data=True)
                pipeline.setup_run()
                pipeline.postproc()
                try:
                    (emd, log_lik) = pipeline.quality_check()
                except BadQualityError:
                    quality_res.add_to('Iterations: {}, fail'.format(n_iter))
                else:
                    quality_res.add_to('Iterations: {}, pass'.format(n_iter))
                    quality_res.append_to("Earth Mover's distance, iterations {}".format(n_iter), emd)
                    quality_res.append_to("Log likelihood, iterations {}".format(n_iter), log_lik)
        emd_all.append(quality_res.get("Earth Mover's distance, iterations {}".format(n_iter)))
        log_lik_all.append(quality_res.get("Log likelihood, iterations {}".format(n_iter)))

    quality_res.print_table()

    fig, axs = plt.subplots(2)
    axs[0].boxplot(emd_all)
    axs[1].boxplot(log_lik_all)
    for ax in axs.reshape(-1):
        ax.xaxis.set_ticks(np.arange(len(n_iters))+1)
        ax.set_xticklabels(n_iters)
    plt.show()
