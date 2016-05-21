import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

from .pipeline import Pipeline, SynSample

Js = [2, 5, 50]
Ks = [2, 4, 8, 16, 32]
ds = [3, 6, 9]
Ns = [100, 1000, 10000]

param0 = (5, 1, 1000, 3)
C = 1

par_files = ['src/tests/param/0.py',
             'src/tests/param/1.py',
             'src/tests/param/2.py']


class SaveDict(object):

    def __init__(self, savefile):
        self.savefile = savefile
        if not os.path.exists(self.savefile):
            self._dic = {}
            self.save()

    def update(self, key, value):
        self.load()
        self._dic.update({key: value})
        self.save()

    def save(self):
        with open(self.savefile, 'w') as f:
            pickle.dump(self._dic, f, -1)

    def load(self):
        with open(self.savefile, 'r') as f:
            self._dic = pickle.load(f)

    def get(self, key):
        self.load()
        return self._dic[key]

    def keys(self):
        self.load()
        return self._dic.keys()

if __name__ == '__main__':

    fname = 'src/tests/cache/dist_dict_11.pkl'
    if not os.path.exists('src/tests/cache'):
        os.mkdir('src/tests/cache')
    dists = SaveDict(fname)

    for i, par in enumerate(par_files):
        J, K, N, d = param0
        for N in Ns:
            if not (i, J, K, N, d) in dists.keys():
                print("computing dists for {}".format((i, J, K, N, d)))
                pipeline = Pipeline(J=J, K=K, N=N, d=d, C=C,
                                    data_class=SynSample, ver='A',
                                    par_file=par)
                pipeline.run()
                print("pipeline.rundir = {}".format(pipeline.rundir))
                dists.update((i, J, K, N, d), (pipeline.res.get_bh_distance_to_own_latent(),
                             pipeline.res.get_center_dist()))

        J, K, N, d = param0
        for d in ds:
            if not (i, J, K, N, d) in dists.keys():
                print("computing dists for {}".format((i, J, K, N, d)))
                pipeline = Pipeline(J=J, K=K, N=N, d=d, C=C,
                                    data_class=SynSample, ver='A',
                                    par_file=par)
                pipeline.run()
                print("pipeline.rundir = {}".format(pipeline.rundir))
                dists.update((i, J, K, N, d), (pipeline.res.get_bh_distance_to_own_latent(),
                             pipeline.res.get_center_dist()))

        # J, K, N, d = 5, 4, 1000, 3
        # for K in Ks:
        #     if not (i, J, K, N, d) in dists.keys():
        #         print("computing dists for {}".format((i, J, K, N, d)))
        #         pipeline = Pipeline(J=J, K=K, N=N, d=d, C=C, data_class=SynSample, ver='A')
        #         pipeline.run()
        #         print("pipeline.rundir = {}".format(pipeline.rundir))
        #         dists.update((i, J, K, N, d), (pipeline.res.get_bh_distance_to_own_latent(),
        #                      pipeline.res.get_center_dist()))

        J, K, N, d = param0
        for J in Js:
            if not (i, J, K, N, d) in dists.keys():
                print("computing dists for {}".format((i, J, K, N, d)))
                pipeline = Pipeline(J=J, K=K, N=N, d=d, C=C,
                                    data_class=SynSample, ver='A',
                                    par_file=par)
                pipeline.run()
                print("pipeline.rundir = {}".format(pipeline.rundir))
                dists.update((i, J, K, N, d), (pipeline.res.get_bh_distance_to_own_latent(),
                             pipeline.res.get_center_dist()))

    print("dists.keys() = {}".format(dists.keys()))

    is_ = [0, 1, 2]
    Js = [2, 5, 50]
    Ks = [2, 4, 8, 16, 32]
    ds = [3, 6, 9]
    Ns = [100, 1000, 10000]

    param0 = 5, 1, 1000, 3
    J, K, N, d = param0

    fig, axs = plt.subplots(2, 3)
    for i in is_:
        bh_data = []
        eu_data = []
        for J in Js:
            bh_dat, eu_dat = dists.get((i, J, K, N, d))
            bh_data.append(bh_dat)
            eu_data.append(eu_dat)
        axs[0, i].boxplot(bh_data)
        axs[1, i].boxplot(eu_data)
        for ax in axs.reshape(-1):
            ax.xaxis.set_ticks(np.arange(len(Js))+1)
            ax.set_xticklabels(Js)
    fig.suptitle('Varying J')

    # J, K, N, d = 5, 4, 1000, 3

    # fig, axs = plt.subplots(2, 3)
    # for i in is_:
    #     bh_data = []
    #     eu_data = []
    #     for K in Ks:
    #         bh_dat, eu_dat = dists.get((i, J, K, N, d))
    #         bh_data.append(bh_dat)
    #         eu_data.append(eu_dat)
    #     axs[0, i].boxplot(bh_data)
    #     axs[1, i].boxplot(eu_data)
    #     for ax in axs.reshape(-1):
    #         ax.xaxis.set_ticks(np.arange(len(Ks))+1)
    #         ax.set_xticklabels(Ks)
    # fig.suptitle('Varying K')

    J, K, N, d = param0

    fig, axs = plt.subplots(2, 3)
    for i in is_:
        bh_data = []
        eu_data = []
        for N in Ns:
            bh_dat, eu_dat = dists.get((i, J, K, N, d))
            bh_data.append(bh_dat)
            eu_data.append(eu_dat)
        axs[0, i].boxplot(bh_data)
        axs[1, i].boxplot(eu_data)
        for ax in axs.reshape(-1):
            ax.xaxis.set_ticks(np.arange(len(Ns))+1)
            ax.set_xticklabels(Ns)
    fig.suptitle('Varying N')

    J, K, N, d = param0

    fig, axs = plt.subplots(2, 3)
    for i in is_:
        bh_data = []
        eu_data = []
        for d in ds:
            bh_dat, eu_dat = dists.get((i, J, K, N, d))
            bh_data.append(bh_dat)
            eu_data.append(eu_dat)
        axs[0, i].boxplot(bh_data)
        axs[1, i].boxplot(eu_data)
        for ax in axs.reshape(-1):
            ax.xaxis.set_ticks(np.arange(len(ds))+1)
            ax.set_xticklabels(ds)
    fig.suptitle('Varying d')

    plt.show()
