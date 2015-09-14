import numpy as np
from mpi4py import MPI

from .. import LazyProperty
from .. import load_fcdata
from .. import mpiutil


class WeightsMPI(object):

    def __init__(self, comm, weights):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.weights = weights

    @property
    def ws_loc(self):
        return [np.sum(w) for w in self.weights]

    @property
    def W_loc(self):
        return sum([np.sum(w) for w in self.weights])

    @property
    def W_locs(self):
        return self.comm.gather(self.W_loc)

    @LazyProperty
    def W(self):
        return sum(self.comm.bcast(self.W_locs))

    def sample_from_all(self, N):
        W = self.W
        W_locs = self.W_locs
        if self.rank == 0:
            alpha = np.sort(W*np.random.random(N))
            ind_W = [0]+list(np.searchsorted(alpha, np.cumsum(W_locs)))
            W_cum = 0.
            for i in range(self.comm.Get_size()):
                alpha_i = alpha[ind_W[i]:ind_W[i+1]] - W_cum
                W_cum += W_locs[i]
                self.comm.send(alpha_i, i)
        alpha_loc = self.comm.recv()
        #print "alpha_loc retrieved"
        ind_w = [0]+list(np.searchsorted(alpha_loc, np.cumsum(self.ws_loc)))
        w_cum = 0.
        indices = []
        for i, weights_samp in enumerate(self.weights):
            alpha_samp = alpha_loc[ind_w[i]:ind_w[i+1]] - w_cum
            w_cum += self.ws_loc[i]
            indices.append(self.retrieve_ind_at_alpha(weights_samp, alpha_samp))
        return indices

    @staticmethod
    def retrieve_ind_at_alpha(weights, alpha):
        i = 0
        j = 0
        indices = np.empty(len(alpha), dtype='i')
        if len(alpha) == 0:
            return indices
        cumweight = weights[0]
        while i < len(alpha):
            if alpha[i] <= cumweight:
                indices[i] = j
                i += 1
            else:
                j += 1
                cumweight += weights[j]
        return indices


class DataMPI(object):

    def __init__(self, comm, data=None):
        self.comm = comm
        self.rank = comm.Get_rank()
        if not data is None:
            self.data = data
        else:
            self.data = []

    @property
    def J_loc(self):
        return len(self.data)

    @property
    def J_locs(self):
        return self.comm.bcast(self.comm.gather(self.J_loc))

    @property
    def J(self):
        return(sum(self.J_locs))

    @property
    def d(self):
        return self.data[0].shape[1]

    @property
    def n_j_loc(self):
        return [dat.shape[0] for dat in self.data]

    @property
    def N_loc(self):
        return sum(self.n_j_loc)

    @property
    def n_j(self):
        n_j = self.comm.gather(self.n_j_loc)
        if self.rank == 0:
            n_j = [nj for nj_w in n_j for nj in nj_w]
        return self.comm.bcast(n_j)

    @property
    def N_locs(self):
        return self.comm.gather(self.N_loc)

    def load(self, sampnames, scale='percentilescale', q=(1, 99), **kw):
        self.data = load_fcdata(sampnames, scale, (1, 99), self.comm, **kw)

    def subsample_to_root(self, N):
        n_j = self.n_j
        J_locs = self.J_locs
        if self.rank == 0:
            ps = np.array(n_j)/sum(n_j)
            ps = [p/N for p in ps]
            n_samp = np.random.multinomial(N, ps)
            n_samp = np.split(n_samp, np.cumsum(J_locs[:-1]))
        else:
            n_samp = None
        n_samp_loc = self.comm.scatter(n_samp)
        data_subsamp = np.vstack([self.data[j][np.random.choice(self.data[j].shape[0], n, replace=True), :]
                                  for j, n in enumerate(n_samp_loc)])
        return mpiutil.collect_data(data_subsamp, self.d, np.float, MPI.DOUBLE,
                                    self.comm)

    def subsample_from_each_to_root(self, N):
        data_subsamp_loc = np.vstack([dat[np.random.choice(dat.shape[0], N, replace=False), :]
                                      for dat in self.data])
        return mpiutil.collect_data(data_subsamp_loc, self.d, np.float,
                                    MPI.DOUBLE, self.comm)

    def subsample_weighted_to_root(self, weights, N):
        weightsMPI = WeightsMPI(self.comm, weights)
        print "tot weight for subsampling {}".format(weightsMPI.W)
        indices = weightsMPI.sample_from_all(N)
        data_loc = np.vstack([self.data[j][ind, :]
                              for j, ind in enumerate(indices)])
        return mpiutil.collect_data(data_loc, self.d, np.float,
                                    MPI.DOUBLE, self.comm)

    def subsample_weighted(self, weights, N):
        weightsMPI = WeightsMPI(self.comm, weights)
        print "tot weight for subsampling {}".format(weightsMPI.W)
        indices = weightsMPI.sample_from_all(N)
        return [self.data[j][ind, :] for j, ind in enumerate(indices)]
