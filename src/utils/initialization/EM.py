from __future__ import division
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
from sklearn import mixture as skmixture
import matplotlib.pyplot as plt
from mpi4py import MPI
import warnings

try:
    from EarthMover import earth_movers_distance
except ImportError as e:
    pass
    #print("Option 'selection=EMD' will not be available: {}".format(e))

from . import DataMPI, WeightsMPI
from ...PurePython.GMM import mixture
from ...exceptions import EmptyClusterError
from ...plot import component_plot


def EM_pooled(comm, data, K, noise_class=False,
              noise_mu=0.5, noise_sigma=0.5**2, noise_pi=0.001,
              n_iter=10, n_init=5, WIS=False,
              selection='likelihood', gamma=None, plotting=False, **kw):
    if comm.Get_rank() > 0:
        plotting = False

    if WIS:
        return EM_weighted_iterated_subsampling(
            comm, data, K, noise_class=noise_class, noise_mu=noise_mu,
            noise_sigma=noise_sigma, noise_pi=noise_pi, n_iter=n_iter,
            n_init=n_init, selection=selection, gamma=gamma, plotting=plotting, **kw)

    if noise_class:
        d = len(data[0])
        mus_fixed = [noise_mu*np.ones(d)]
        Sigmas_fixed = [noise_sigma*np.eye(d)]
        pis_fixed = [noise_pi]
    else:
        mus_fixed = []
        Sigmas_fixed = []
        pis_fixed = []

    mus, Sigmas, pis = EM_pooled_fixed(comm, data, K, n_iter, n_init, mus_fixed,
                                       Sigmas_fixed, pis_fixed, selection, gamma=gamma)

    if plotting:
        fig, ax = plt.subplots()
        ax.scatter(data[0][:, 0], data[0][:, 1])
        component_plot(mus, Sigmas, [0, 1], ax, colors=[(1, 0, 0)]*len(mus), lw=2)

    return mus, Sigmas, pis


#@profile
def EM_pooled_fixed(comm, data, K, n_iter=10, n_init=5,
                    mus_fixed=None, Sigmas_fixed=None, pis_fixed=None,
                    selection='likelihood', gamma=None):
    """
        Fitting GMM with EM algorithm with fixed components

        comm            - MPI communicator.
        data            - list of data sets.
        K               - number of components.
        n_iter          - number of EM steps.
        n_init          - number of random initializations.
        mus_fixed       - mu values for fixed components.
        Sigmas_fixed    - Sigma values for fixed components.
        pis_fixed       - pi values for fixed components. Proportions
                          will be fixed if values are not nan, but not
                          fixed when pi is nan.
    """

    if mus_fixed is None:
        mus_fixed = []
    if Sigmas_fixed is None:
        Sigmas_fixed = []
    if pis_fixed is None:
        pis_fixed = []

    mu0, Sigma0, _ = E_step_pooled(
        comm, data, [np.array([1./K for i in range(dat.shape[0])]).reshape(-1, 1)
                     for dat in data])
    d = data[0].shape[1]
    if selection == 'likelihood':
        max_log_lik = -np.inf
    elif selection == 'EMD':
        min_emd = np.inf
        data_mpi = DataMPI(comm, data)
        N_synsamp = int(sum(data_mpi.n_j)*.1)
    elif selection == 'sum_min_dist':
        max_sum_min_dist = -np.inf
    else:
        raise ValueError("Selection {} not possible".format(selection))

    K_fix = len(mus_fixed)
    K -= K_fix
    k_pi_fixed = [K+k for k, pi_k in enumerate(pis_fixed) if
                  not np.isnan(pi_k)]

    for init in range(n_init):
        mus = stats.multivariate_normal.rvs(mu0, Sigma0, size=K)
        mus = mus.reshape(-1, d).tolist() + mus_fixed
        Sigmas = [Sigma0 for k in range(K)] + Sigmas_fixed
        pis = np.array([np.nan for k in range(K)] + pis_fixed)
        pis = normalize_pi(pis, k_pi_fixed)
        for it in range(n_iter):
            weights = M_step_pooled(comm, data, mus, Sigmas, pis)
            for k in range(K):
                try:
                    mus[k], Sigmas[k], pis[k] = E_step_pooled(
                        comm, data, [weight[:, k] for weight in weights])
                except EmptyClusterError:
                    mus[k] = stats.multivariate_normal.rvs(mu0, Sigma0)
                    Sigmas[k] = Sigma0
                    pis[k] = 0.01
            for k_fix in range(K, K+K_fix):
                if not k_fix in k_pi_fixed:
                    pis[k] = WeightsMPI(comm, [weight[:, k] for weight in weights]).W
            pis = normalize_pi(pis, k_pi_fixed)

        if selection == 'likelihood':
            log_lik_loc = np.sum([np.sum(np.log(np.sum(weight, axis=1))) for weight in weights])
            log_lik = sum(comm.bcast(comm.gather(log_lik_loc)))
            if log_lik > max_log_lik:
                best_mus, best_Sigmas, best_pis = mus, Sigmas, pis
                max_log_lik = log_lik

        elif selection == 'EMD':
            emd = max(EMD_to_generated_from_model(data_mpi, mus, Sigmas, pis,
                                                  N_synsamp, gamma))
            if emd < min_emd:
                best_mus, best_Sigmas, best_pis = mus, Sigmas, pis
                min_emd = emd

        elif selection == 'sum_min_dist':
            mus_all = comm.gather(mus)
            if comm.Get_rank() == 0:
                mus_all = np.vstack(mus_all)
                L = mus_all.shape[0]
                D = np.empty((L, L))
                for i in range(L):
                    for j in range(L):
                        D[i, j] = np.linalg.norm(mus_all[i] - mus_all[j])
                sum_min_dist = np.sum(np.min(D, axis=1))
            else:
                sum_min_dist = None
            sum_min_dist = comm.bcast(sum_min_dist)
            if sum_min_dist > max_sum_min_dist:
                best_mus, best_Sigmas, best_pis = mus, Sigmas, pis
                max_sum_min_dist = sum_min_dist

    return best_mus, best_Sigmas, best_pis


def EMD_to_generated_from_model(data_mpi, mus, Sigmas, pis, N_synsamp, gamma=1,
                                nbins=50, dims=None):
    comm = data_mpi.comm
    d = data_mpi.d
    if dims is None:
        dims = [(i, j) for i in range(d) for j in range(i+1, d)]
    real_data = data_mpi.subsample_to_root(N_synsamp)
    if comm.Get_rank() == 0:
        syn_data = mixture.simulate_mixture(mus, Sigmas, pis, N_synsamp)
        emd = [earth_movers_distance(syn_data, real_data, nbins=nbins, dim=dim,
                                     gamma=gamma)
               for dim in dims]
    else:
        emd = None
    return comm.bcast(emd)


def data_log_likelihood(data_mpi, mus, Sigmas, pis):
    comm = data_mpi.comm
    weights = M_step_pooled(data_mpi.comm, data_mpi.data, mus, Sigmas, pis)
    log_lik_loc = np.sum([np.sum(np.log(np.sum(weight, axis=1))) for weight in weights])
    log_lik = sum(comm.bcast(comm.gather(log_lik_loc)))
    return log_lik


#@profile
def EM_weighted_iterated_subsampling(comm, data, K, N, noise_class=False,
                                     noise_mu=0.5, noise_sigma=0.5**2, noise_pi=0.001,
                                     n_iter=10, iter_final=2, n_init=5, rho=3,
                                     likelihood_weights=True,
                                     plotting=False, selection='likelihood', gamma=None):
    K_it = int(np.ceil(K/n_iter))
    data_mpi = DataMPI(comm, data)

    if noise_class:
        mus_fixed = [noise_mu*np.ones(data_mpi.d)]
        Sigmas_fixed = [noise_sigma*np.eye(data_mpi.d)]
        pis_fixed = [noise_pi]
    else:
        mus_fixed = []
        Sigmas_fixed = []
        pis_fixed = []
    K_fix = len(mus_fixed)

    data_subsamp = data

    while K_fix < K+int(noise_class):
        mus, Sigmas, pis = EM_pooled_fixed(
            comm, data_subsamp, K+int(noise_class), mus_fixed=mus_fixed,
            Sigmas_fixed=Sigmas_fixed, pis_fixed=pis_fixed, selection=selection,
            n_init=n_init, gamma=gamma)

        if plotting:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(data_subsamp[0][:, 0], data_subsamp[0][:, 1])
            component_plot(mus, Sigmas, [0, 1], ax, colors=[(1, 0, 0)]*len(mus), lw=2)
            component_plot(mus_fixed, Sigmas_fixed, [0, 1], ax,
                           colors=[(0, 1, 0)]*len(mus_fixed), lw=2)

        k_fixed = np.argpartition(-pis[:len(pis)-K_fix], K_it-1)[:K_it]
        mus_fixed += [mus[k] for k in k_fixed]
        Sigmas_fixed += [Sigmas[k] for k in k_fixed]
        pis_fixed += [np.nan for k in k_fixed]
        if not likelihood_weights:
            weights = M_step_pooled(comm, data, mus, Sigmas, pis)
            weights_subsamp = [1-(np.sum(weight[:, weight.shape[1]-K_fix:], axis=1)
                                  + sum([weight[:, k] for k in k_fixed]))
                               for weight in weights]
        else:
            weights = component_likelihoods(data, mus_fixed, Sigmas_fixed)
            #fig = plt.figure()
            #plt.hist(weights)
            weights_subsamp = [np.abs((-np.log((np.sum(weight, axis=1)))))**rho for weight in weights]
        data_subsamp = data_mpi.subsample_weighted(weights_subsamp, N)

        K_fix = len(mus_fixed)

    # Extra EM it
    # print("before extra EM it")
    mus, Sigmas = mus_fixed, Sigmas_fixed
    pis = np.array([1./K for k in range(K)])
    if noise_class:
        pis = np.hstack([noise_pi, pis])
        pis = normalize_pi(pis, [0])

    for it in range(iter_final):
        weights = M_step_pooled(comm, data, mus, Sigmas, pis)
        for k in range(int(noise_class), weights[0].shape[1]):
            mus[k], Sigmas[k], pis[k] = E_step_pooled(comm, data,
                                                      [weight[:, k] for weight in weights])
        if not noise_class:
            pis = normalize_pi(pis)
        else:
            pis = normalize_pi(pis, [0])
    if plotting:
        plt.show()

    if noise_class:
        mus = mus[1:]
        Sigmas = Sigmas[1:]
        pis = pis[1:]

    return mus, Sigmas, pis


#@profile
def E_step_pooled(comm, data, weights):
    weights = [ma.masked_array(weight, np.isnan(weight)) if np.isnan(weight).any()
               else weight for weight in weights]
    weights_mpi = WeightsMPI(comm, weights)
    if weights_mpi.W == 0:
        raise EmptyClusterError
    #print("tot weight of cluster = {}".format(weights_mpi.W))
    mu_loc = sum([np.sum(dat*weight.reshape(-1, 1), axis=0)
                  for weight, dat in zip(weights, data)])
    mu = sum(comm.bcast(comm.gather(mu_loc)))/weights_mpi.W
    if weights_mpi.W < data[0].shape[1]:
        Sigma = np.eye(data[0].shape[1])
    else:
        if weights[0].shape == (data[0].shape[0], 1):
            #wXXT_loc = sum([(dat*weight).T.dot(dat) for (weight, dat) in
            #               zip(weights, data)])
            wXXT_loc = sum([np.ma.dot((dat*weight).T, dat) for (weight, dat) in
                           zip(weights, data)])
        elif weights[0].shape == (data[0].shape[0], ):
            #wXXT_loc = sum([(dat*weight[:, np.newaxis]).T.dot(dat)
            #                for (weight, dat) in zip(weights, data)])
            wXXT_loc = sum([np.ma.dot((dat*weight[:, np.newaxis]).T, dat)
                            for (weight, dat) in zip(weights, data)])
        else:
            raise ValueError("weight has shape {}".format(weights[0].shape))
        # wXXT_loc = np.zeros((data[0].shape[1], data[0].shape[1]))
        # for j, dat in enumerate(data):
        #     if weights[j].shape == (dat.shape[0], 1):
        #         wXXT_loc += (dat*weights[j]).T.dot(dat)
        #     elif weights[j].shape == (dat.shape[0],):
        #         wXXT_loc += (dat*weights[j][:, np.newaxis]).T.dot(dat)
        #     else:
        #         raise ValueError("weight has shape {}".format(weights[j].shape))
        wXXT = sum(comm.bcast(comm.gather(wXXT_loc)))/weights_mpi.W
        Sigma = wXXT - mu.reshape(-1, 1).dot(mu.reshape(1, -1))

    return mu, Sigma, weights_mpi.W


#@profile
def M_step_pooled(comm, data, mus, Sigmas, pis):
    K = len(mus)
    #print("mus = {}".format(mus))
    #print("Sigmas = {}".format(Sigmas))
    weights = [np.empty((dat.shape[0], K)) for dat in data]
    for j, dat in enumerate(data):
        for k in range(K):
            #try:
            weights[j][:, k] = stats.multivariate_normal.pdf(dat, mus[k],
                                                             Sigmas[k])
            #except ValueError:
            #    weights[j][:, k] = 0
    for weight in weights:
        weight *= pis
        weight /= np.sum(weight, axis=1).reshape(-1, 1)
    return weights


def component_likelihoods(data, mus, Sigmas):
    K = len(mus)
    weights = [np.empty((dat.shape[0], K)) for dat in data]
    for j, dat in enumerate(data):
        for k in range(K):
            weights[j][:, k] = stats.multivariate_normal.pdf(dat, mus[k],
                                                             Sigmas[k])
    return weights


def normalize_pi(p, k_fixed=[]):
    p[np.isnan(p)] = 1./len(p)
    p_fixed = sum(p[k] for k in k_fixed)
    W = sum([p_k for k, p_k in enumerate(p) if not k in k_fixed])
    for k, p_k in enumerate(p):
        if not k in k_fixed:
            p[k] *= (1-p_fixed)/W
    return p


def GMM_means_for_best_BIC(data, Ks, n_init=10, n_iter=100, covariance_type='full'):
    data = data[~np.isnan(data[:, 0]), :]
    data = data[~np.isinf(data[:, 0]), :]
    bestbic = np.inf
    for K in Ks:
        g = skmixture.GMM(n_components=K, covariance_type=covariance_type, n_init=n_init, n_iter=n_iter)
        g.fit(data)
        bic = g.bic(data)
        print("BIC for {} clusters: {}".format(K, bic))
        if bic < bestbic:
            means = g.means_
            bestbic = bic
    if means.shape[0] == np.max(Ks):
        warnings.warn("Best BIC obtained for maximum K")
    if means.shape[0] == np.min(Ks):
        warnings.warn("Best BIC obtained for minimum K")
    return means


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if 1:
        data = DataMPI(MPI.COMM_WORLD, [np.eye(3) for k in range(3)])
        weights = [range(3) for k in range(3)]
        print("data.J_loc = {}".format(data.J_loc))
        print("data.J = {}".format(data.J))
        print("data.n_j = {}".format(data.n_j))
        print("data.subsample_from_each_to_root(2) = {}".format(data.subsample_from_each_to_root(2)))
        print("data.subsample_weighted_to_root(weights, 20) = {}".format(data.subsample_weighted_to_root(weights, 20)))
        print("data.subsample_to_root(5) = {}".format(data.subsample_to_root(5)))

    if 0:
        pi = np.array([1, 3, 5, 0.1, 0.2])
        k_fixed = [3, 4]
        print("normalize_pi(pi, k_fixed) = {}".format(normalize_pi(pi, k_fixed)))
        print("sum(normalize_pi(pi, k_fixed)) = {}".format(sum(normalize_pi(pi, k_fixed))))

        pi = np.array([1, 3, 5, 0.1, 0.2, np.nan])
        k_fixed = [3, 4]
        print("normalize_pi(pi, k_fixed) = {}".format(normalize_pi(pi, k_fixed)))
        print("sum(normalize_pi(pi, k_fixed)) = {}".format(sum(normalize_pi(pi, k_fixed))))
    if 0:
        d = 2
        mus = [m*np.ones(d)+np.random.normal(0, m*0.1) for m in [1, 2, 6]]
        Sigmas = [m*np.eye(d) for m in [1, 0.5, 3]]
        #mus = [m*np.ones(d)+np.random.normal(0, 0.1) for m in [0, 1, 10]]
        #Sigmas = [m*np.eye(d) for m in [1, 0.5, 0.1]]
        pis = np.array([10000, 10000, 100])
        pis = pis/np.sum(pis)
        N = 10000
        data = [mixture.simulate_mixture(mus, Sigmas, pis, N)]
        K = 5
        (mus_fitted, Sigmas_fitted,
         pis_fitted) = EM_weighted_iterated_subsampling(comm, data, K, N/10,
                                                        n_iter=3, iter_final=3,
                                                        plotting=False)
        # print("mus_fitted = {}".format(mus_fitted))
        # print("Sigmas_fitted = {}".format(Sigmas_fitted))
        # print("pis_fitted = {}".format(pis_fitted))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[0][:, 0], data[0][:, 1])
        component_plot(mus_fitted, Sigmas_fitted, [0, 1], ax,
                       colors=[(1, 1, 0)]*len(mus_fitted), lw=2)
        plt.show()

    if 0:
        # Test with noise class
        d = 2
        mus = [m*np.ones(d)+np.random.normal(0, 0.1) for m in [1, 2, 6, 4.5]]
        Sigmas = [m*np.eye(d) for m in [1, 0.5, 3, 9**2]]
        #mus = [m*np.ones(d)+np.random.normal(0, 0.1) for m in [0, 1, 10]]
        #Sigmas = [m*np.eye(d) for m in [1, 0.5, 0.1]]
        pis = np.array([10000, 10000, 100])
        pis = pis/np.sum(pis)
        N = 10000
        data = [mixture.simulate_mixture(mus, Sigmas, pis, N)]
        K = 5
        (mus_fitted, Sigmas_fitted,
         pis_fitted) = EM_weighted_iterated_subsampling(comm, data, K, N/10,
                                                        noise_class=True,
                                                        noise_mu=5, noise_sigma=10**2,
                                                        n_iter=3, iter_final=3)
        # print("mus_fitted = {}".format(mus_fitted))
        # print("Sigmas_fitted = {}".format(Sigmas_fitted))
        # print("pis_fitted = {}".format(pis_fitted))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data[0][:, 0], data[0][:, 1])
        component_plot(mus_fitted, Sigmas_fitted, [0, 1], ax,
                       colors=[(1, 1, 0)]*len(mus_fitted), lw=2)
        plt.show()
