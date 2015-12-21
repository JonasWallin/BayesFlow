from mpi4py import MPI
import copy
import numpy as np

from .. import hierarical_mixture_mpi


def extract_from_prior(prior, ks, dims):
    '''
        ts, Sks - k is fist dim, dim is second dim
        n_theta, Q, n_Psi, H - k is only dim
    '''
    prior_new = copy.deepcopy(prior)
    prior_new.t = prior_new.t[ks, :][:, dims]
    prior_new.S = prior_new.S[ks, :][:, dims]
    prior_new.n_theta = prior_new.n_theta[ks]
    prior_new.n_Psi = prior_new.n_Psi[ks]
    prior_new.Q = prior_new.Q[ks]
    prior_new.H = prior_new.H[ks]
    return prior_new


def extract_from_hGMM(hGMM, ks, dims, alldata=True):

    if alldata:
        data = [np.ascontiguousarray(gmm.data[:, dims]) for gmm in hGMM.GMMs]
    else:
        data = [np.ascontiguousarray(gmm.data[gmm.x in ks, :][:, dims]) for gmm in hGMM.GMMs]
    names = [gmm.name for gmm in hGMM.GMMs]

    hGMM_new = hierarical_mixture_mpi(K=len(ks), data=data, sampnames=names,
                                      prior=extract_from_prior(hGMM.prior, ks, dims),
                                      high_memory=hGMM.high_memory, timing=hGMM.timing,
                                      comm=MPI.COMM_WORLD, init=False)

    if hGMM.rank == 0:
        hGMM_new.normal_p_wisharts = [hGMM.normal_p_wisharts[k] for k in ks]
        hGMM_new.wishart_p_nus = [hGMM.wishart_p_nus[k] for k in ks]
    hGMM_new.update_GMM()
    for gmm, gmm_new in zip(hGMM.GMMs, hGMM_new.GMMs):
        param = gmm.write_param()
        for attr in ['mu', 'sigma']:
            try:
                param[attr] = param[attr][ks]
            except:
                param[attr] = [param[attr][k] for k in ks]
        for attr in ['p', 'alpha_vec', 'active_komp']:
            if hGMM_new.noise_class:
                param[attr] = param[attr][ks+[-1]]
            else:
                param[attr] = param[attr][ks]
        gmm_new.load_param(param)

    return hGMM_new


def merge_hGMMs(hGMM1, hGMM2):
    pass
