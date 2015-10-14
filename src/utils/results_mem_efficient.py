from __future__ import division
import numpy as np
import warnings
from mpi4py import MPI
from sklearn.covariance import MinCovDet
#from scipy.stats import multivariate_normal

from ..PurePython.distribution.wishart import invwishartrand
from ..HMlog import HMlog
from ..exceptions import NoOtherClusterError, EmptyClusterError
from . import Bhattacharyya as bhat
from . import diptest
from .lazy_property import LazyProperty
from .initialization.MatchComponents import MatchComponents
from .random_ import rmvn
from .plot_util import get_colors


class Mres(object):

    def __init__(self, d, K, p, classif_freq, p_noise=None, sim=None,
                 maxnbrsucocol=7):

        self.d = d
        self.K = K  # NB! Noise cluster not included in this
        self.J = len(self.data)
        self.p = p
        self.p_noise = p_noise
        self.maxnbrsucocol = maxnbrsucocol

        self.add_noise()
          # adding small amounts of noise to remove artifacts in the data

        if sim is None:
            sim = np.sum(classif_freq[0].tocsr().getrow(0).data)
        self.clusts = []
        for j in range(self.J):
            self.clusts.append(SampleClustering(self.data[j], classif_freq[j],
                               sim, K, self.names[j]))
        #self.clust_nm = Clustering(self.data, classif_freq, self.p, self.p_noise)

        self.mergeind = [[k] for k in range(self.K)]
        self.merged = False

    def add_noise(self):
        self.data = [dat + 0.003*np.random.randn(*dat.shape) for dat in self.data]

    @property
    def mergeind(self):
        return self._mergeind

    @mergeind.setter
    def mergeind(self, mergeind):
        '''
            Sets also canonical ordering and colors for plots.
        '''
        for s, suco in enumerate(mergeind):
            sc_ord = np.argsort(-np.array(np.sum(self.p, axis=0)[suco]))
            mergeind[s] = [suco[i] for i in sc_ord]  # Change order within each super component
        self._mergeind = mergeind
        self._suco_ord = np.argsort(-np.sum(self.p_merged, axis=0))
        self._comp_ord = [ind for sco in self._suco_ord for ind in self._mergeind[sco]]
        self._comp_colors, self._suco_colors = get_colors(
            self._mergeind, self._suco_ord, self.maxnbrsucocol)
        self._sucoid = np.array([s for k in range(self.K) for (s, comp) in
                                 enumerate(self.mergeind) if k in comp])
        for clust in self.clusts:
            clust.mergeind = self.mergeind

    @property
    def mergeind_sorted(self):
        return [self.mergeind[sco] for sco in self.suco_ord]

    @property
    def comp_ord(self):
        return self._comp_ord

    @property
    def suco_ord(self):
        return self._suco_ord

    @property
    def comp_colors(self):
        return self._comp_colors

    @property
    def suco_colors(self):
        return self._suco_colors

    @property
    def sucoid(self):
        return self._sucoid

    @property
    def p_merged(self):
        p_mer = np.zeros((self.J, len(self.mergeind)))
        for s, m_ind in enumerate(self.mergeind):
            p_mer[:, s] = sum([self.p[:, k] for k in m_ind])
        return p_mer

    def get_order(self):
        return self._comp_ord, self._suco_ord

    def merge(self, method, thr, **mmfArgs):
        self.complist = [[k] for k in range(self.K)]
        if method == 'demp':
            self.hierarchical_merge(self.get_median_overlap, thr, **mmfArgs)
            #self.hclean()
        elif method == 'bhat_hier':
            self.hierarchical_merge(self.get_median_bh_overlap_data, thr, **mmfArgs)
            #self.hclean()
        elif method == 'bhat_hier_dip':
            lowthr = mmfArgs.pop('lowthr')
            dipthr = mmfArgs.pop('dipthr')
            if 'tol' in mmfArgs:
                tol = mmfArgs.pop('tol')
            else:
                tol = 1./4
            if 'min' in mmfArgs and mmfArgs.pop('min'):
                fun_bh = self.get_min_bh_overlap_data
                fun_dip = self.get_min_bh_dt_overlap_dip2
            else:
                fun_bh = self.get_median_bh_overlap_data
                fun_dip = self.get_median_bh_dt_overlap_dip2
            print "Merging components with median/min Bhattacharyya overlap at least {}".format(thr)
            self.hierarchical_merge(fun_bh, thr, **mmfArgs)
            print """Merging components with median/min Bhattacharyya overlap
                     at least {} and robust dip test at least {}""".format(lowthr, dipthr)
            self.hierarchical_merge(fun_dip, thr=lowthr,
                                    bhatthr=lowthr, dipthr=dipthr, tol=tol, **mmfArgs)
            #self.hclean()
        elif method == 'no_merging':
            pass
        else:
            raise NameError('Unknown method for merging')
        del self.complist
        self.merged = True
        self.mergeMeth = method
        self.mergeThr = thr
        self.mergeArgs = mmfArgs

    def hierarchical_merge(self, mergeMeasureFun, thr, **mmfArgs):
        if (self.p < 0).any():
            raise ValueError('Negative p')
        mm = mergeMeasureFun(**mmfArgs)
        ind = np.unravel_index(np.argmax(mm), mm.shape)
        if mm[ind] > thr:
            print "Threshold passed at value {} for {} and {}, merging".format(
                mm[ind], self.mergeind[ind[0]], self.mergeind[ind[1]])
            self.merge_kl(ind)
            self.hierarchical_merge(mergeMeasureFun, thr, **mmfArgs)

    def merge_kl(self, ind):
        mergeind = self.mergeind
        mergeind[ind[1]] += mergeind[ind[0]]
        mergeind.pop(ind[0])
        self.mergeind = mergeind
        #self.complist[ind[1]] = [self.complist[ind[1]], self.complist[ind[0]]]
        #self.complist[ind[0]] = []

    def greedy_merge(self, mergeMeasureFun, thr, **mmfKw):
        mm = mergeMeasureFun(**mmfKw)
        suco_assign = -np.ones(mm.shape[0], dtype='int')
        sucos = []
        while np.amax(mm) > thr:
            nextmerge = np.unravel_index(np.argmax(mm), mm.shape)
            nextsuco = []
            if suco_assign[nextmerge[0]] == -1 or suco_assign[nextmerge[0]] != suco_assign[nextmerge[1]]:
                for nm in nextmerge:
                    if suco_assign[nm] == -1:
                        nextsuco += [nm]
                    else:
                        nextsuco += sucos[suco_assign[nm]]
                #print "nextmerge = {}".format(nextmerge)
                suco_assign[np.array(nextsuco)] = len(sucos)
                sucos.append(nextsuco)
            mm[nextmerge[0], nextmerge[1]] = -np.inf
            mm[nextmerge[1], nextmerge[0]] = -np.inf
        # Empty mergeind list
        while len(self.mergeind) > 0:
            self.mergeind.pop()
        # Add new content
        self.mergeind += [sucos[s] for s in np.unique(suco_assign[suco_assign > -1])]
        self.mergeind += [[k] for k in np.nonzero(suco_assign == -1)[0]]

    def get_median_overlap(self, fixvalind=None, fixval=-1):
        if fixvalind is None:
            fixvalind = []
        overlap = [clust.get_overlap() for clust in self.clusts]
        return self.get_medprop_pers(overlap, fixvalind, fixval)

    def get_median_bh_overlap_data(self, fixvalind=None, fixval=-1, min_weight=0):
        if fixvalind is None:
            fixvalind = []
        bhd = [clust.get_bh_overlap_data(min_weight=min_weight) for clust in self.clusts]
        #print "median bhattacharyya distance overlap = {}".format(get_medprop_pers(bhd, fixvalind, fixval))
        return self.get_medprop_pers(bhd, fixvalind, fixval)

    def get_min_bh_overlap_data(self, fixvalind=None, fixval=-1, min_weight=0):
        if fixvalind is None:
            fixvalind = []
        bhd = [clust.get_bh_overlap_data(min_weight=min_weight) for clust in self.clusts]
        #print "median bhattacharyya distance overlap = {}".format(get_medprop_pers(bhd, fixvalind, fixval))
        return self.get_minprop_pers(bhd, fixvalind, fixval)

    def get_median_bh_dt_overlap_dip(self, bhatthr, dipthr, fixvalind=None, fixval=-1):
        if fixvalind is None:
            fixvalind = []
        mbhd = self.get_median_bh_overlap_data(fixvalind, fixval)
        while (mbhd > bhatthr).any():
            ind = np.unravel_index(np.argmax(mbhd), mbhd.shape)
            print "Dip test for {} and {}".format(self.mergeind[ind[0]], self.mergeind[ind[1]])
            if self.okdiptest(ind, dipthr):
                return mbhd
            fixvalind.append(ind)
            mbhd = self.get_median_bh_overlap_data(fixvalind, fixval)
        return mbhd

    def get_median_bh_dt_overlap_dip2(self, bhatthr, dipthr, tol=1./4,
                                      fixvalind=None, fixval=-1, min_weight=0):
        if fixvalind is None:
            fixvalind = []
        mbhd = self.get_median_bh_overlap_data(fixvalind, fixval, min_weight)
        while (mbhd > bhatthr).any():
            ind = np.unravel_index(np.argmax(mbhd), mbhd.shape)
            print "Dip test for {} and {}".format(self.mergeind[ind[0]], self.mergeind[ind[1]])
            if self.okdiptest2(ind, dipthr, tol):
                return mbhd
            fixvalind.append(ind)
            mbhd = self.get_median_bh_overlap_data(fixvalind, fixval, min_weight)
        return mbhd

    def get_min_bh_dt_overlap_dip2(self, bhatthr, dipthr, tol=1./4,
                                   fixvalind=None, fixval=-1, min_weight=0):
        if fixvalind is None:
            fixvalind = []
        mbhd = self.get_min_bh_overlap_data(fixvalind, fixval, min_weight)
        while (mbhd > bhatthr).any():
            ind = np.unravel_index(np.argmax(mbhd), mbhd.shape)
            print "Dip test for {} and {}".format(self.mergeind[ind[0]], self.mergeind[ind[1]])
            if self.okdiptest2(ind, dipthr, tol):
                return mbhd
            fixvalind.append(ind)
            mbhd = self.get_min_bh_overlap_data(fixvalind, fixval, min_weight)
        return mbhd

    def okdiptest(self, ind, thr):
        k, l = ind
        for dim in [None]+range(self.d):
            nbr_computable = self.J
            below = 0
            for j, clust in enumerate(self.clusts):
                try:
                    if clust.get_pdip_discr_jkl(k, l, dim) < thr:
                        below += 1
                except EmptyClusterError:
                    nbr_computable -= 1

                if below > np.floor(nbr_computable/4):
                    print "For {} and {}, diptest failed in dim {}: {} below out of {}".format(
                        self.mergeind[k], self.mergeind[l], dim, below, nbr_computable)
                    return False
            print "Diptest ok for {} and {} in dim {}: {} below out of {}".format(
                self.mergeind[k], self.mergeind[l], dim, below, nbr_computable)
        return True

    def okdiptest2(self, ind, thr, tol=1./4, min_dip=0.01):
        '''
            tol is the proportion of samples that is allowed to
            be below dip test threshold thr.
        '''
        k, l = ind
        dims = [None]+self.get_dims_with_low_bhat_overlap(ind, tol=tol)
        for dim in dims:
            nbr_computable = self.J
            below = 0
            for j, clust in enumerate(self.clusts):
                try:
                    dip, pdip = clust.get_dip_discr_jkl(k, l, dim)
                    if clust.get_pdip_discr_jkl(k, l, dim) < thr:
                        below += 1
                except EmptyClusterError:
                    nbr_computable -= 1

                if below > np.floor(nbr_computable*tol):
                    print "For {} and {}, diptest failed in dim {}: {} below out of {}".format(
                        self.mergeind[k], self.mergeind[l], dim, below, nbr_computable)
                    return False
            print "Diptest ok for {} and {} in dim {}: {} below out of {}".format(
                self.mergeind[k], self.mergeind[l], dim, below, nbr_computable)
        return True

    def get_dims_with_low_bhat_overlap(self, ind, tol=1./4):
        k, l = ind
        dims = []
        for dd in range(self.d):
            nbr_computable = len(self.clusts)
            below = 0
            for clust in self.clusts:
                bhat_overlap = clust.get_bh_overlap_data(dd, [k, l])[0, 1]
                if np.isnan(bhat_overlap):
                    nbr_computable -= 1
                else:
                    W = clust.get_W(k)+clust.get_W(l)
                    if bhat_overlap < self.bhat_overlap_1d_threshold(W):
                        below += 1
            if below > np.floor(nbr_computable*tol):
                dims.append(dd)
        return dims

    @staticmethod
    def bhat_overlap_1d_threshold(weight):
        '''
            Based on Table 1 in Hennig (2010): Methods for merging Gaussian
            mixture components, Adv Data Anal Classif 4:3-34.
        '''
        if weight <= 50:
            return 0.201
        if weight <= 200:
            return 0.390
        return 0.490

    def get_pdip(self, suco=True):
        '''
            Diptest p-values for each cluster, each dimension and each sample.
        '''
        if not suco:
            return self.get_pdip_comp()
        try:
            return self.pdiplist
        except:
            pass

        S = len(self.mergeind)
        pdiplist = []
        for k in range(S):
            pdip = np.zeros((self.J, self.d))
            for j, clust in enumerate(self.clusts):
                pdip[j, :] = clust.get_pdip(k)
            print "\r Diptest computed for component {}".format(k),
            pdiplist.append(np.copy(pdip))
        print ''
        self.pdiplist = pdiplist
        return self.pdiplist

    def get_pdip_comp(self):
        '''
            Diptest p-values for each component, each dimension and each sample.
        '''
        try:
            return self.pdiplist_comp
        except:
            pass

        pdiplist = []
        for k in range(self.K):
            pdip = np.zeros((self.J, self.d))
            for j, clust in enumerate(self.clusts):
                pdip[j, :] = clust.clusters[k].get_pdip()
            print "\r Diptest computed for component {}".format(k),
            pdiplist.append(np.copy(pdip))
        print ''
        self.pdiplist_comp = pdiplist
        return self.pdiplist_comp

    def get_pdip_summary(self, suco=True):
        '''
            Medians, 25th percentiles and minima of diptest p-values for each cluster/component and each data dimension
        '''
        pdiplist = self.get_pdip(suco)

        d = self.d
        K = len(pdiplist)
        pdipsummary = {'Median': np.empty((K, d)), '25th percentile': np.empty((K, d)), 'Minimum': np.empty((K, d))}
        for k in range(K):
            pdk = pdiplist[k][~np.isnan(pdiplist[k][:, 0]), :]
            if len(pdk) == 0:
                pdipsummary['Median'][k, :] = np.nan
                pdipsummary['25th percentile'][k, :] = np.nan
                pdipsummary['Minimum'][k, :] = np.nan
            else:
                pdipsummary['Median'][k, :] = np.median(pdk, 0)
                pdipsummary['25th percentile'][k, :] = np.percentile(pdk, 25, 0)
                pdipsummary['Minimum'][k, :] = np.min(pdk, 0)
        return pdipsummary

    def get_data_kdj(self, min_clf, k, dd, j=None, suco=True):
        '''
            Get data points belonging to a certain cluster

            min_clf    -    min classification frequency for the point into the given cluster
            k        -    cluster number
            dd        -    dimonsion for which data should be returned
            j        -     sample nbr
        '''
        if not j is None:
            return self.clusts[j].get_data_kd(min_clf, k, dd, suco)
        else:
            return np.vstack([clust.get_data_kd(min_clf, k, dd, suco) for clust in self.clusts])

    def get_quantiles(self, alpha, j=None, ks=None, dds=None, suco=True):
        '''
            Returns alpha quantile(s) in each dimension of sample j
            (the pooled data if j = None) for each of the clusters.
        '''
        if not j is None:
            return self.clusts[j].get_quantiles(alpha, ks, dds, suco=suco)

        if dds is None:
            dds = range(self.d)

        if ks is None:
            if suco:
                ks = range(len(self.mergeind))
            else:
                ks = range(self.K)

        quantiles = np.zeros((len(ks), len(alpha), len(dds)))
        for ik, k in enumerate(ks):
            if suco:
                clfs = [clust.get_classif_freq(k) for clust in self.clusts]
                data = np.vstack([clust.data[clf.indices, :] for (clust, clf) in zip(self.clusts, clfs)])
                weights = np.hstack([clf.data for clf in clfs])
            else:
                clusters = [clust.clusters[k] for clust in self.clusts]
                data = np.vstack([cluster.data[cluster.indices, :] for cluster in clusters])
                weights = np.hstack([cluster.weights for cluster in clusters])
            weights = weights*1./np.sum(weights)
            for idd, dd in enumerate(dds):
                quantiles[ik, :, idd] = SampleClustering.quantile(
                    data[:, dd], weights, alpha)
        return quantiles

    @staticmethod
    def get_medprop_pers(prop, fixvalind=None, fixval=-1):
        if fixvalind is None:
            fixvalind = []
        med_prop = np.empty(prop[0].shape)
        for k in range(med_prop.shape[0]):
            for l in range(med_prop.shape[1]):
                prop_kl = np.array([pr[k, l] for pr in prop])
                med_prop[k, l] = np.median(prop_kl[~np.isnan(prop_kl)])
                if np.isnan(med_prop[k, l]):
                    med_prop[k, l] = fixval

        for ind in fixvalind:
            if len(ind) == 1:
                med_prop[ind, :] = fixval
                med_prop[:, ind] = fixval
            else:
                med_prop[ind[0], ind[1]] = fixval
                med_prop[ind[1], ind[0]] = fixval
        return med_prop

    @staticmethod
    def get_minprop_pers(prop, fixvalind=None, fixval=-1):
        if fixvalind is None:
            fixvalind = []
        med_prop = np.empty(prop[0].shape)
        for k in range(med_prop.shape[0]):
            for l in range(med_prop.shape[1]):
                prop_kl = np.array([pr[k, l] for pr in prop])
                non_nan_ind = ~np.isnan(prop_kl)
                if np.sum(non_nan_ind) == 0:
                    med_prop[k, l] = fixval
                else:
                    med_prop[k, l] = np.min(prop_kl[non_nan_ind])
                if np.isnan(med_prop[k, l]):
                    med_prop[k, l] = fixval

        for ind in fixvalind:
            if len(ind) == 1:
                med_prop[ind, :] = fixval
                med_prop[:, ind] = fixval
            else:
                med_prop[ind[0], ind[1]] = fixval
                med_prop[ind[1], ind[0]] = fixval
        return med_prop


class MetaData(object):

    def __init__(self, meta_data):
        self.samp = meta_data['samp'].copy()
        self.marker_lab = meta_data['marker_lab'][:]

    def sort(self, names):
        self.order = []
        for name in names:
            self.order.append(self.samp['names'].index(name))
        for key in self.samp:
            self.samp[key] = [self.samp[key][j] for j in self.order]


class Traces(object):
    '''
        Object containing information for traceplots.
    '''

    def __init__(self, bmlog_burn, bmlog_prod):
        self.saveburn = bmlog_burn.theta_sim.shape[0]
        self.saveprod = bmlog_prod.theta_sim.shape[0]
        self.savefrqburn = bmlog_burn.savefrq
        self.savefrqprod = bmlog_prod.savefrq

        self.burnind = np.arange(1, self.saveburn+1)*self.savefrqburn
        self.prodind = self.burnind[-1] + np.arange(1, self.saveprod+1)*self.savefrqprod
        self.ind = np.hstack([self.burnind, self.prodind])

        self.mulat_burn = bmlog_burn.theta_sim
        self.mulat_prod = bmlog_prod.theta_sim
        self.nu_burn = bmlog_burn.nu_sim
        self.nu_prod = bmlog_prod.nu_sim
        try:
            self.nu_sigma_burn = bmlog_burn.nu_sigma_sim
            self.nu_sigma_prod = bmlog_prod.nu_sigma_sim
            print "self.nu_sigma_burn.shape = {}".format(self.nu_sigma_burn.shape)
            print "self.nu_sigma_prod.shape = {}".format(self.nu_sigma_prod.shape)
        except AttributeError:
            print "No nu_sigma in log."
            pass

        self.K = self.mulat_burn.shape[1]

    def get_mulat_k(self, k):
        return np.vstack([self.mulat_burn[:, k, :], self.mulat_prod[:, k, :]])

    def get_nu(self):
        return np.vstack([self.nu_burn, self.nu_prod])

    def get_nu_sigma(self):
        return np.vstack([self.nu_sigma_burn, self.nu_sigma_prod])


class FCsample(object):
    '''
        Object containing results for synthetic data sample or corresponding real sample.
    '''

    def __init__(self, data, name):
        self.data = data
        self.name = name

    def get_data(self, N=None):
        if N is None:
            return self.data
        if N > self.data.shape[0]:
            warnings.warn('Requested sample is larger than data')
            N = self.data.shape[0]
        ind = np.random.choice(range(self.data.shape[0]), N, replace=False)
        return self.data[ind, :]


class SynSample(FCsample):
    '''
        Object containing results for synthetic data sample.
    '''

    def __init__(self, syndata, genname, fcmimic):
        super(SynSample, self).__init__(syndata, fcmimic.name)
        self.genname = genname
        self.realsize = fcmimic.data.shape[0]

    def get_data(self, N=None):
        if N is None:
            N = self.realsize
        return super(SynSample, self).get_data(N)


class MimicSample(object):
    '''
        Object containing results for synthetic data sample and corresponding real sample.
    '''

    def __init__(self, data, name, syndata, modelname):
        self.realsamp = FCsample(data, name)
        self.synsamp = SynSample(syndata, modelname, self.realsamp)


class Cluster(object):

    def __init__(self, classif_freq, data, sim):
        self.sim = sim
        self.data = data
        self.classif_freq = classif_freq/sim
        self.indices = self.classif_freq.indices
        self.weights = self.classif_freq.data

    @LazyProperty
    def W(self):
        #print "computing W"
        return sum(self.weights)

    @LazyProperty
    def wX(self):
        #print "computing wX"
        return np.dot(self.weights, self.data[self.indices, :]).reshape(1, -1)

    @LazyProperty
    def wXXT(self):
        d = self.data.shape[1]
        dat = self.data[self.indices, :]
        wXXT = (dat*self.weights.reshape(-1, 1)).T.dot(dat)
        if wXXT.shape != (d, d):
            raise ValueError
        return wXXT
        #wXXT = np.zeros((self.data.shape[1], self.data.shape[1]))
#        for i, ind in enumerate(self.indices):
#            x = self.data[ind, :].reshape(1, -1)
#            wXXT += self.weights[i]*x.T.dot(x)
#        return wXXT

    def get_pdip(self, dims=None):
        if dims is None:
            dims = range(self.data.shape[1])
        pdip = np.zeros(len(dims))
        if len(self.indices) == 0:
            return np.nan*pdip
        for i, dd in enumerate(dims):
            xcum, ycum = diptest.cum_distr(self.data[self.indices, dd], self.weights/self.W)
            dip = diptest.dip_from_cdf(xcum, ycum)
            pdip[i] = diptest.dip_pval_tabinterpol(dip, self.W)
        return pdip


class SampleClustering(object):
    '''
        Object containing information about clustering of the data.
    '''

    def __init__(self, data, classif_freq, sim, K, name):
        self.data = data
        self.sim = sim
        self.K = K
        self.name = name

        classif_freq = classif_freq.tocsc()
            # Transform sparse matrix to enable column slicing
        self.clusters = []
        for k in range(self.K):
            self.clusters.append(Cluster(classif_freq.getcol(k), data, sim))
        self.d = data.shape[1]
        #self.vacuous_komp = np.vstack([np.sum(clf, axis=0) < 3 for clf in self.classif_freq])

    @property
    def x_sample(self):
        N = self.data.shape[0]
        x = self.K*np.ones(N)
        cum_freq = np.zeros(N)
        alpha = np.random.random(N)
        notfound = np.arange(N)
        for k, cluster in enumerate(self.clusters):
            cum_freq += cluster.classif_freq.toarray().reshape(-1)
            newfound_bool = alpha[notfound] < cum_freq[notfound]
            newfound = notfound[newfound_bool]
            x[newfound] = k
            notfound = notfound[~newfound_bool]
        return x

    def get_mean(self, s):
        ks = self.mergeind[s]
        return sum([self.clusters[k].wX for k in ks])/sum([self.clusters[k].W for k in ks])

    def get_scatter(self, s):
        ks = self.mergeind[s]
        try:
            wXXT = sum([self.clusters[k].wXXT for k in ks])/sum([self.clusters[k].W for k in ks])
        except ZeroDivisionError:
            return np.nan*self.clusters[0].wXXT
        mu = self.get_mean(s).reshape(1, -1)
        return wXXT - mu.T.dot(mu)

    def get_W(self, s):
        ks = self.mergeind[s]
        return sum([self.clusters[k].W for k in ks])

    def get_classif_freq(self, s):
        ks = self.mergeind[s]
        clf = sum([self.clusters[k].classif_freq for k in ks])
        return clf

    def get_pdip(self, k, dims=None):
        if dims is None:
            dims = range(self.data.shape[1])
        clf = self.get_classif_freq(k)
        W = self.get_W(k)
        pdip = np.zeros(len(dims))
        if len(clf.indices) == 0:
            return np.nan*pdip
        for i, dd in enumerate(dims):
            xcum, ycum = diptest.cum_distr(self.data[clf.indices, dd], clf.data/W)
            dip = diptest.dip_from_cdf(xcum, ycum)
            pdip[i] = diptest.dip_pval_tabinterpol(dip, W)
        return pdip

    def get_overlap(self):
        '''
            Estimated misclassification probability between two super
            clusters, i.e. probability that Y_i is classified as
            belonging to l when it truly belongs to k.
        '''
        S = len(self.mergeind)
        overlap = np.zeros((S, S))
        for k in range(S):
            for l in range(S):
                overlap[k, l] = self.get_classif_freq(k).T.dot(
                    self.get_classif_freq(l)).todense()
            overlap[k, :] /= self.get_W(k)
            overlap[k, k] = 0
        return overlap

    def get_bh_overlap_data(self, dd=None, ks=None, min_weight=0):
        '''
            Clusters with less than min_weight will get weight nan.
        '''
        if ks is None:
            S = len(self.mergeind)
            ks = range(S)
        else:
            S = len(ks)
        bhd = np.nan*np.ones((S, S))
        mus = [self.get_mean(k) for k in ks]
        Sigmas = [self.get_scatter(k) for k in ks]
        weights = [self.get_W(k) for k in ks]

        for k in range(S):
            bhd[k, k] = 0
            if weights[k] < min_weight:
                continue
            for l in range(S):
                if l != k:
                    if weights[l] < min_weight:
                        continue
                    if dd is None:
                        bhd[k, l] = bhat.bhattacharyya_overlap(
                            mus[k], Sigmas[k], mus[l], Sigmas[l])
                    else:
                        bhd[k, l] = bhat.bhattacharyya_overlap(
                            mus[k][0, dd], Sigmas[k][dd, dd].reshape(1, 1),
                            mus[l][0, dd], Sigmas[l][dd, dd].reshape(1, 1))
            #print "nbr nan in bhd[j]: {}".format(np.sum(np.isnan(bhd[j])))
            #print "nbr not nan in bhd[j]: {}".format(np.sum(~np.isnan(bhd[j])))
        return bhd

    def get_dip_discr_jkl(self, k, l, dim=None):
        '''
            p-value of diptest of unimodality for the merger of super
            cluster k and l

            Input:
                dim     - dimension along which the test should be
                          performed. If dim is None, the test will be
                          performed on the projection onto Fisher's
                          discriminant coordinate.
        '''
        clf_k = self.get_classif_freq(k)
        clf_l = self.get_classif_freq(l)
        clf = clf_k + clf_l
        W_k = self.get_W(k)
        W_l = self.get_W(l)
        W = W_k + W_l

        if dim is None:
            if W_k == 0 or W_l == 0:
                raise EmptyClusterError
            dataproj = self.discriminant_projection(k, l)
        else:
            if W == 0:
                raise EmptyClusterError
            dataproj = self.data[:, dim]

        xcum, ycum = diptest.cum_distr(dataproj[clf.indices], clf.data/W)
        dip = diptest.dip_from_cdf(xcum, ycum)
        return dip, diptest.dip_pval_tabinterpol(dip, W)

    def get_pdip_discr_jkl(self, k, l, dim=None):
        '''
            p-value of diptest of unimodality for the merger of super
            cluster k and l

            Input:
                dim     - dimension along which the test should be
                          performed. If dim is None, the test will be
                          performed on the projection onto Fisher's
                          discriminant coordinate.
        '''
        dip, pdip = self.get_dip_discr_jkl(k, l, dim)
        return pdip

    def discriminant_projection(self, s1, s2):
        '''
            Projection of a data set onto Fisher's discriminant
            coordinate between two super clusters.
        '''
        mu1, Sigma1 = self.get_mean(s1).T, self.get_scatter(s1)
        mu2, Sigma2 = self.get_mean(s2).T, self.get_scatter(s2)
        dc = self.discriminant_coordinate(mu1, Sigma1, mu2, Sigma2)
        proj = np.dot(self.data, dc)
        return proj

    def get_data_kd(self, min_clf, k, dd, suco=True):
        '''
            Get data points belonging to a certain cluster

            min_clf    -    min classification frequency for the point into the given cluster
            k          -    cluster number
            dd         -    dimonsion for which data should be returned
        '''
        if suco:
            clf = self.get_classif_freq(k)
        else:
            clf = self.clusters[k].classif_freq
        return self.data[clf.indices[clf.data > min_clf], dd]

    def get_quantiles(self, alpha, ks=None, dds=None, suco=True):
        '''
            Returns alpha quantile(s) in each dimension of sample j
            (the pooled data if j = None) for each of the clusters.
        '''

        if ks is None:
            ks = range(self.K)
        if dds is None:
            dds = range(self.d)

        quantiles = np.zeros((len(ks), len(alpha), len(dds)))
        for ik, k in enumerate(ks):
            if suco:
                clf = self.get_classif_freq(k)
            else:
                clf = self.clusters[k].classif_freq
            data = self.data[clf.indices, :]
            weights = clf.data*1./np.sum(clf.data)
            for idd, dd in enumerate(dds):
                quantiles[ik, :, idd] = self.quantile(
                    data[:, dd], weights, alpha)
        return quantiles

    @staticmethod
    def discriminant_coordinate(mu1, Sigma1, mu2, Sigma2):
        '''
            Fisher's discriminant coordinate
        '''
        w = np.linalg.solve(Sigma1+Sigma2, mu2-mu1)
        w /= np.linalg.norm(w)
        return w

    @staticmethod
    def quantile(y, w, alpha):
        '''
            Returns alpha quantile(s) (alpha can be a vector) of empirical
            probability distribution of data y weighted by w
        '''
        alpha = np.array(alpha)
        alpha_ord = np.argsort(alpha)
        alpha_sort = alpha[alpha_ord]
        y = y[w > 0]
        w = w[w > 0]
        y_ord = np.argsort(y)
        y_sort = y[y_ord]
        cumdistr = np.cumsum(w[y_ord])
        q = np.empty(alpha.shape)
        if len(cumdistr) == 0 or cumdistr[-1] == 0:
            q[:] = float('nan')
            return q
        cumdistr /= cumdistr[-1]
        j = 0
        i = 0
        while j < len(alpha) and i < len(cumdistr):
            if alpha_sort[j] < cumdistr[i]:
                q[j] = y_sort[i]
                j += 1
            else:
                i += 1
        if j < len(alpha):
            q[j:] = y_sort[-1]
        return q[np.argsort(alpha_ord)]


class DataSetClustering(object):

    def __init__(self, sample_clusts):
        self.sample_clusts = sample_clusts
        self.J = len(self.sample_clusts)
        self.K = self.sample_clusts[0].K
        self.d = self.sample_clusts[0].d

    def get_pdip_summary(self):
        pdipsummary = {'Median': np.empty((self.K, self.d)),
                       '25th percentile': np.empty((self.K, self.d)),
                       'Minimum': np.empty((self.K, self.d))}
        for k in range(self.K):
            pdk = np.vstack([samp_cl.get_pip(k) for samp_cl in self.sample_clusts])
            pdk = pdk[~np.isnan(pdk[:, 0]), :]
            if len(pdk) == 0:
                pdipsummary['Median'][k, :] = np.nan
                pdipsummary['25th percentile'][k, :] = np.nan
                pdipsummary['Minimum'][k, :] = np.nan
            else:
                pdipsummary['Median'][k, :] = np.median(pdk, 0)
                pdipsummary['25th percentile'][k, :] = np.percentile(pdk, 25, 0)
                pdipsummary['Minimum'][k, :] = np.min(pdk, 0)
        return pdipsummary


class Components(object):
    '''
        Object containing information about mixture components
    '''

    def __init__(self, bmlog, p):
        self.J = bmlog.J
        self.K = bmlog.K
        self.d = bmlog.d
        self.mupers = bmlog.mupers_sim_mean
        self.mulat = bmlog.theta_sim_mean
        try:
            self.Sigma_mu = bmlog.Sigma_mu_sim_mean
        except AttributeError:
            print "Sigma mu not in bmlog, estimate from mupers and mulat used."
            self.Sigma_mu = self.estimate_Sigma_mu()
        self.Sigmapers = bmlog.Sigmapers_sim_mean
        self.Sigmalat = bmlog.Sigmaexp_sim_mean
        self.nu = np.mean(bmlog.nu_sim, axis=0)
        self.p = p
        self.active_komp = bmlog.active_komp[:, :self.K]  # excluding noise component
        self.names = bmlog.names

    @classmethod
    def load(cls, savedir, comm=MPI.COMM_WORLD):
        hmlog = HMlog.load(savedir, comm)
        p = hmlog.prob_sim_mean[:, :hmlog.K]
        return cls(hmlog, p)

    @classmethod
    def load_matched(cls, savedir, lamb, verbose=False, dist='bhat', comm=MPI.COMM_WORLD):
        if comm.Get_rank() == 0:
            start_components = cls.load(savedir, MPI.COMM_SELF)
            match_comp = MatchComponents(start_components, lamb, verbose, dist=dist)
            match_comp.names = start_components.names
        else:
            match_comp = None

        match_comp = comm.bcast(match_comp)
        return match_comp

    def estimate_Sigma_mu(self):
        Sigma_mu = np.empty((self.K, self.d, self.d))
        mu_centered = self.mupers - self.mulat[np.newaxis, :, :]
        for k in range(self.K):
            mu_centered_k = mu_centered[:, k, :]
            mu_centered_k = mu_centered_k[~np.isnan(mu_centered_k[:, 0]), :]
            if mu_centered_k.shape[0] <= 1:
                Sigma_mu[k, :, :] = np.nan
            else:
                Sigma_mu[k, :, :] = mu_centered_k.T.dot(mu_centered_k)/(mu_centered_k.shape[0]-1)
        return Sigma_mu

    def new_thetas_from_GMM_fit(self, Ks=None, n_init=10, n_iter=100,
                                covariance_type='full'):
        if Ks is None:
            Ks = range(int(self.K/4), self.K*2)
        allmus = np.vstack([self.mupers[j, :, :] for j in range(self.mupers.shape[0])])
        thetas = GMM_means_for_best_BIC(allmus, Ks, n_init, n_iter, covariance_type)
        self.new_thetas = thetas
        return thetas

    def classif(self, Y, j, p_noise=None, mu_noise=0.5*np.ones((1, 4)),
                Sigma_noise=0.5**2*np.eye(4)):
        '''
            Classify data points according to highest
            likelihood by mixture model.

            Input:

                Y (Nxd)     -   data
                j           -   sample number
                p_noise     -   noise component probability
                mu_noise    -   noise component mean
                Sigma_noise -   noise component covariance matrix
        '''
        if not p_noise is None:
            mus = np.vstack([self.mupers[j, :, :], mu_noise.reshape(1, self.d)])
            Sigmas = np.vstack([self.Sigmapers[j, :, :, :], Sigma_noise.reshape(1, self.d, self.d)])
            ps = np.vstack([self.p[j, :].reshape(-1, 1), np.array(p_noise[j]).reshape(1, 1)])
            ps /= np.sum(ps)
        else:
            mus = self.mupers[j, :, :]
            Sigmas = self.Sigmapers[j, :, :, :]
            ps = self.p[j, :]

        dens = np.empty((Y.shape[0], mus.shape[0]))
        for k in range(mus.shape[0]):
            dens[:, k] = lognorm_pdf(Y, mus[k, :], Sigmas[k, :, :], ps[k])
        dens[np.isnan(dens)] = -np.inf
        return np.argmax(dens, axis=1)

    def get_bh_distance_to_latent(self, j):
        '''
            Get Bhattacharyya distance for components in sample j.

            Returns a (self.K x self.K) matrix where element (k, l)
            is the distance from sample component k to latent
            component l.
        '''
        bhd = np.empty((self.K, self.K))
        for k in range(self.K):
            if self.active_komp[j, k] < 0.05:
                bhd[k, :] = np.nan
                continue
            for l in range(self.K):
                bhd[k, l] = bhat.bhattacharyya_distance(
                    self.mupers[j, k, :], self.Sigmapers[j, k, :, :],
                    self.mulat[l, :], self.Sigmalat[l, :])
        return bhd

    def get_bh_distance_to_own_latent(self):
        '''
            Get Bhattacharyya distance from sample component to
            corresponding (own) latent component.

            Returns a (self.J x self.K) array where element (j, k)
            is the distance from sample component k to latent
            component k in sample j.
        '''
        bhd = np.empty((self.J, self.K))
        for j in range(self.J):
            for k in range(self.K):
                if self.active_komp[j, k] < 0.05:
                    bhd[j, k] = np.nan
                else:
                    bhd[j, k] = bhat.bhattacharyya_distance(
                        self.mupers[j, k, :], self.Sigmapers[j, k, :, :],
                        self.mulat[k, :], self.Sigmalat[k, :])
        return bhd

    def get_bh_overlap(self, j):
        '''
            Get bhattacharyya overlap between all sample components in
            sample j.

            Returns a symmeetric (self.K x self.K) matrix.
        '''
        bhd = np.empty((self.K, self.K))
        for k in range(self.K):
            if self.active_komp[j, k] > 0.05:
                muk = self.mupers[j, k, :]
                Sigmak = self.Sigmapers[j, k, :, :]
            else:
                muk = self.mulat[k, :]
                Sigmak = self.Sigmalat[k, :, :]
            for l in range(self.K):
                if self.active_komp[j, l] > 0.05:
                    mul = self.mupers[j, l, :]
                    Sigmal = self.Sigmapers[j, l, :, :]
                else:
                    mul = self.mulat[l, :]
                    Sigmal = self.Sigmalat[l, :, :]
                bhd[k, l] = bhat.bhattacharyya_overlap(muk, Sigmak, mul, Sigmal)
            bhd[k, k] = 0
        return bhd

    def get_median_bh_overlap(self, fixvalind=None, fixval=-1):
        if fixvalind is None:
            fixvalind = []
        bhd = [self.get_bh_overlap(j) for j in range(self.J)]
        return self.get_medprop_pers(bhd, fixvalind, fixval)

    def get_center_dist(self):
        '''
            Get distance from component mean to latent 
            component mean for each component in each sample.
        '''
        dist = np.zeros((self.J, self.K))
        for k in range(self.K):
            for j in range(self.J):
                if self.active_komp[j, k] <= 0.05:
                    dist[j, k] = np.nan
                else:
                    dist[j, k] = np.linalg.norm(self.mupers[j, k, :] 
                                                - self.mulat[k, :])
        return dist

    def get_cov_dist(self, norm='F'):
        '''
            Get distance from component covariance matrix to 
            latent covariance matrix for each component in 
            each sample.

            norm    -   'F' gives Frobenius distance, 
                         2 gives operator 2-norm.
        '''
        covdist = np.zeros((self.J, self.K))
        for j in range(self.J):
            for k in range(self.K):
                if self.active_komp[j, k] <= 0.05:
                    covdist[j, k] = np.nan
                else:
                    if norm == 'F':
                        covdist[j, k] = np.linalg.norm(self.Sigmapers[j, k, :, :]
                                                       -self.Sigmalat[k, :, :])
                    elif norm == 2:
                        covdist[j, k] = np.linalg.norm(self.Sigmapers[j, k, :, :]
                                                       -self.Sigmalat[k, :, :], ord=2)
        return covdist

    def get_center_distance_quotient(self):
        '''
            Get for each component in each sample, the quotient 
            between the distance from the component mean to the 
            correct latent component mean and the distance from 
            the component mean to the closest latent component 
            mean which has not been merged into the component.
        '''
        distquo = np.zeros((self.J, self.K))
        for suco in self.mergeind:
            for k in suco:
                otherind = np.array([not (kk in suco) for kk in range(self.K)])
                if sum(otherind) == 0:
                    raise NoOtherClusterError
                corrdist = self.get_center_dist()
                for j in range(self.J):                
                    wrongdist = min(np.linalg.norm(self.mupers[j, [k]*sum(otherind), :] 
                        - self.mulat[otherind, :], axis=1))
                    distquo[j, k] = wrongdist/corrdist[j, k]
        return distquo

    def get_latent_bhattacharyya_overlap_quotient(self):
        distquo = np.zeros((self.J, self.K))
        for suco in self.mergeind:
            for k in suco:
                otherind = [kk for kk in range(self.K) if not kk in suco]# np.array([not (kk in suco) for kk in range(self.K)])
                if len(otherind) == 0:
                    raise NoOtherClusterError
                for j in range(self.J):
                    corrdist = bhat.bhattacharyya_overlap(self.mupers[j, k, :],
                        self.Sigmapers[j, k, :, :], self.mulat[k, :],
                        self.Sigmalat[k, :, :])
                    wrongdist = max(
                        [bhat.bhattacharyya_overlap(self.mupers[j, k, :],
                            self.Sigmapers[j, k, :, :], self.mulat[kk, :],
                            self.Sigmalat[kk, :, :]) for kk in otherind])
                    distquo[j, k] = corrdist/wrongdist
        return distquo

    def mu_dist_percentiles(self, q=99.99, Ntest=100000, robust_Sigma_mu=False):
        try:
            percentile_dict = self._mu_dist_percentiles
        except AttributeError:
            self._mu_dist_percentiles = {}
            percentile_dict = self._mu_dist_percentiles
        try:
            return percentile_dict[(q,Ntest)]
        except KeyError:
            pass

        if not robust_Sigma_mu:
            Sigma_mu = self.Sigma_mu
        else:
            mcd = MinCovDet().fit(self.mupers)
            Sigma_mu = mcd.covariance_
            print "Number of MCD outliers: {}".format(np.sum(~mcd.support_))
        percentiles = np.empty(self.K)
        frobenius_dists = np.empty(Ntest)
        for k in range(self.K):
            if np.isnan(self.Sigma_mu[k, 0, 0]):
                percentiles[k] = np.nan
            else:
                for n in range(Ntest):
                    mu = rmvn(
                        self.mulat[k, :], Sigma_mu[k, :, :])
                    frobenius_dists[n] = np.linalg.norm(mu - self.mulat[k, :])
                percentiles[k] = np.percentile(frobenius_dists, q)
                print "\r Percentiles for {} out of {} computed".format(k+1, self.K),
        print ''
        percentile_dict[(q,Ntest)] = percentiles
        return percentiles
        
    def mu_outliers(self, q=99.99, Ntest=100000):
        '''
            Find which components in which sample that have
            locations (mu_jk values) that are outliers at a given
            percentile, i.e. that have a Euclidean distance to the
            latent mean (theta_k) that is larger than the given
            percentile in a sample of distances to the latent mean
            with locations drawn from the prior model of mu_jk.

            q       - percentile
            Ntest   - number of random samples

            Returns (J X K) matrix.
        '''
        centerd = self.get_center_dist()
        outliers = (centerd -
                    self.mu_dist_percentiles(q, Ntest)[np.newaxis, :]) > 0
        return outliers

    def Sigma_dist_percentiles(self, q=99.99, Ntest=100000):
        try:
            percentile_dict = self._Sigma_dist_percentiles
        except AttributeError:
            self._Sigma_dist_percentiles = {}
            percentile_dict = self._Sigma_dist_percentiles
        try:
            return percentile_dict[(q, Ntest)]
        except KeyError:
            pass
        percentiles = np.empty(self.K)
        frobenius_dists = np.empty(Ntest)
        for k in range(self.K):
            for n in range(Ntest):
                Sig = invwishartrand(
                    self.nu[k], self.Sigmalat[k, :, :]*(self.nu[k]-self.d-1))
                frobenius_dists[n] = np.linalg.norm(Sig - self.Sigmalat[k, :, :])
            percentiles[k] = np.percentile(frobenius_dists, q)
            print "\r Percentiles for {} out of {} computed".format(k+1, self.K),
        print ''
        percentile_dict[(q, Ntest)] = percentiles
        return percentiles

    def Sigma_outliers(self, q=99.99, Ntest=100000):  
        '''
            Find which components in which sample that have
            covariances (Sigma_jk values) that are outliers at a given
            percentile, i.e. that have a Frobenius distance to the
            latent covariance (Psi_k/(nu_k-d-1)) that is larger
            than the given percentile in a sample of distances 
            to the latent covariance matrix with matrices drawn
            from the prior model of Sigma_jk.
            
            q       - percentile
            Ntest   - number of random samples
            
            Returns (J X K) matrix.
        '''
        covd = self.get_cov_dist()
        outliers = (covd - self.Sigma_dist_percentiles(q, Ntest)[np.newaxis, :]) > 0
        return outliers

