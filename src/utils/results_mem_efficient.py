from __future__ import division
import numpy as np
import warnings
from sklearn import mixture as skmixture

import Bhattacharyya as bhat
import diptest
from plot_util import get_colors 

class LazyProperty(object):

    def __init__(self, func):
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls=None):
        if obj is None: 
            return None
        result = obj.__dict__[self.__name__] = self._func(obj)
        return result

def GMM_means_for_best_BIC(data,Ks,n_init=10,n_iter=100,covariance_type='full'):
    data = data[~np.isnan(data[:,0]),:]
    data = data[~np.isinf(data[:,0]),:] 
    bestbic = np.inf
    for K in Ks:
        g = skmixture.GMM(n_components=K,covariance_type=covariance_type,n_init=n_init,n_iter=n_iter)
        g.fit(data)
        bic = g.bic(data)
        print "BIC for {} clusters: {}".format(K,bic)
        if bic < bestbic:
            means = g.means_
            bestbic = bic
    if means.shape[0] == np.max(Ks):
        warnings.warn("Best BIC obtained for maximum K")
    if means.shape[0] == np.min(Ks):
        warnings.warn("Best BIC obtained for minimum K")
    return means

def lognorm_pdf(Y, mu, Sigma, p):

    Ycent = Y - mu
    return np.log(p) - np.log(np.linalg.det(Sigma))/2 - np.sum(Ycent*np.linalg.solve(Sigma,Ycent.T).T,axis=1)/2


def quantile(y,w,alpha):
    '''
        Returns alpha quantile(s) (alpha can be a vector) of empirical probability distribution of data y weighted by w
    '''
    alpha = np.array(alpha)
    alpha_ord = np.argsort(alpha)
    alpha_sort = alpha[alpha_ord]
    y_ord = np.argsort(y)
    y_sort = y[y_ord]
    cumdistr = np.cumsum(w[y_ord])
    q = np.empty(alpha.shape)
    if cumdistr[-1] == 0:
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

class Mres(object):
    
    def __init__(self,d,K,p,classif_freq,data,meta_data,p_noise=None,sim=None):

        self.d = d
        self.K = K # NB! Noise cluster not included in this
        self.J = len(data)
        self.p = p
        self.p_noise = p_noise

        self.meta_data = MetaData(meta_data)

        self.merged = False
        self.mergeMeth = ''
        self.mergeind = [[k] for k in range(self.K)]
        
        self.data = data
        #self.datapooled = np.vstack(self.data)
        self.J = len(data)
        if sim is None:
            sim = np.sum(classif_freq[0].tocsr().getrow(0))
        self.clusts = []
        for j in range(self.J):
            self.clusts.append(SampleClustering(data[j],classif_freq[j],self.mergeind,sim,K))
        #self.clust_nm = Clustering(self.data,classif_freq,self.p,self.p_noise)                 
    
    @property
    def p_merged(self):
        p_mer = np.zeros((self.J,len(self.mergeind)))
        for s,m_ind in enumerate(self.mergeind):
            p_mer[:,s] = sum([self.p[:,k] for k in m_ind])
        return p_mer

    def get_order(self):
        for s,suco in enumerate(self.mergeind):
            sc_ord = np.argsort(-np.array(np.sum(self.p,axis=0)[suco]))
            self.mergeind[s] = [suco[i] for i in sc_ord] # Change order within each super component
        prob_mer = [np.sum(self.p[:,scind]) for scind in self.mergeind]
        suco_ord = np.argsort(-np.array(prob_mer))
        mergeind_sort = [self.mergeind[i] for i in suco_ord]
        print "mergeind_sort = {}".format(mergeind_sort)
        comp_ord = [ind for suco in mergeind_sort for ind in suco]
        return comp_ord,suco_ord

    def get_colors_and_order(self,maxnbrsucocol = 8):
        comp_ord,suco_ord = self.get_order()
        comp_col,suco_col = get_colors(self.mergeind,suco_ord,comp_ord,maxnbrsucocol)      
        return comp_col,suco_col,comp_ord,suco_ord    
            
    def merge(self,method,thr,**mmfArgs):
        self.complist = [[k] for k in range(self.K)]
        if method == 'demp':
            self.hierarchical_merge(self.get_median_overlap,thr,**mmfArgs)
            #self.hclean()
        elif method == 'bhat_hier':
            self.hierarchical_merge(self.get_median_bh_dist_data,thr,**mmfArgs)
            #self.hclean()
        elif method == 'bhat_hier_dip':
            lowthr = mmfArgs.pop('lowthr')
            dipthr = mmfArgs.pop('dipthr')
            print "Merging components with median Bhattacharyya overlap at least {}".format(thr)
            self.hierarchical_merge(self.get_median_bh_dist_data,thr,**mmfArgs)
            print "Merging components with median Bhattacharyya overlap at least {} and robust dip test at least {}".format(lowthr,dipthr)
            self.hierarchical_merge(self.get_median_bh_dt_dist_dip,thr=lowthr,bhatthr=lowthr,dipthr=dipthr,**mmfArgs)
            #self.hclean()
        elif method == 'no_merging':
            pass
        else:
            raise NameError, 'Unknown method for merging'
        del self.complist
        self.merged = True
        self.mergeMeth = method
        self.mergeThr = thr
        self.mergeArgs = mmfArgs

    def hierarchical_merge(self,mergeMeasureFun,thr,**mmfArgs):
        if (self.p < 0).any():
            raise ValueError, 'Negative p'
        mm = mergeMeasureFun(**mmfArgs)
        ind = np.unravel_index(np.argmax(mm),mm.shape)
        if mm[ind] > thr:
            print "Threshold passed at value {} for {} and {}, merging".format(mm[ind],self.mergeind[ind[0]],self.mergeind[ind[1]])
            self.merge_kl(ind)
            self.hierarchical_merge(mergeMeasureFun,thr,**mmfArgs)

    # def hierarchical_merge(self,mergeMeasureFun,thr,**mmfArgs):
    #     if (self.p < 0).any():
    #         raise ValueError, 'Negative p'
    #     mm = mergeMeasureFun(**mmfArgs)
    #     if (mm > thr).any():
    #         self.merge_kl(np.unravel_index(np.argmax(mm),mm.shape))
    #         self.hierarchical_merge(mergeMeasureFun,thr,**mmfArgs)
            
    def merge_kl(self,ind):
        self.mergeind[ind[1]] += self.mergeind[ind[0]]
        self.mergeind.pop(ind[0])
        #self.complist[ind[1]] = [self.complist[ind[1]],self.complist[ind[0]]]
        #self.complist[ind[0]] = []

    def greedy_merge(self,mergeMeasureFun,thr,**mmfKw):
        mm = mergeMeasureFun(**mmfKw)
        suco_assign = -np.ones(mm.shape[0],dtype='int')
        sucos = []
        while np.amax(mm) > thr:
            nextmerge = np.unravel_index(np.argmax(mm),mm.shape)
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
            mm[nextmerge[0],nextmerge[1]] = -np.inf
            mm[nextmerge[1],nextmerge[0]] = -np.inf
        # Empty mergeind list
        while len(self.mergeind) > 0:
            self.mergeind.pop()
        # Add new content
        self.mergeind += [sucos[s] for s in np.unique(suco_assign[suco_assign > -1])]
        self.mergeind += [[k] for k in np.nonzero(suco_assign == -1)[0]]

    def get_median_overlap(self,fixvalind=[],fixval=-1):
        overlap = []
        for clust in self.clusts:
            overlap.append(clust.get_overlap())
        return self.get_medprop_pers(overlap,fixvalind,fixval)
    
    def get_median_bh_dist_data(self,fixvalind=[],fixval=-1):
        bhd = []
        for clust in self.clusts:
            bhd.append(clust.get_bh_dist_data())
        #print "median bhattacharyya distance overlap = {}".format(get_medprop_pers(bhd,fixvalind,fixval))
        return self.get_medprop_pers(bhd,fixvalind,fixval)

    def get_median_bh_dt_dist_dip(self,bhatthr,dipthr,fixvalind=[],fixval=-1):
        mbhd = self.get_median_bh_dist_data(fixvalind,fixval)
        while (mbhd > bhatthr).any():
            ind = np.unravel_index(np.argmax(mbhd),mbhd.shape)
            print "Dip test for {} and {}".format(*ind)
            if self.okdiptest(ind,dipthr):
                return mbhd
            fixvalind.append(ind)
            mbhd = self.get_median_bh_dist_data(fixvalind,fixval)
        return mbhd  

    def okdiptest(self,ind,thr):
        k,l = ind
        for dim in [None]+range(self.d):
            nbr_computable = self.J
            below = 0
            for j,clust in enumerate(self.clusts):
                try:
                    if clust.get_pdip_discr_jkl(k,l,dim) < thr:
                        below += 1
                except EmptyClusterException:
                    nbr_computable -= 1

                if below > np.floor(nbr_computable):
                    print "For ind {} and {}, diptest failed for dim: {}".format(k,l,dim)
                    return False
        print "Diptest ok for {} and {}".format(k,l)
        return True

    def get_pdip(self,suco=True):
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
            pdip = np.zeros((self.J,self.d))
            for j,clust in enumerate(self.clusts):
                pdip[j,:] = clust.get_pdip(k)
            print "Diptest computed for component {}".format(k)
            pdiplist.append(np.copy(pdip))           
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
            pdip = np.zeros((self.J,self.d))
            for j,clust in enumerate(self.clusts):
                pdip[j,:] = clust.cluster[k].get_pdip()
            print "Diptest computed for component {}".format(k)
            pdiplist.append(np.copy(pdip))           
        self.pdiplist_comp = pdiplist
        return self.pdiplist_comp

    def get_pdip_summary(self,suco=True):
        '''
            Medians, 25th percentiles and minima of diptest p-values for each cluster/component and each data dimension
        '''
        pdiplist = self.get_pdip(suco)
            
        d = self.d
        K = len(pdiplist)
        pdipsummary = {'Median': np.empty((K,d)), '25th percentile': np.empty((K,d)), 'Minimum': np.empty((K,d))}
        for k in range(K):
            pdk = pdiplist[k][~np.isnan(pdiplist[k][:,0]),:]
            if len(pdk) == 0:
                pdipsummary['Median'][k,:] = np.nan
                pdipsummary['25th percentile'][k,:] = np.nan
                pdipsummary['Minimum'][k,:] = np.nan
            else:
                pdipsummary['Median'][k,:] = np.median(pdk,0)
                pdipsummary['25th percentile'][k,:] = np.percentile(pdk,25,0)
                pdipsummary['Minimum'][k,:] = np.min(pdk,0)
        return pdipsummary

    @staticmethod
    def get_medprop_pers(prop,fixvalind=[],fixval=-1):
        med_prop = np.empty(prop[0].shape)
        for k in range(med_prop.shape[0]):
            for l in range(med_prop.shape[1]):
                prop_kl = np.array([pr[k,l] for pr in prop])
                med_prop[k,l] = np.median(prop_kl[~np.isnan(prop_kl)])
                if np.isnan(med_prop[k,l]):
                    med_prop[k,l] = fixval
    
        for ind in fixvalind:
            if len(ind) == 1:
                med_prop[ind,:] = fixval
            else:
                med_prop[ind[0],ind[1]] = fixval
                med_prop[ind[1],ind[0]] = fixval
        return med_prop

class MetaData(object):
    
    def __init__(self,meta_data):
        self.samp = meta_data['samp']
        self.marker_lab = meta_data['marker_lab']

    def sort(self,names):
        self.order = []
        for name in names:
            self.order.append(self.samp['names'].index(name))
        for key in self.samp:
            self.samp[key] = [self.samp[key][j] for j in self.order]
        
class Traces(object):
    '''
        Object containing information for traceplots.
    '''
    
    def __init__(self,bmlog_burn,bmlog_prod):
        self.saveburn = bmlog_burn.theta_sim.shape[0]#bmlog_burn.nbrsave
        self.saveprod = bmlog_prod.theta_sim.shape[0]#bmlog_prod.nbrsave
        self.savefrqburn = bmlog_burn.savefrq
        self.savefrqprod = bmlog_prod.savefrq
        
        self.burnind = np.arange(1,self.saveburn+1)*self.savefrqburn
        self.prodind = self.burnind[-1] + np.arange(1,self.saveprod+1)*self.savefrqprod
        self.ind = np.hstack([self.burnind,self.prodind])
        
        self.mulat_burn = bmlog_burn.theta_sim
        self.mulat_prod = bmlog_prod.theta_sim
        self.nu_burn = bmlog_burn.nu_sim
        self.nu_prod = bmlog_prod.nu_sim
        
        self.K = self.mulat_burn.shape[1]
        
    def get_mulat_k(self,k):
        return np.vstack([self.mulat_burn[:,k,:],self.mulat_prod[:,k,:]])
        
    def get_nu(self):
        return np.vstack([self.nu_burn,self.nu_prod])

class FCsample(object):
    '''
        Object containing results for synthetic data sample or corresponding real sample.
    '''
    
    def __init__(self,data,name):
        self.data = data
        self.name = name
        
    def get_data(self,N = None):
        if N is None:
            return self.data
        if N > self.data.shape[0]:
            warnings.warn('Requested sample is larger than data')
            N = self.data.shape[0]
        ind = np.random.choice(range(self.data.shape[0]),N,replace=False)
        return self.data[ind,:]
            
        
class SynSample(FCsample):
    '''
        Object containing results for synthetic data sample.
    '''

    def __init__(self,syndata,genname,fcmimic):
        super(SynSample,self).__init__(syndata,fcmimic.name)
        self.genname = genname
        self.realsize = fcmimic.data.shape[0]
        
    def get_data(self,N = None):
        if N is None:
            N = self.realsize
        return super(SynSample,self).get_data(N)

class MimicSample(object):
    '''
        Object containing results for synthetic data sample and corresponding real sample.
    '''

    def __init__(self,data,name,syndata,modelname):
        self.realsamp = FCsample(data,name)
        self.synsamp = SynSample(syndata,modelname,self.realsamp)            

class Cluster(object):

    def __init__(self,classif_freq,data,sim):
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
        return np.dot(self.weights,self.data[self.indices,:]).reshape(1,-1)

    @LazyProperty
    def wXXT(self):
        wXXT = np.zeros((self.data.shape[1],self.data.shape[1]))
        for i,ind in enumerate(self.indices):
            x = self.data[ind,:].reshape(1,-1)
            wXXT += self.weights[i]*x.T.dot(x)
        return wXXT

    def get_pdip(self,suco=True,dims=None):
        if dims is None:
            dims = range(self.data.shape[1])
        pdip = np.zeros(len(dims))
        for i,dd in enumerate(dims):
            xcum,ycum = diptest.cum_distr(self.data[self.indices,dd],self.weights/self.W)
            dip = diptest.dip_from_cdf(xcum,ycum)
            pdip[i] = diptest.dip_pval_tabinterpol(dip,self.W)
        return pdip
        
class SampleClustering(object):
    '''
        Object containing information about clustering of the data.
    '''
    
    def __init__(self,data,classif_freq,mergeind,sim,K):
        self.data = data
        self.mergeind = mergeind
        self.sim = sim
        self.K = K

        classif_freq = classif_freq.tocsc() # transform sparse matrix to enable column slicing
        self.clusters = []
        for k in range(self.K):
            self.clusters.append(Cluster(classif_freq.getcol(k),data,sim))
        self.d = data.shape[1]
        #self.vacuous_komp = np.vstack([np.sum(clf,axis=0) < 3 for clf in self.classif_freq])        
 
    @property
    def x_sample(self):
        N = self.data.shape[0]
        x = self.K*np.ones(N)
        cum_freq = np.zeros(N)
        alpha = np.random.random(N)
        notfound = np.arange(N)
        for k,cluster in enumerate(self.clusters):
            cum_freq += cluster.classif_freq.toarray().reshape(-1)
            newfound_bool = alpha[notfound] < cum_freq[notfound]
            newfound = notfound[newfound_bool]
            x[newfound] = k
            notfound = notfound[~newfound_bool]
        return x

    def get_mean(self,s):
        ks = self.mergeind[s]
        return sum([self.clusters[k].wX for k in ks])/sum([self.clusters[k].W for k in ks])

    def get_scatter(self,s):
        ks = self.mergeind[s]
        try:
            wXXT = sum([self.clusters[k].wXXT for k in ks])/sum([self.clusters[k].W for k in ks])
        except ZeroDivisionError:
            return np.nan*self.clusters[0].wXXT
        mu = self.get_mean(s).reshape(1,-1)
        return wXXT - mu.T.dot(mu)

    def get_W(self,s):
        ks = self.mergeind[s]
        return sum([self.clusters[k].W for k in ks])

    def get_classif_freq(self,s):
        ks = self.mergeind[s]
        clf = sum([self.clusters[k].classif_freq for k in ks])
        return clf

    def get_pdip(self,k,dims=None):
        if dims is None:
            dims = range(self.data.shape[1])
        clf = self.get_classif_freq(k)
        W = self.get_W(k)
        pdip = np.zeros(len(dims))
        for i,dd in enumerate(dims):
            xcum,ycum = diptest.cum_distr(self.data[clf.indices,dd],clf.data/W)
            dip = diptest.dip_from_cdf(xcum,ycum)
            pdip[i] = diptest.dip_pval_tabinterpol(dip,W)
                           
    def get_overlap(self):
        '''
            Estimated misclassification probability between two super clusters, i.e.
            probability that Y_i is classified as belonging to l when it
            truly belongs to k.
        '''
        S = len(self.mergeind)
        overlap = np.zeros((S,S))
        for k in range(S):
            for l in range(S):
                overlap[k,l] = self.get_classif_freq(k).T.dot(self.get_classif_freq(l)).todense()
            overlap[k,:] /= self.get_W(k)
            overlap[k,k] = 0
        return overlap
        
    def get_bh_dist_data(self):
        S = len(self.mergeind)
        bhd = -np.ones((S,S))
        mus = [self.get_mean(k) for k in range(S)]
        Sigmas = [self.get_scatter(k) for k in range(S)]
        for k in range(S):
            for l in range(S):
                if l != k:
                    bhd[k,l] = bhat.bhattacharyya_dist(mus[k],Sigmas[k],mus[l],Sigmas[l])

            bhd[k,k] = 0
            #print "nbr nan in bhd[j]: {}".format(np.sum(np.isnan(bhd[j])))
            #print "nbr not nan in bhd[j]: {}".format(np.sum(~np.isnan(bhd[j])))                
        return bhd
  
    def get_pdip_discr_jkl(self,k,l,dim=None):
        '''
            p-value of diptest of unimodality for the merger of super cluster k and l
            
            Input:
                dim     - dimension along which the test should be performed. If dim is None, the test will be performed on the projection onto Fisher's discriminant coordinate.
        '''
        clf_k = self.get_classif_freq(k)
        clf_l = self.get_classif_freq(l)
        clf = clf_k + clf_l
        W_k = self.get_W(k)
        W_l = self.get_W(l)
        W = W_k + W_l

        if dim is None:
            if W_k == 0 or W_l == 0:
                raise EmptyClusterException
            dataproj = self.discriminant_projection(k,l)
        else:
            if W == 0:
                raise EmptyClusterException             
            dataproj = self.data[:,dim]

        xcum,ycum = diptest.cum_distr(dataproj[clf.indices],clf.data/W)
        dip = diptest.dip_from_cdf(xcum,ycum)
        return diptest.dip_pval_tabinterpol(dip,W)

    def discriminant_projection(self,s1,s2):
        '''
            Projection of a data set onto Fisher's discriminant coordinate between two super clusters.
            
        '''
        mu1,Sigma1 = self.get_mean(s1).T,self.get_scatter(s1)
        mu2,Sigma2 = self.get_mean(s2).T,self.get_scatter(s2)
        dc = self.discriminant_coordinate(mu1,Sigma1,mu2,Sigma2)
        proj = np.dot(self.data,dc)
        return proj        

    @staticmethod
    def discriminant_coordinate(mu1,Sigma1,mu2,Sigma2):
        '''
            Fisher's discriminant coordinate
        '''
        w = np.linalg.solve(Sigma1+Sigma2,mu2-mu1)
        w /= np.linalg.norm(w)
        return w

    # def sample_x(self,j):
    #     N = self.data[j].shape[0]
    #     x = np.zeros(N)
    #     cum_freq = np.cumsum(self.classif_freq[j],axis=1)
    #     alpha = np.random.random(N)*cum_freq[0,-1]
    #     notfound = np.arange(N)
    #     for i in range(self.classif_freq[j].shape[1]):
    #         newfound_bool = alpha[notfound] < cum_freq[notfound,i]
    #         newfound = notfound[newfound_bool]
    #         x[newfound] = i
    #         notfound = notfound[~newfound_bool]
    #     return x

    # def get_quantiles(self,alpha,j=None,ks=None,dds=None):
    #     '''
    #         Returns alpha quantile(s) in each dimension of sample j (the pooled data if j = None) for each
    #         of the clusters.
    #     '''
    #     if j is None:    
    #         clf = np.vstack(self.classif_freq)
    #         data = np.vstack(self.data)
    #     else:
    #         clf = self.classif_freq[j]
    #         data = self.data[j]
            
    #     if ks is None:
    #         ks = range(self.K)
    #     if dds is None:
    #         dds = range(self.d)
            
    #     weights_all = clf/sum(clf[0,:])
    #     quantiles = np.zeros((len(ks),len(alpha),len(dds)))
    #     for ik,k in enumerate(ks):
    #         for id,dd in enumerate(dds):
    #             quantiles[ik,:,id] = quantile(data[:,dd],weights_all[:,k],alpha)
    #     return quantiles

    # def get_data_kdj(self,min_clf,k,dd,j=None):
    #     '''
    #         Get data points belonging to a certain cluster

    #         min_clf	-	min classification frequency for the point into the given cluster
    #         k		-	cluster number
    #         dd		-	dimonsion for which data should be returned
    #         j		- 	sample nbr
    #     '''
    #     if j is None:
    #         data = np.vstack(self.data)[:,dd]
    #         clf = np.vstack(self.classif_freq)[:,k]/sum(self.classif_freq[0][0,:])
    #     else:
    #         data = self.data[j][:,dd]
    #         clf = self.classif_freq[j][:,k]/sum(self.classif_freq[0][0,:])
    #     return data[clf > min_clf]
            
class EmptyClusterException(Exception):
    pass

class Components(object):
    '''
        Object containing information about mixture components
    '''
    
    def __init__(self,bmlog,p):
        self.J = bmlog.J
        self.K = bmlog.K
        self.d = bmlog.d
        self.mupers = bmlog.mupers_sim_mean
        self.Sigmapers = bmlog.Sigmapers_sim_mean
        self.mulat = bmlog.theta_sim_mean
        self.Sigmalat = bmlog.Sigmaexp_sim_mean
        self.p = p
        self.active_komp = bmlog.active_komp

    def new_thetas_from_GMM_fit(self,Ks=None,n_init=10,n_iter=100,covariance_type='full'):
        if Ks is None:
            Ks = range(int(self.K/4),self.K*2)
        allmus = np.vstack([self.mupers[j,:,:] for j in range(self.mupers.shape[0])])
        thetas = GMM_means_for_best_BIC(allmus,Ks,n_init,n_iter,covariance_type)
        self.new_thetas = thetas
        return thetas
        
    def classif(self,Y,j,p_noise = None,mu_noise=0.5*np.ones((1,4)),Sigma_noise = 0.5**2*np.eye(4)):
        '''
            Classify data points according to highest likelihood by mixture model.
            Input:
            
                Y (Nxd)     -   data
                j           -   sample number
                p_noise     -   noise component probability
                mu_noise    -   noise component mean
                Sigma_noise -   noise component covariance matrix
        '''
        if not p_noise is None:
            mus = np.vstack([self.mupers[j,:,:],mu_noise.reshape(1,self.d)])
            Sigmas = np.vstack([self.Sigmapers[j,:,:,:],Sigma_noise.reshape(1,self.d,self.d)])
            ps = np.vstack([self.p[j,:].reshape(-1,1),np.array(p_noise[j]).reshape(1,1)])
        else:
            mus = self.mupers[j,:,:]
            Sigmas = self.Sigmapers[j,:,:,:]
            ps = self.p[j,:]
            
        dens = np.empty((Y.shape[0],mus.shape[0]))
        for k in range(mus.shape[0]):
            dens[:,k] = lognorm_pdf(Y, mus[k,:], Sigmas[k,:,:], ps[k])
            #Ycent = Y - mus[k,:]
            #dens[:,k] = np.log(ps[k]) - np.log(np.linalg.det(Sigmas[k,:,:]))/2 - np.sum(Ycent*np.linalg.solve(Sigmas[k,:,:],Ycent.T).T,axis=1)/2
        dens[np.isnan(dens)] = -np.inf
        return np.argmax(dens,axis=1)

    def get_bh_dist(self):
        '''
            Get bhattacharyya distance between components.
        '''
        bhd = [np.empty((self.K,self.K)) for j in range(self.J)]
        for j in range(self.J):
            for k in range(self.K):
                if self.active_komp[j,k] > 0.05:
                    muk = self.mupers[j,k,:]
                    Sigmak = self.Sigmapers[j,k,:,:]
                else:
                    muk = self.mulat[k,:]
                    Sigmak = self.Sigmalat[k,:,:]
                for l in range(self.K):
                    if self.active_komp[j,l] > 0.05:
                        mul = self.mupers[j,l,:]
                        Sigmal = self.Sigmapers[j,l,:,:]
                    else:
                        mul = self.mulat[l,:]
                        Sigmal = self.Sigmalat[l,:,:]
                    bhd[j][k,l] = bhat.bhattacharyya_dist(muk,Sigmak,mul,Sigmal)
                bhd[j][k,k] = 0
        return bhd
        
    def get_median_bh_dist(self,fixvalind=[],fixval=-1):
        bhd = self.get_bh_dist()
        return self.get_medprop_pers(bhd,fixvalind,fixval)

    def get_center_distance(self):
        '''
            Get distance from component mean to latent component mean for each component in each sample.
        '''
        dist = np.zeros((self.J,self.K))
        for k in range(self.K):
            for j in range(self.J):
                if self.active_komp[j,k] <= 0.05:
                    dist[j,k] = np.nan
                else:
                    dist[j,k] = np.linalg.norm(self.mupers[j,k,:] - self.mulat[k,:])
        return dist
    
    def get_cov_dist(self,norm='F'):
        '''
            Get distance from component covariance matrix to latent covariance matrix for each component in each sample.
            
            norm    -   'F' gives Frobenius distance, 2 gives operator 2-norm.
        '''
        covdist = np.zeros((self.J,self.K))
        for j in range(self.J):
            for k in range(self.K):
                if self.active_komp[j,k] <= 0.05:
                    covdist[j,k] = np.nan
                else:
                    if norm == 'F':
                        covdist[j,k] = np.linalg.norm(self.Sigmapers[j,k,:,:]-self.Sigmalat[k,:])
                    elif norm == 2:
                        covdist[j,k] = np.linalg.norm(self.Sigmapers[j,k,:,:]-self.Sigmalat[k,:],ord=2)
        return covdist

        
    def get_center_distance_quotient(self):
        '''
            Get for each component in each sample, the quotient between the distance from the component mean to the correct latent component mean
            and the distance from the component mean to the closest latent component mean which has not been merged into the component.
        '''
        distquo = np.zeros((self.J,self.K))
        for suco in self.mergeind:
            for k in suco:
                otherind = np.array([not (kk in suco) for kk in range(self.K)])
                for j in range(self.J):
                    corrdist = np.linalg.norm(self.mupers[j,k,:] - self.mulat[k,:])
                    wrongdist = min(np.linalg.norm(self.mupers[j,[k]*sum(otherind),:] - self.mulat[otherind,:],axis = 1))
                    distquo[j,k] = wrongdist/corrdist
        return distquo



