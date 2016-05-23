from __future__ import division
import numpy as np
import warnings
import copy as cp
from sklearn import mixture as skmixture
import matplotlib.pyplot as plt

from . import Bhattacharyya as bhat
from . import diptest
from . import discriminant as discr
from .plot_util import get_colors

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

def GMM_means_for_best_BIC(data,Ks,n_init=10,n_iter=100,covariance_type='full'):
    data = data[~np.isnan(data[:,0]),:]
    data = data[~np.isinf(data[:,0]),:] 
    bestbic = np.inf
    for K in Ks:
        g = skmixture.GMM(n_components=K,covariance_type=covariance_type,n_init=n_init,n_iter=n_iter)
        g.fit(data)
        bic = g.bic(data)
        print("BIC for {} clusters: {}".format(K,bic))
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
    
    def __init__(self,d,K,p,classif_freq,data,meta_data,p_noise=None):

        self.d = d
        self.K = K # NB! Noise cluster not included in this
        self.J = len(data)
        self.p = p
        self.p_noise = p_noise

        self.meta_data = MetaData(meta_data)
        
        self.data = data
        self.datapooled = np.vstack(self.data)

        self.clust_nm = Clustering(self.data,classif_freq,self.p,self.p_noise)                 
        
        self.merged = False
        self.mergeMeth = ''
        self.mergeind = [[k] for k in range(self.K)]
        
    def get_order(self):
        for s,suco in enumerate(self.mergeind):
            sc_ord = np.argsort(-np.array(np.sum(self.p,axis=0)[suco]))
            self.mergeind[s] = [suco[i] for i in sc_ord] # Change order within each super component
        prob_mer = [np.sum(self.p[:,scind]) for scind in self.mergeind]
        suco_ord = np.argsort(-np.array(prob_mer))
        mergeind_sort = [self.mergeind[i] for i in suco_ord]
        print("mergeind_sort = {}".format(mergeind_sort))
        comp_ord = [ind for suco in mergeind_sort for ind in suco]
        return comp_ord,suco_ord

    def get_colors_and_order(self,maxnbrsucocol = 8):
        comp_ord,suco_ord = self.get_order()
        comp_col,suco_col = get_colors(self.mergeind,suco_ord,comp_ord,maxnbrsucocol)      
        return comp_col,suco_col,comp_ord,suco_ord    
            
    def merge(self,method,thr,**mmfArgs):

        self.clust_m = Clustering(self.data, cp.deepcopy(self.clust_nm.classif_freq),np.copy(self.clust_nm.p),self.p_noise)
        self.complist = [[k] for k in range(self.K)]
        if method == 'demp':
            self.hierarchical_merge(self.clust_m.get_median_overlap,thr,**mmfArgs)
            self.hclean()
        elif method == 'bhat_hier':
            self.hierarchical_merge(self.clust_m.get_median_bh_dt_overlap,thr,**mmfArgs)
            self.hclean()
        elif method == 'bhat_hier_dip':
            lowthr = mmfArgs.pop('lowthr')
            dipthr = mmfArgs.pop('dipthr')
            self.hierarchical_merge(self.clust_m.get_median_bh_dt_overlap,thr,**mmfArgs)
            self.hierarchical_merge(self.clust_m.get_median_bh_dt_overlap_dip,thr=lowthr,bhatthr=lowthr,dipthr=dipthr,**mmfArgs)
            self.hclean()
        elif method == 'no_merging':
            self.mergeind = [[k] for k in range(self.K)]
        else:
            raise NameError('Unknown method for merging')
        del self.complist
        self.merged = True
        self.mergeMeth = method
        self.mergeThr = thr
        self.mergeArgs = mmfArgs

    def hierarchical_merge(self,mergeMeasureFun,thr,**mmfArgs):
        if (self.p < 0).any():
            raise ValueError('Negative p')
        mm = mergeMeasureFun(**mmfArgs)
        if (mm > thr).any():
            self.merge_kl(np.unravel_index(np.argmax(mm),mm.shape))
            self.hierarchical_merge(mergeMeasureFun,thr,**mmfArgs)
            
    def merge_kl(self,ind):
        self.clust_m.merge_kl(ind)
        self.complist[ind[1]] = [self.complist[ind[1]],self.complist[ind[0]]]
        self.complist[ind[0]] = []

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
        self.mergeind = [sucos[s] for s in np.unique(suco_assign[suco_assign > -1])]
        self.mergeind += [[k] for k in np.nonzero(suco_assign == -1)[0]]

    def hclean(self):
        for i in range(self.complist.count([])):
            self.complist.remove([])
        self.mergeind = self.flatten_complist()
        self.clust_m.hclean()
        print("self.mergeind = {}".format(self.mergeind))
        
        #left_comp = np.array([suco[0] for suco in self.mergeind])
        #print "left_comp = {}".format(left_comp)

    def gclean(self):
        self.clust_m.gclean(self.mergeind)

    def flatten_complist(self):
        fllist  = []
        for suco in self.complist:
            suco = self.flatten_supercomp(suco)
            fllist.append(suco)
        #fllist = sort_sucolist(fllist)
        return fllist

    def flatten_supercomp(self,suco):
        i = 0
        while i < len(suco):
            if type(suco[i]) is list:
                suco  = suco[:i] + suco[i] + suco[(i+1):]
                suco = self.flatten_supercomp(suco)
            i += 1
        return suco

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
        self.saveburn = bmlog_burn.nbrsave
        self.saveprod = bmlog_prod.nbrsave
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
        
class Clustering(object):
    '''
        Object containing information about clustering of the data.
        Can either represent merged or non-merged data
    '''
    
    def __init__(self,data,classif_freq,p,p_noise=0):
        self.data = data
        self.classif_freq = classif_freq
        self.sim = sum(self.classif_freq[0][0,:])
        self.p = p
        self.p_noise = p_noise
        self.J = len(classif_freq)
        self.K = p.shape[1]
        self.d = data[0].shape[1]
        self.vacuous_komp = np.vstack([np.sum(clf,axis=0) < 3 for clf in self.classif_freq])        
        
    def merge_kl(self,ind):
        for cl in self.classif_freq:
            cl[:,ind[1]] += cl[:,ind[0]]
            cl[:,ind[0]] = 0
        self.p[:,ind[1]] += self.p[:,ind[0]]
        self.p[:,ind[0]] = 2 #Avoiding division by 0
        
    def hclean(self):
        removed_comp = self.p[0,:] == 2
        #print "removed_comp.shape = {}".format(removed_comp.shape)
        #print "np.hstack([removed_comp,True]) = {}".format(np.hstack([removed_comp,True]))
        if self.p.shape[1] < self.classif_freq[0].shape[1]:
            removed_comp_freq = np.hstack([removed_comp,False])
        else:
            removed_comp_freq = removed_comp
        self.classif_freq = [clf[:,~removed_comp_freq] for clf in self.classif_freq]
        self.p = self.p[:,~removed_comp]
        self.K = self.p.shape[1]
        print("Sizes of merged clusters: {}".format(self.p))
        
    def gclean(self,mergeind):
        newp = np.empty((self.J,len(mergeind)))
        new_clf = [np.empty((clf.shape[0],len(mergeind))) for clf in self.classif_freq]
        for k,co in enumerate(mergeind):
            newp[:,k] = np.sum(self.p[:,np.array(co)],axis=1)
            for j,clf in enumerate(self.classif_freq):
                new_clf[j][:,k] = np.sum(clf[:,np.array(co)],axis=1)
        self.p = newp
        self.classif_freq = new_clf
        self.K = self.p.shape[1]

    def get_median_overlap(self,fixvalind=[],fixval=-1):
        overlap = self.get_overlap()
        return get_medprop_pers(overlap,fixvalind,fixval)
    
    def get_median_bh_dt_overlap(self,fixvalind=[],fixval=-1):
        bhd = self.get_bh_overlap_data()
        #print "median bhattacharyya distance overlap = {}".format(get_medprop_pers(bhd,fixvalind,fixval))
        return get_medprop_pers(bhd,fixvalind,fixval)

    def get_median_bh_dt_overlap_dip(self,bhatthr,dipthr,fixvalind=[],fixval=-1):
        bhd = self.get_bh_overlap_data()
        mbhd = get_medprop_pers(bhd,fixvalind,fixval)
        while (mbhd > bhatthr).any():
            ind = np.unravel_index(np.argmax(mbhd),mbhd.shape)
            print("Dip test for {} and {}".format(*ind))
            if self.okdiptest(ind,dipthr):
                return mbhd
            fixvalind.append(ind)
            mbhd = get_medprop_pers(bhd,fixvalind,fixval)
        return mbhd        
                        
    def get_overlap(self):
        overlap = [np.zeros((self.K,self.K)) for cl in self.classif_freq]
        for j,cl in enumerate(self.classif_freq):
            for k in range(self.K):
                for l in range(self.K):
                    overlap[j][k,l] += np.mean(np.prod(cl[:,(k,l)],1))
                overlap[j][k,:] /= self.p[j,k]
                overlap[j][k,k] = 0
            overlap[j] /= self.sim**2
        return overlap
        
    def get_bh_overlap_data(self):
        bhd = [-np.ones((self.K,self.K)) for j in range(self.J)]
        for j in range(self.J):
            for k in range(self.K):
                if (self.p[:,k] < 1.5).any():
                    if np.sum(self.classif_freq[j][:,k] > .5) < self.d:
                        # Not enough data points to estimate distance
                        bhd[j][k,:] = np.nan
                    else:
                        muk,Sigmak = discr.population_mu_Sigma(self.data[j],self.classif_freq[j][:,k])
                        for l in range(self.K):
                            if l != k and (self.p[:,l] < 1.5).any():
                                if np.sum(self.classif_freq[j][:,l] > .5) < self.d:
                                    # Not enough data points to estimate distance
                                    bhd[j][k,l] = np.nan
                                else:
                                    mul,Sigmal = discr.population_mu_Sigma(self.data[j],self.classif_freq[j][:,l])
                                    #print "muk = {}".format(muk)
                                    #print "mul = {}".format(mul)
                                    #print "Sigmak = {}".format(Sigmak)
                                    #print "Sigmal = {}".format(Sigmal)
                                    bhd[j][k,l] = bhat.bhattacharyya_overlap(muk,Sigmak,mul,Sigmal)   
                bhd[j][k,k] = 0
            #print "nbr nan in bhd[j]: {}".format(np.sum(np.isnan(bhd[j])))
            #print "nbr not nan in bhd[j]: {}".format(np.sum(~np.isnan(bhd[j])))                
        return bhd
        
    def get_quantiles(self,alpha,j=None,ks=None,dds=None):
        '''
            Returns alpha quantile(s) in each dimension of sample j (the pooled data if j = None) for each
            of the clusters.
        '''
        if j is None:    
            clf = np.vstack(self.classif_freq)
            data = np.vstack(self.data)
        else:
            clf = self.classif_freq[j]
            data = self.data[j]
            
        if ks is None:
            ks = range(self.K)
        if dds is None:
            dds = range(self.d)
            
        weights_all = clf/sum(clf[0,:])
        quantiles = np.zeros((len(ks),len(alpha),len(dds)))
        for ik,k in enumerate(ks):
            for id,dd in enumerate(dds):
                quantiles[ik,:,id] = quantile(data[:,dd],weights_all[:,k],alpha)
        return quantiles

    def get_data_kdj(self,min_clf,k,dd,j=None):
        '''
            Get data points belonging to a certain cluster

            min_clf	-	min classification frequency for the point into the given cluster
            k		-	cluster number
            dd		-	dimonsion for which data should be returned
            j		- 	sample nbr
        '''
        if j is None:
            data = np.vstack(self.data)[:,dd]
            clf = np.vstack(self.classif_freq)[:,k]/sum(self.classif_freq[0][0,:])
        else:
            data = self.data[j][:,dd]
            clf = self.classif_freq[j][:,k]/sum(self.classif_freq[0][0,:])
        return data[clf > min_clf]

    def get_pdip(self):
        '''
            Diptest p-values for each cluster, each dimension and each sample.
        '''
       
        if hasattr(self,'pdiplist'):
            return self.pdiplist

        pdiplist = []
        for k in range(self.K):
            pdip = np.zeros((self.J,self.d))
            for j in range(self.J):
                for dd in range(self.d):
                    xcum,ycum = diptest.cum_distr(self.data[j][:,dd],self.classif_freq[j][:,k])
                    dip = diptest.dip_from_cdf(xcum,ycum)
                    pdip[j,dd] = diptest.dip_pval_tabinterpol(dip,self.p[j,k]*self.classif_freq[j].shape[0])
            print("Diptest computed for component {}".format(k))
            pdiplist.append(np.copy(pdip))
            
        self.pdiplist = pdiplist
        return self.pdiplist
        
    def get_pdip_k(self,k):
        '''
            Diptest p-values for one cluster
        '''
        if not hasattr(self,'pdiplist'):
            self.get_pdip()
        return self.pdiplist[k]
            
    def get_pdip_summary(self):
        '''
            Medians, 25th percentiles and minima of diptest p-values for each cluster/component and each data dimension
        '''
        if not hasattr(self,'pdiplist'):
            self.get_pdip()
            
        d = self.d
        K = len(self.pdiplist)
        pdipsummary = {'Median': np.empty((K,d)), '25th percentile': np.empty((K,d)), 'Minimum': np.empty((K,d))}
        for k in range(K):
            pdk = self.pdiplist[k][~np.isnan(self.pdiplist[k][:,0]),:]#np.ma.masked_array(self.pdiplist[k],np.isnan(self.pdiplist[k]))
            if len(pdk) == 0:
                pdipsummary['Median'][k,:] = np.nan
                pdipsummary['25th percentile'][k,:] = np.nan
                pdipsummary['Minimum'][k,:] = np.nan
            else:
                pdipsummary['Median'][k,:] = np.median(pdk,0)
                pdipsummary['25th percentile'][k,:] = np.percentile(pdk,25,0)
                pdipsummary['Minimum'][k,:] = np.min(pdk,0)
        return pdipsummary
    
    def get_pdip_discr_jkl(self,j,k,l,dim=None):
        '''
            p-value of diptest of unimodality for the merger of cluster k and l in flow cytometry sample j
            
            Input:
                dim     - dimension along which the test should be performed. If dim is None, the test will be performed on the projection onto Fisher's discriminant coordinate.
        '''
        if dim is None:
            dataproj = discr.discriminant_projection(self.data[j],self.classif_freq[j][:,k],self.classif_freq[j][:,l])
        else:
            dataproj = self.data[j][:,dim]
        xcum,ycum = diptest.cum_distr(dataproj,self.classif_freq[j][:,k] + self.classif_freq[j][:,l])
        dip = diptest.dip_from_cdf(xcum,ycum)
        return diptest.dip_pval_tabinterpol(dip,(self.p[j,k]+self.p[j,l])*self.classif_freq[j].shape[0])
        
    def okdiptest(self,ind,thr):
        if not hasattr(self,'vacuous_komp'):
            self.vacuous_komp = np.vstack([np.sum(clf,axis=0) < 3 for clf in self.classif_freq])
            print("vacuous_komp = {}".format(self.vacuous_komp))
        maxbelow = np.ceil((self.J-sum(self.vacuous_komp[:,ind[0]]+self.vacuous_komp[:,ind[1]]))/4) - 1
        print("maxbelow = {}".format(maxbelow))
        for dim in [None]+range(self.d):
            below = 0
            for j in range(self.J):
                if not (self.vacuous_komp[j,ind[0]] or self.vacuous_komp[j,ind[1]]):
                    if self.get_pdip_discr_jkl(j,ind[0],ind[1],dim) < thr:
                        below += 1
                        if below > maxbelow:
                            print("For ind {} and {}, diptest failed for dim: {}".format(ind[0],ind[1],dim))
                            return False
        print("Diptest ok for {} and {}".format(ind[0],ind[1]))
        return True
        
    def sample_x(self,j):
        N = self.data[j].shape[0]
        x = np.zeros(N)
        cum_freq = np.cumsum(self.classif_freq[j],axis=1)
        alpha = np.random.random(N)*cum_freq[0,-1]
        notfound = np.arange(N)
        for i in range(self.classif_freq[j].shape[1]):
            newfound_bool = alpha[notfound] < cum_freq[notfound,i]
            newfound = notfound[newfound_bool]
            x[newfound] = i
            notfound = notfound[~newfound_bool]
        return x
            

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

    def get_bh_overlap(self):
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
                    bhd[j][k,l] = bhat.bhattacharyya_overlap(muk,Sigmak,mul,Sigmal)
                bhd[j][k,k] = 0
        return bhd
        
    def get_median_bh_overlap(self,fixvalind=[],fixval=-1):
        bhd = self.get_bh_overlap()
        return get_medprop_pers(bhd,fixvalind,fixval)

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
